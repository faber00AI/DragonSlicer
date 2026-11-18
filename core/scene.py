from dataclasses import dataclass, field, replace
import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def _rot(rx, ry, rz):
    m = trimesh.transformations.euler_matrix(np.radians(rx), np.radians(ry), np.radians(rz), "sxyz")
    return m


@dataclass
class SceneObject:
    name: str
    mesh: trimesh.Trimesh
    x: float = 0.0
    y: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    sz: float = 1.0
    lift: float = 0.0
    text: dict | None = None            # parameters of a 3D text object (core.text.make_text), so it can be edited later
    sup: trimesh.Trimesh | None = None  # this object's supports, in the frame of matrix() (they follow move/scale; rotation drops them)
    _cache: trimesh.Trimesh | None = field(default=None, repr=False)
    _bounds: tuple | None = field(default=None, repr=False)
    _hull: tuple | None = field(default=None, repr=False)      # (mesh, convex-hull points, centre): exact bounds, cheaply

    @property
    def scale(self) -> float:
        """Uniform scale (geometric mean when the object is stretched non-uniformly)."""
        return float((self.sx * self.sy * self.sz) ** (1 / 3))

    @scale.setter
    def scale(self, v: float):
        self.sx = self.sy = self.sz = float(v)

    @property
    def uniform(self) -> bool:
        return abs(self.sx - self.sy) < 1e-9 and abs(self.sy - self.sz) < 1e-9

    def rotation(self) -> np.ndarray:
        return _rot(self.rx, self.ry, self.rz)[:3, :3]

    def invalidate(self):
        self._cache = None

    def _points(self):
        h = self._hull
        if h is None or h[0] is not self.mesh:
            v = np.asarray(self.mesh.vertices, float)
            try:
                pts = v[ConvexHull(v).vertices]
            except Exception:                                   # flat / degenerate: use every vertex
                pts = v
            h = self._hull = (self.mesh, pts, self.mesh.bounds.mean(axis=0))
        return h

    def matrix(self) -> np.ndarray:
        """4x4 local->world transform (local = the stored mesh); same result as world_mesh() without touching vertices."""
        _, pts, c = self._points()
        A = _rot(self.rx, self.ry, self.rz) @ np.diag([self.sx, self.sy, self.sz, 1.0])
        A[:3, 3] = -A[:3, :3] @ c
        minz = (pts @ A[2, :3]).min() + A[2, 3]
        A[:3, 3] += [self.x, self.y, -minz + self.lift]
        return A

    def bounds(self) -> np.ndarray:
        """World bounding box [[min],[max]], computed from the hull points (fast, exact)."""
        _, pts, _ = self._points()
        key = (id(self.mesh), self.x, self.y, self.rx, self.ry, self.rz, self.sx, self.sy, self.sz, self.lift)
        if self._bounds is not None and self._bounds[0] == key:
            return self._bounds[1]
        A = self.matrix()
        w = pts @ A[:3, :3].T + A[:3, 3]
        out = np.array([w.min(axis=0), w.max(axis=0)])
        self._bounds = (key, out)
        return out

    def fits(self, machine, tol: float = 0.01) -> bool:
        """True when the whole object is inside the print area (plate centred on the origin, height included)."""
        b = self.bounds()
        hx, hy = machine.bed_x / 2 + tol, machine.bed_y / 2 + tol
        return bool(b[0][0] >= -hx and b[1][0] <= hx and b[0][1] >= -hy and b[1][1] <= hy
                    and b[1][2] <= machine.bed_z + tol)

    def world_mesh(self) -> trimesh.Trimesh:
        if self._cache is None:
            m = self.mesh.copy()
            m.apply_translation(-m.bounds.mean(axis=0))
            m.apply_transform(np.diag([self.sx, self.sy, self.sz, 1.0]))
            m.apply_transform(_rot(self.rx, self.ry, self.rz))
            m.apply_translation([self.x, self.y, -m.bounds[0][2] + self.lift])
            self._cache = m
        return self._cache

    def lay_flat(self) -> bool:
        """Turn the object so that its largest flat face on which it can stand stably lies on the plate (rotation only:
        mesh and scale are kept, the new rotation is composed with the current one). Returns False if no face is found."""
        try:
            hull = self.world_mesh().convex_hull
            com = np.asarray(hull.centroid, float)
            facets = [np.asarray(f) for f in hull.facets]
            used = set(int(i) for f in facets for i in f)
            groups = [(f, float(a), np.asarray(n, float)) for f, a, n in zip(facets, hull.facets_area, hull.facets_normal)]
            for i in range(len(hull.faces)):                                    # faces that are not part of a larger facet
                if i not in used:
                    groups.append((np.array([i]), float(hull.area_faces[i]), np.asarray(hull.face_normals[i], float)))
            best = best_any = None
            for f, area, n in sorted(groups, key=lambda g: -g[1]):
                if best_any is None:
                    best_any = n
                if self._stands(hull, f, n, com):
                    best = n
                    break
            n = best if best is not None else best_any
            if n is None:
                return False
            R = trimesh.geometry.align_vectors(n, [0.0, 0.0, -1.0])[:3, :3]     # outward normal of the face -> straight down
            M = np.eye(4)
            M[:3, :3] = R @ self.rotation()
            self.rx, self.ry, self.rz = np.degrees(trimesh.transformations.euler_from_matrix(M, "sxyz")).tolist()
            self.lift = 0.0
            self.invalidate()
            return True
        except Exception:
            return False

    @staticmethod
    def _stands(hull, faces, n, com) -> bool:
        """True if the centre of mass, dropped along -n, lands inside the facet (so the object does not tip over)."""
        pts = np.unique(hull.vertices[np.unique(hull.faces[faces].ravel())], axis=0)
        u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        P = np.c_[pts @ u, pts @ v]
        q = np.array([com @ u, com @ v])
        try:
            h = ConvexHull(P).vertices
        except Exception:
            return False
        poly = P[h]                                                             # counter-clockwise
        e = np.roll(poly, -1, axis=0) - poly
        d = q - poly
        return bool(((e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]) >= -1e-9).all())


class Scene:
    def __init__(self):
        self.objects: list[SceneObject] = []
        self._sup_cache = (None, None)

    # ---- supports: stored per object (SceneObject.sup); this is the merged world-space mesh of all of them ----
    @property
    def supports(self) -> trimesh.Trimesh | None:
        parts = [o for o in self.objects if o.sup is not None]
        if not parts:
            return None
        key = tuple((id(o.sup), o.matrix().tobytes()) for o in parts)
        if self._sup_cache[0] == key:
            return self._sup_cache[1]
        ms = []
        for o in parts:
            m = o.sup.copy()
            m.apply_transform(o.matrix())
            ms.append(m)
        out = ms[0] if len(ms) == 1 else trimesh.util.concatenate(ms)
        self._sup_cache = (key, out)
        return out

    @supports.setter
    def supports(self, mesh):
        """None clears every object's supports. A mesh (project load, pillar deletion) is split into its shells,
        each given to the nearest object."""
        for o in self.objects:
            o.sup = None
        self._sup_cache = (None, None)
        if mesh is None or not len(mesh.faces) or not self.objects:
            return
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
        V = len(v)
        g = coo_matrix((np.ones(2 * len(f)), (np.r_[f[:, 0], f[:, 1]], np.r_[f[:, 1], f[:, 2]])), shape=(V, V))
        nc, lab = connected_components(g, directed=False)
        cnt = np.maximum(np.bincount(lab, minlength=nc), 1)
        cen = np.c_[np.bincount(lab, v[:, 0], nc), np.bincount(lab, v[:, 1], nc)] / cnt[:, None]
        dist = []
        for o in self.objects:                                   # distance of each shell from the object's footprint
            b = o.bounds()
            d = np.maximum(np.maximum(b[0][:2] - cen, cen - b[1][:2]), 0)
            dist.append(np.hypot(d[:, 0], d[:, 1]))
        owner = np.argmin(np.array(dist), axis=0)[lab[f[:, 0]]]
        for k, o in enumerate(self.objects):
            m = owner == k
            if not m.any():
                continue
            sub = trimesh.Trimesh(v, f[m], process=False)
            sub.remove_unreferenced_vertices()
            sub.apply_transform(np.linalg.inv(o.matrix()))
            o.sup = sub

    def set_object_supports(self, o: SceneObject, world_mesh: trimesh.Trimesh | None):
        """Give `o` the supports `world_mesh` (built for it at its current position), or none."""
        if world_mesh is None:
            o.sup = None
        else:
            o.sup = world_mesh.copy()
            o.sup.apply_transform(np.linalg.inv(o.matrix()))

    def add(self, mesh: trimesh.Trimesh, name: str) -> SceneObject:
        o = SceneObject(name, mesh)
        self.objects.append(o)
        return o

    def remove(self, i: int):
        del self.objects[i]

    def snapshot(self):
        """Cheap undo state: object parameters (supports included) + shared (never mutated in place) meshes."""
        return [replace(o, _cache=None) for o in self.objects], None

    def restore(self, snap):
        objs, _ = snap
        self.objects = [replace(o, _cache=None) for o in objs]

    def printable(self, machine) -> list[SceneObject]:
        """The objects that lie inside the print area; the others are ignored by supports, slicing and export."""
        return [o for o in self.objects if o.fits(machine)]

    def combined(self, with_supports=True, machine=None) -> trimesh.Trimesh | None:
        """All geometry to slice; with `machine`, objects outside its print area are left out."""
        parts = [o.world_mesh() for o in (self.printable(machine) if machine is not None else self.objects)]
        if not parts:
            return None
        if with_supports and self.supports is not None:
            parts.append(self.supports)
        return trimesh.util.concatenate(parts)
