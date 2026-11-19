"""Simple modelling operations on trimesh meshes (all lengths in mm, Z up)."""
from dataclasses import dataclass

import numpy as np
import trimesh

ENGINE = "manifold"


@dataclass
class Param:
    label: str
    default: float
    lo: float = 0.1
    hi: float = 500.0
    integer: bool = False


# name -> (params, builder(**values)); every primitive is centred on the origin
PRIMITIVES = {
    "Cubo": ([Param("Larghezza X", 20), Param("Profondità Y", 20), Param("Altezza Z", 20)],
             lambda a, b, c: trimesh.creation.box(extents=(a, b, c))),
    "Sfera": ([Param("Raggio", 10)],
              lambda r: trimesh.creation.icosphere(subdivisions=4, radius=r)),
    "Cilindro": ([Param("Raggio", 10), Param("Altezza", 20), Param("Segmenti", 64, 3, 256, True)],
                 lambda r, h, n: trimesh.creation.cylinder(radius=r, height=h, sections=int(n))),
    "Cono": ([Param("Raggio base", 10), Param("Altezza", 20), Param("Segmenti", 64, 3, 256, True)],
             lambda r, h, n: _centered(trimesh.creation.cone(radius=r, height=h, sections=int(n)))),
    "Toro": ([Param("Raggio maggiore", 12), Param("Raggio tubo", 3)],
             lambda R, r: trimesh.creation.torus(major_radius=R, minor_radius=r, major_sections=64, minor_sections=24)),
    "Prisma": ([Param("Lati", 6, 3, 64, True), Param("Raggio", 10), Param("Altezza", 20)],
               lambda n, r, h: trimesh.creation.cylinder(radius=r, height=h, sections=int(n))),
    "Piramide": ([Param("Lati base", 4, 3, 64, True), Param("Raggio", 10), Param("Altezza", 20)],
                 lambda n, r, h: _centered(trimesh.creation.cone(radius=r, height=h, sections=int(n)))),
    "Capsula": ([Param("Raggio", 5), Param("Lunghezza cilindro", 20)],
                lambda r, h: trimesh.creation.capsule(radius=r, height=h)),
}


def _centered(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m.apply_translation(-m.bounds.mean(axis=0))
    return m


def make_primitive(kind: str, *values) -> trimesh.Trimesh:
    params, build = PRIMITIVES[kind]
    if len(values) != len(params):
        raise ValueError(f"{kind}: servono {len(params)} parametri")
    m = build(*values)
    m.fix_normals()
    return m


def diagnose(m: trimesh.Trimesh) -> dict:
    """Why a mesh is not watertight: open (boundary) edges and edges shared by 3+ faces (fast, integer keys)."""
    if len(m.faces) == 0:
        return dict(watertight=False, faces=0, open_edges=0, nonmanifold_edges=0)
    f = m.faces.astype(np.int64)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e[:, 0] * len(m.vertices) + e[:, 1], return_counts=True)
    opened, nonman = int((counts == 1).sum()), int((counts > 2).sum())
    return dict(watertight=bool(opened == 0 and nonman == 0 and m.is_watertight), faces=len(m.faces),
                open_edges=opened, nonmanifold_edges=nonman)


def _basic_clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    # weld tolerance relative to the model size (trimesh's absolute 1e-8 fuses distinct vertices of files in metres)
    diag = float(np.linalg.norm(m.extents)) if len(m.vertices) else 0.0
    digits = int(np.clip(round(-np.log10(max(diag, 1e-12) * 2e-8)), 4, 12))
    m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=digits)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(m)
    trimesh.repair.fill_holes(m)
    m.fix_normals()                                     # includes the inversion fix
    return m


def _fill_boundary_loops(m: trimesh.Trimesh, max_frac: float = 0.15) -> trimesh.Trimesh:
    """Close the open boundary loops of a consistently wound mesh with a fan around each loop's centroid: local and
    O(boundary), unlike PyMeshFix which rebuilds the whole mesh. Loops longer than `max_frac` of the bbox diagonal
    (big holes, where a fan looks bad) and non-simple loops are left for PyMeshFix."""
    f = np.asarray(m.faces, np.int64)
    he = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = he[:, 0] * len(m.vertices) + he[:, 1]
    rkey = he[:, 1] * len(m.vertices) + he[:, 0]
    bnd = he[~np.isin(key, rkey)]                       # directed edges with no opposite half-edge
    if len(bnd) == 0:
        return m
    nxt = {}
    for a, b in bnd.tolist():
        nxt.setdefault(a, []).append(b)
    loops = []
    while nxt:                                          # walk the boundary; a revisited vertex cuts a cycle out as a loop
        start = next(iter(nxt))
        loop, pos, cur = [start], {start: 0}, start
        while cur in nxt:
            outs = nxt[cur]
            step = outs.pop()
            if not outs:
                del nxt[cur]
            if step in pos:
                i = pos[step]
                cyc = loop[i:]
                if len(cyc) >= 3:
                    loops.append(cyc)
                for u in cyc[1:]:
                    del pos[u]
                loop = loop[:i + 1]
            else:
                pos[step] = len(loop)
                loop.append(step)
            cur = step
    verts = np.asarray(m.vertices, float)
    limit = max_frac * float(np.linalg.norm(m.extents))
    new_v, new_f = [], []
    nv = len(verts)
    for loop in loops:
        pts = verts[loop]
        if float(np.ptp(pts, axis=0).max()) > limit:
            continue
        new_v.append(pts.mean(axis=0))
        c = nv + len(new_v) - 1
        new_f += [(loop[(i + 1) % len(loop)], loop[i], c) for i in range(len(loop))]
    if not new_f:
        return m
    return trimesh.Trimesh(np.vstack([verts, new_v]), np.vstack([f, new_f]), process=False)


def _glue_near_vertices(m: trimesh.Trimesh, frac: float = 0.05) -> trimesh.Trimesh:
    """Merge vertices closer than `frac` x the median edge length (SatelLite's "stitch close triangles": it glues
    at 0.05 mm). Double-layered exports have two copies of the surface a few microns apart; welding them turns most of
    the non-manifold edges and cracks into ordinary shared edges instead of deleting triangles."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    v = np.asarray(m.vertices, float)
    f = np.asarray(m.faces, np.int64)
    if len(f) == 0:
        return m
    edges = np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1)
    tol = float(np.median(edges[edges > 0])) * frac if (edges > 0).any() else 0.0
    if tol <= 0:
        return m
    pairs = cKDTree(v).query_pairs(tol, output_type="ndarray")
    if len(pairs) == 0:
        return m
    g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(v), len(v)))
    n, lab = connected_components(g, directed=False)
    nv = np.zeros((n, 3))
    np.add.at(nv, lab, v)
    nv /= np.bincount(lab, minlength=n)[:, None]
    f2 = lab[f]
    f2 = f2[(f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 2] != f2[:, 0])]
    _, first = np.unique(np.sort(f2, axis=1), axis=0, return_index=True)
    out = trimesh.Trimesh(nv, f2[np.sort(first)], process=False)
    out.remove_unreferenced_vertices()
    return out


def _drop_nonmanifold(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove every face that touches an edge shared by 3+ faces (what usually breaks AI/sculpt exports)."""
    f = np.asarray(m.faces, np.int64)
    nv = len(m.vertices)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    k = np.minimum(e[:, 0], e[:, 1]) * nv + np.maximum(e[:, 0], e[:, 1])
    _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    bad = (cnt[inv] > 2).reshape(3, -1).any(axis=0)
    if not bad.any():
        return m
    f = f[~bad]
    import scipy.sparse as sp                           # debris left over by the removal: pieces < 1% of the biggest
    from scipy.sparse.csgraph import connected_components
    g = sp.coo_matrix((np.ones(f.size), (np.repeat(f[:, 0], 3), f.ravel())), shape=(nv, nv))
    n, lab = connected_components(g, directed=False)
    comp = lab[f[:, 0]]
    size = np.bincount(comp, minlength=n)
    f = f[size[comp] >= 0.01 * size.max()]
    out = trimesh.Trimesh(m.vertices, f, process=False)
    out.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(out)
    return out


def _meshfix(m: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """Heavy repair (large holes, non-manifold edges, self-intersections) with PyMeshFix."""
    try:
        import pymeshfix
    except ImportError:
        return None
    fixer = pymeshfix.MeshFix(np.asarray(m.vertices, float), np.asarray(m.faces, np.int32))
    fixer.repair(joincomp=True, remove_smallest_components=False)
    out = trimesh.Trimesh(np.asarray(fixer.points), np.asarray(fixer.faces), process=True)
    if len(out.faces) == 0:
        return None
    out.fix_normals()
    return out


HEAVY_FACES = 1_000_000          # above this a mesh is too heavy to draw, slice and repair comfortably
SIMPLIFY_TARGET = 500_000


def simplify(mesh: trimesh.Trimesh, target_faces: int = SIMPLIFY_TARGET) -> trimesh.Trimesh:
    """Quadric decimation (fast_simplification). Keeps the shape, drops triangles."""
    import fast_simplification
    if len(mesh.faces) <= target_faces:
        return mesh
    v, f = fast_simplification.simplify(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32),
                                        target_count=int(target_faces), agg=5)
    return trimesh.Trimesh(v, f, process=False)


def _similar(a: trimesh.Trimesh, b: trimesh.Trimesh, tol: float = 0.25) -> bool:
    """True when `b` still looks like `a` (bounding box); guards against repairs that keep only a fragment."""
    ea, eb = np.asarray(a.extents, float), np.asarray(b.extents, float)
    return bool(np.all(eb >= ea * (1 - tol)) and np.all(eb <= ea * (1 + tol)))


def repair(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    """Clean the mesh, drop non-manifold faces, close holes, fix winding; PyMeshFix only as a last resort
    (and its result is discarded if it no longer resembles the original)."""
    before = diagnose(mesh)
    m = _basic_clean(mesh)
    steps = ["pulizia"]
    after = diagnose(m)
    if not after["watertight"]:
        g = _glue_near_vertices(m)
        if g is not m:
            trimesh.repair.fix_winding(g)
            m, after = g, diagnose(g)
            steps.append("vertici vicini uniti")
    if after["nonmanifold_edges"]:
        m = _drop_nonmanifold(m)
        after = diagnose(m)
        steps.append("spigoli non-manifold rimossi")
    if not after["watertight"] and after["open_edges"] and not after["nonmanifold_edges"]:
        local = _fill_boundary_loops(m)                 # fast local hole closing before the heavy global repair
        if local is not m:
            local.fix_normals()
            d = diagnose(local)
            if d["open_edges"] + d["nonmanifold_edges"] < after["open_edges"] + after["nonmanifold_edges"]:
                m, after = local, d
                steps.append("chiusura buchi")
    if not after["watertight"]:
        fixed = _meshfix(m)
        if fixed is not None and _similar(m, fixed):
            d = diagnose(fixed)
            if d["watertight"] or d["open_edges"] + d["nonmanifold_edges"] < after["open_edges"] + after["nonmanifold_edges"]:
                m, after = fixed, d
                steps.append("PyMeshFix")
    if not after["watertight"] and after["open_edges"] and not after["nonmanifold_edges"]:
        big = _fill_boundary_loops(m, max_frac=10.0)    # last resort: cap the large openings too, so it can be sliced
        if big is not m:
            big.fix_normals()
            d = diagnose(big)
            if d["open_edges"] + d["nonmanifold_edges"] < after["open_edges"] + after["nonmanifold_edges"]:
                m, after = big, d
                steps.append("chiusura buchi grandi")
    return m, dict(before=before, after=after, steps=steps)


def _clean(m: trimesh.Trimesh) -> trimesh.Trimesh | None:
    if m is None or len(m.faces) == 0 or abs(m.volume) < 1e-6:
        return None
    m.remove_unreferenced_vertices()
    return m


def boolean(op: str, meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """op: 'union' | 'difference' (first minus the others) | 'intersection'."""
    if len(meshes) < 2:
        raise ValueError("servono almeno due oggetti")
    fn = {"union": trimesh.boolean.union, "difference": trimesh.boolean.difference,
          "intersection": trimesh.boolean.intersection}[op]
    prepared = []
    for m in meshes:
        if not m.is_watertight:
            m, _ = repair(m)
        if not m.is_watertight:
            raise ValueError("una delle mesh non è a tenuta e non è riparabile: il booleano non è affidabile")
        prepared.append(m)
    res = _clean(fn(prepared, engine=ENGINE))
    if res is None:
        raise ValueError("il risultato è vuoto (gli oggetti non si sovrappongono?)")
    return res


def plane_normal(axis: str) -> np.ndarray:
    return {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}[axis.upper()]


_FRAMES = {"X": ((0, 1, 0), (0, 0, 1)), "Y": ((0, 0, 1), (1, 0, 0)), "Z": ((1, 0, 0), (0, 1, 0))}


def tilted_normal(axis: str, tilt_deg: float = 0.0, azimuth_deg: float = 0.0) -> np.ndarray:
    """Normal of a plane that starts perpendicular to `axis` and is tilted by `tilt_deg`
    towards the direction `azimuth_deg` (measured in the plane's own frame: Z -> from +X to +Y,
    X -> from +Y to +Z, Y -> from +Z to +X)."""
    n0 = plane_normal(axis)
    u, v = (np.array(f, float) for f in _FRAMES[axis.upper()])
    t, a = np.radians(tilt_deg), np.radians(azimuth_deg)
    n = n0 * np.cos(t) + (u * np.cos(a) + v * np.sin(a)) * np.sin(t)
    return n / np.linalg.norm(n)


def cut(mesh: trimesh.Trimesh, normal, offset: float):
    """Cut by the plane n·p = offset. Returns (positive_side, negative_side), each capped and watertight
    (or None when that side is empty). Done with a boolean against a half-space box, so caps are exact."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    if not mesh.is_watertight:
        mesh, _ = repair(mesh)
        if not mesh.is_watertight:
            raise ValueError("la mesh non è a tenuta: il taglio con tappo non è affidabile")
    c = mesh.bounds.mean(axis=0)
    origin = c - n * (np.dot(c, n) - offset)            # centre projected on the plane
    S = float(np.linalg.norm(mesh.extents)) * 2 + 10
    box = trimesh.creation.box(extents=(S, S, S))
    box.apply_translation([0, 0, S / 2])                # occupies the +n half-space
    T = trimesh.geometry.align_vectors([0, 0, 1], n)
    T[:3, 3] = origin
    box.apply_transform(T)
    pos = _clean(trimesh.boolean.intersection([mesh, box], engine=ENGINE))
    neg = _clean(trimesh.boolean.difference([mesh, box], engine=ENGINE))
    return pos, neg


def split_shells(mesh: trimesh.Trimesh, min_faces: int = 4) -> list[trimesh.Trimesh]:
    """Separate disconnected pieces (shells that share no vertices). Largest first."""
    m = mesh.copy()
    m.merge_vertices()
    parts = m.split(only_watertight=False)
    parts = [p for p in parts if len(p.faces) >= min_faces]
    parts.sort(key=lambda p: len(p.faces), reverse=True)
    return parts


MAX_HOLLOW_CELLS = 40_000_000


def hollow(mesh: trimesh.Trimesh, wall: float = 2.0, drain_diameter: float = 3.0, drain_holes: bool = True,
           min_cavity: float = 4.0) -> tuple[trimesh.Trimesh, dict]:
    """Hollow a closed mesh: keep a shell `wall` mm thick, optionally with a drain hole at the lowest point of each cavity and
    a vent at its highest, so the resin can run out (a sealed cavity traps resin and can burst under the peel force).
    The cavity is the level set of the inward distance field (voxel grid + marching cubes, sub-voxel accurate), added to the
    mesh as an inverted inner shell. Cavities smaller than `min_cavity` mm across are left solid.
    Returns (mesh, info)."""
    from scipy import ndimage
    from skimage import measure
    from .slicer import voxelize

    if not mesh.is_watertight:
        mesh, _ = repair(mesh)
        if not mesh.is_watertight:
            raise ValueError("la mesh non è a tenuta: lo svuotamento non è affidabile (usa «Ripara mesh»)")
    pitch = max(wall / 3.0, 0.15)
    cells = float(np.prod(mesh.extents + 4 * pitch)) / pitch ** 3
    if cells > MAX_HOLLOW_CELLS:                                   # keep the grid affordable on big parts
        pitch *= (cells / MAX_HOLLOW_CELLS) ** (1 / 3)
        if pitch > wall / 1.5:
            raise ValueError("pezzo troppo grande per lo svuotamento con questo spessore: aumenta lo spessore della parete")
    vol, lo = voxelize(mesh, pitch)
    dist = ndimage.gaussian_filter(ndimage.distance_transform_edt(vol).astype(np.float32) * pitch, 0.6)
    if dist.max() <= wall + pitch:
        raise ValueError(f"il pezzo è troppo piccolo o sottile per una parete di {wall:g} mm")
    v, f, _, _ = measure.marching_cubes(dist, level=wall + pitch / 2, spacing=(pitch,) * 3)   # EDT is to the first empty voxel centre: half a voxel more than the surface
    inner = trimesh.Trimesh(v[:, ::-1] + lo + pitch / 2, f[:, ::-1], process=True)
    cavities = []
    for c in inner.split(only_watertight=False):
        if c.is_watertight and c.extents.min() >= min_cavity:
            if c.volume > 0:
                c.invert()                                         # normals into the cavity: it is a void
            cavities.append(c)
    if not cavities:
        raise ValueError("nessuna cavità abbastanza grande: riduci lo spessore della parete")
    res = trimesh.util.concatenate([mesh] + cavities)
    holes = []
    if drain_holes:
        r = drain_diameter / 2
        dirs = [np.array([np.cos(a), np.sin(a), 0.0]) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        for c in cavities:
            for p in (c.vertices[c.vertices[:, 2].argmin()], c.vertices[c.vertices[:, 2].argmax()]):
                best = None
                for d in dirs:
                    hit, _, _ = mesh.ray.intersects_location([p + d * 0.01], [d], multiple_hits=False)
                    if len(hit):
                        dist_out = float(np.linalg.norm(hit[0] - p))
                        if best is None or dist_out < best[0]:
                            best = (dist_out, d)
                if best is None:
                    continue
                L = best[0] + r + 1.0
                start = p - best[1] * r
                cyl = trimesh.creation.cylinder(radius=r, height=L, sections=24)
                T = trimesh.geometry.align_vectors([0, 0, 1], best[1])
                T[:3, 3] = start + best[1] * L / 2
                cyl.apply_transform(T)
                holes.append(cyl)
    if holes:
        res = trimesh.boolean.difference([res] + holes, engine=ENGINE)
    res.fix_normals()
    return res, dict(cavities=len(cavities), holes=len(holes), pitch=round(pitch, 3),
                     volume_before=float(mesh.volume), volume_after=float(res.volume))
