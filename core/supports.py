"""Automatic pillar supports for resin printing."""
from dataclasses import dataclass
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree


@dataclass
class SupportParams:
    critical_angle: float = 30.0   # overhang angle from vertical beyond which a face needs support
    spacing: float = 3.0
    tip_radius: float = 0.3
    body_radius: float = 0.7
    base_radius: float = 1.6
    base_height: float = 1.0
    tip_length: float = 1.0
    lift_height: float = 5.0
    raft_height: float = 0.0       # 0 = no raft; otherwise a slab of this thickness on the plate under the pillars
    raft_margin: float = 2.0       # how far the raft extends beyond the outermost pillar bases
    raft_chamfer: float = 0.5      # bevel of the raft's top edge (solid raft only; clamped to the raft thickness)
    raft_hole: float = 0.0         # >0: perforated raft, square holes of this size between the pillars (0 = solid)
    raft_bar: float = 1.0          # width of the raft material kept between holes, around pillars and along the edge
    tip_sink: float = 0.02         # mm the tip end goes into the model (ChituBox: contact depth; more = firmer hold)
    trunk_radius: float = 0.0      # >0: pillars at least trunk_min_height tall are thicker in their lower half (ChituBox "lower diameter")
    trunk_min_height: float = 5.0
    cross_radius: float = 0.0      # >0: X braces of this radius between neighbouring tall pillars (ChituBox cross supports)
    cross_reach: float = 0.0       # max horizontal distance between two pillars joined by braces (0 = 1.2 x spacing: direct neighbours only)
    cross_rise: float = 2.0        # vertical rise of one brace
    cross_percent: float = 30.0    # how many X levels: 10 = about 3 per 100 mm of pillar height, 100 = about 30 per 100 mm
    max_points: int = 20000

    def __post_init__(self):
        if self.raft_height > 0:   # the model must clear the raft by enough room for pillars (base + shaft + tip)
            self.lift_height = max(self.lift_height, self.raft_height + self.base_height + 2 * self.tip_length)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


MIN_OVERHANG_AREA = 6.0   # mm2: a smaller patch of overhang (a tuft in scanned fur) holds itself, it gets no pillar


def _big_enough(mesh, over, p):
    """Mask over the face ids `over`: True for faces of an overhang patch of at least MIN_OVERHANG_AREA. Patches are
    grown over a looser angle than the support angle, so a rough real overhang is not cut into small bits."""
    if len(over) == 0:
        return np.zeros(0, bool)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    loose = np.where(mesh.face_normals[:, 2] < -np.sin(np.radians(max(p.critical_angle - 20.0, 0.0))))[0]
    adj = trimesh.Trimesh(mesh.vertices, mesh.faces[loose], process=False).face_adjacency
    n = len(loose)
    _, lab = connected_components(coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n)), directed=False)
    area = np.bincount(lab, weights=mesh.area_faces[loose])
    return (area >= MIN_OVERHANG_AREA)[lab[np.searchsorted(loose, over)]]


def find_points(mesh: trimesh.Trimesh, p: SupportParams) -> np.ndarray:
    pts = []
    zmin = mesh.bounds[0][2]
    # overhanging faces, sampled on a grid: lowest sample per (cell, height band)
    over = np.where(mesh.face_normals[:, 2] < -np.sin(np.radians(p.critical_angle)))[0]
    over = over[_big_enough(mesh, over, p)]
    if len(over):
        area = mesh.area_faces[over].sum()
        count = int(min(p.max_points * 3, max(200, area / (p.spacing * 0.5) ** 2)))
        sub = trimesh.Trimesh(mesh.vertices, mesh.faces[over], process=False)
        s, _ = trimesh.sample.sample_surface(sub, count)
        cell = np.floor(s[:, :2] / p.spacing).astype(np.int64)
        band = np.floor(s[:, 2] / p.spacing).astype(np.int64)
        key = np.stack([cell[:, 0], cell[:, 1], band], 1)
        _, uniq = np.unique(key, axis=0, return_index=True)
        pts.append(s[uniq])
    # local minima (islands) always need a tip
    v, e = mesh.vertices, mesh.edges_unique
    nbr_min = np.full(len(v), np.inf)
    np.minimum.at(nbr_min, e[:, 0], v[e[:, 1], 2])
    np.minimum.at(nbr_min, e[:, 1], v[e[:, 0], 2])
    mins = v[(v[:, 2] < nbr_min) & (v[:, 2] > zmin + 1e-3)]
    if len(mins):
        pts.append(mins)
    if not pts:
        return np.zeros((0, 3))
    return np.concatenate(pts)[: p.max_points]


def _ground(mesh, pts):
    """(z where each pillar ends: the plate (0) or the model surface directly below, True where the point is inside the
    model). A ray going down from a point in open air meets the surface below from its outer side (normal up); met from
    its inner side (normal down) the point was inside the solid, e.g. a downward face buried in scanned fur."""
    ends = np.zeros(len(pts))
    inside = np.zeros(len(pts), bool)
    if len(pts) == 0:
        return ends, inside
    try:
        loc, ray, tri = mesh.ray.intersects_location(
            pts - [0, 0, 0.05], np.tile([0, 0, -1.0], (len(pts), 1)), multiple_hits=False)
        ends[ray] = loc[:, 2]
        inside[ray] = mesh.face_normals[tri, 2] < 0
    except Exception:
        pass
    return ends, inside




def _clipped(mesh, pts, top, low, tl, p):
    """True where the pillar shaft would cut into the model beside its axis (creases, walls, fur): a vertical line
    offset by the shaft radius hits the model well above where the pillar ends."""
    bad = np.zeros(len(pts), bool)
    if len(pts) == 0:
        return bad
    r = p.body_radius
    offs = np.array([[r, 0], [-r, 0], [0, r], [0, -r]])
    start = np.repeat(top - tl, len(offs))
    org = np.c_[(pts[:, None, :2] + offs[None]).reshape(-1, 2), start]
    try:
        loc, ray, _ = mesh.ray.intersects_location(org, np.tile([0, 0, -1.0], (len(org), 1)), multiple_hits=False)
    except Exception:
        return bad
    if len(ray) == 0:
        return bad
    hit = ray // len(offs)
    above = loc[:, 2] > low[hit] + 1.0
    bad[hit[above]] = True
    return bad


RAFT_CELL = 0.5   # mm; hole and bar sizes of the perforated raft are rounded to this grid


def _hull_eq(xy, r, sides=16):
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    cloud = (xy[:, None, :] + r * np.stack([np.cos(a), np.sin(a)], 1)[None]).reshape(-1, 2)
    return cloud, ConvexHull(cloud)


def _raft_perforated(xy: np.ndarray, p: SupportParams):
    """Raft with a grid of square holes between the pillars, so it peels off the plate and breaks away from the
    pillars after printing. Built from grid cells: solid = inside the footprint and (near a pillar base, near the
    edge or on a bar between holes). The bars and the rim connect everything, so the raft stays one piece."""
    g, h = RAFT_CELL, p.raft_height
    nh = max(1, round(p.raft_hole / g))
    nb = max(1, round(p.raft_bar / g))
    bar = nb * g
    cloud, hull = _hull_eq(xy, p.base_radius + p.raft_margin)
    lo = cloud.min(0)
    nx, ny = np.ceil((cloud.max(0) - lo) / g).astype(int)
    cx, cy = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * g, lo[1] + (np.arange(ny) + 0.5) * g, indexing="ij")
    c = np.stack([cx.ravel(), cy.ravel()], 1)
    depth = -(c @ hull.equations[:, :2].T + hull.equations[:, 2]).max(1)        # distance inside the footprint outline
    inside = depth >= 0
    near = cKDTree(xy).query(c)[0] < p.base_radius + g
    ij = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), -1).reshape(-1, 2)
    pitch = nh + nb
    hole = ((ij[:, 0] % pitch) >= nb) & ((ij[:, 1] % pitch) >= nb)
    solid = (inside & ((depth < bar) | near | ~hole)).reshape(nx, ny)
    for _ in range(2):                                  # cells touching only at a corner would make the mesh non-manifold
        a, b, c, d = solid[:-1, :-1], solid[1:, :-1], solid[:-1, 1:], solid[1:, 1:]
        solid[1:, :-1] |= a & d & ~b & ~c
        solid[:-1, :-1] |= b & c & ~a & ~d
    # faces: top/bottom quad per solid cell, side quad where the neighbour cell is empty
    vid = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    X, Y = np.meshgrid(lo[0] + np.arange(nx + 1) * g, lo[1] + np.arange(ny + 1) * g, indexing="ij")
    n0 = len(vid.ravel())
    verts = np.concatenate([np.c_[X.ravel(), Y.ravel(), np.zeros(n0)], np.c_[X.ravel(), Y.ravel(), np.full(n0, h)]])
    i, j = np.nonzero(solid)
    a, b, cc, d = vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]
    f = [np.stack([a, cc, b], 1), np.stack([a, d, cc], 1),                       # bottom, facing -z
         np.stack([a + n0, b + n0, cc + n0], 1), np.stack([a + n0, cc + n0, d + n0], 1)]   # top, facing +z
    pad = np.pad(solid, 1)
    for di, dj, e0, e1 in ((0, -1, "a", "b"), (1, 0, "b", "cc"), (0, 1, "cc", "d"), (-1, 0, "d", "a")):
        m = ~pad[i + 1 + di, j + 1 + dj]
        v0, v1 = {"a": a, "b": b, "cc": cc, "d": d}[e0][m], {"a": a, "b": b, "cc": cc, "d": d}[e1][m]
        f += [np.stack([v0, v1, v1 + n0], 1), np.stack([v0, v1 + n0, v0 + n0], 1)]
    mesh = trimesh.Trimesh(verts, np.concatenate(f), process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def _raft(xy: np.ndarray, p: SupportParams, sides: int = 16):
    """Raft slab (convex hull of the pillar bases grown by base_radius + raft_margin), z from 0 to raft_height."""
    if p.raft_hole > 0:
        return _raft_perforated(xy, p)
    h = p.raft_height
    r_out = p.base_radius + p.raft_margin
    ch = min(max(p.raft_chamfer, 0.0), h * 0.9, r_out * 0.9)
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    circle = np.stack([np.cos(a), np.sin(a)], 1)
    rings = []
    for r in (r_out, r_out - ch):                       # bottom outline, then the (smaller) top outline
        cloud = (xy[:, None, :] + r * circle[None]).reshape(-1, 2)
        if len(xy) == 1:
            hull = cloud
        else:
            hull = cloud[ConvexHull(cloud).vertices]    # counter-clockwise
        rings.append(hull)
    bot, top = rings
    # the two hulls can have different vertex counts: resample both on the same angular grid around the centre
    c = bot.mean(0)
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    out = []
    for ring in (bot, top):
        d = ring - c
        th = np.arctan2(d[:, 1], d[:, 0])
        o = np.argsort(th)
        th, d = th[o], d[o]
        th2 = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
        out.append(c + np.stack([np.interp(ang, th2, np.tile(d[:, k], 3)) for k in (0, 1)], 1))
    n = len(ang)
    verts = np.concatenate([np.c_[out[0], np.zeros(n)], np.c_[out[1], np.full(n, h)], [[*c, 0.0], [*c, h]]])
    i = np.arange(n)
    j = (i + 1) % n
    f = [np.stack([i, j, n + j], 1), np.stack([i, n + j, n + i], 1),         # chamfered side
         np.stack([np.full(n, 2 * n), j, i], 1),                              # bottom cap
         np.stack([np.full(n, 2 * n + 1), n + i, n + j], 1)]                  # top cap
    return trimesh.Trimesh(verts, np.concatenate(f), process=False)


BRACE_SIDES = 6


def _surface_tree(mesh, spacing=0.4, cap=600000):
    """KD-tree of points spread over the model surface about `spacing` mm apart (vertices included)."""
    n = int(min(cap, max(2000, mesh.area / spacing ** 2)))
    s = trimesh.sample.sample_surface(mesh, n)[0]
    return cKDTree(np.concatenate([s, np.asarray(mesh.vertices)]))
BRACE_VERTS = 2 * BRACE_SIDES


def _braces(mesh, pts, low, bh, shaft_top, p):
    """X braces between neighbouring pillars (only in their shafts, in open air): thin prisms from the axis of one
    pillar to the axis of the next, rising by cross_rise. They stiffen the pillar field against the peel force and
    stay between pillars: nothing goes into the model. Returns a mesh or None."""
    S = len(pts)
    if S < 2 or p.cross_percent <= 0 or (shaft_top - low).max() < p.trunk_min_height:
        return None
    pitch = max(100.0 / (0.3 * p.cross_percent), p.cross_rise)   # vertical distance between two X levels
    pairs = cKDTree(pts[:, :2]).query_pairs(p.cross_reach or 1.2 * p.spacing, output_type="ndarray")
    if len(pairs) == 0:
        return None
    a, b = pairs[:, 0], pairs[:, 1]
    dxy = np.linalg.norm(pts[a, :2] - pts[b, :2], axis=1)
    ok = (dxy > 1.2) & ((shaft_top[a] - low[a]) >= p.trunk_min_height) & ((shaft_top[b] - low[b]) >= p.trunk_min_height)
    a, b = a[ok], b[ok]
    zlo = np.maximum(low[a] + bh[a], low[b] + bh[b]) + 0.6
    zhi = np.minimum(shaft_top[a], shaft_top[b]) - 0.6
    nlev = np.floor((zhi - zlo - p.cross_rise) / pitch).astype(int) + 1
    nlev = np.where(zhi - zlo >= p.cross_rise, nlev, 0)
    if nlev.sum() == 0:
        return None
    pi = np.repeat(np.arange(len(a)), nlev)
    k = np.arange(len(pi)) - np.repeat(np.cumsum(nlev) - nlev, nlev)
    z0 = zlo[pi] + k * pitch
    A, B = a[pi], b[pi]
    P = np.concatenate([np.c_[pts[A, :2], z0], np.c_[pts[B, :2], z0]])               # the two diagonals of each X
    Q = np.concatenate([np.c_[pts[B, :2], z0 + p.cross_rise], np.c_[pts[A, :2], z0 + p.cross_rise]])
    P, Q = P[: p.max_points * 4], Q[: p.max_points * 4]
    d = Q - P
    L = np.linalg.norm(d, axis=1)
    d /= L[:, None]
    u = np.cross([0.0, 0.0, 1.0], d)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    r = p.cross_radius
    # a brace must not touch the model: its sample points must stay away from the surface (sampled points + KD-tree;
    # much faster than ray casting against the whole mesh)
    k = max(2, int(np.ceil(L.max() / 0.5)) + 1)
    t = np.linspace(0, 1, k)
    tree = _surface_tree(mesh)
    clear = np.ones(len(P), bool)
    for c in range(0, len(P), 4000):
        sl = slice(c, c + 4000)
        pts_ = P[sl, None] + (Q[sl] - P[sl])[:, None] * t[None, :, None]
        dist = tree.query(pts_.reshape(-1, 3), distance_upper_bound=r + 0.4)[0].reshape(len(pts_), k)
        clear[sl] = ~np.isfinite(dist).any(axis=1)
    P, Q, u, v = P[clear], Q[clear], u[clear], v[clear]
    M = len(P)
    if M == 0:
        return None
    al = np.linspace(0, 2 * np.pi, BRACE_SIDES, endpoint=False)
    rim = r * (np.cos(al)[None, :, None] * u[:, None] + np.sin(al)[None, :, None] * v[:, None])   # (M, sides, 3)
    verts = np.concatenate([P[:, None] + rim, Q[:, None] + rim], axis=1).reshape(-1, 3)
    i = np.arange(BRACE_SIDES)
    j = (i + 1) % BRACE_SIDES
    n = BRACE_SIDES
    f = np.concatenate([np.stack([i, j, n + j], 1), np.stack([i, n + j, n + i], 1),
                        np.stack([np.zeros(n - 2, int), i[2:], i[1:-1]], 1),            # cap at P
                        np.stack([np.full(n - 2, n), n + i[1:-1], n + i[2:]], 1)])      # cap at Q
    faces = ((np.arange(M) * BRACE_VERTS)[:, None, None] + f[None]).reshape(-1, 3)
    return trimesh.Trimesh(verts, faces, process=False)


def build_supports(mesh: trimesh.Trimesh, p: SupportParams, sides: int = 8):
    """Return (support_mesh | None, count). Geometry is built vectorised, not per pillar."""
    pts = find_points(mesh, p)
    if len(pts) == 0:
        return None, 0
    low, buried = _ground(mesh, pts)
    pts, low = pts[~buried], low[~buried]
    if len(pts) == 0:
        return None, 0
    on_plate = low < 1e-3
    if p.raft_height > 0:
        low = np.where(on_plate, p.raft_height, low)            # pillars stand exactly on the raft: the slicer fills
                                                                # even-odd, so overlapping shells would punch holes
    top = pts[:, 2]
    length = top - low
    keep = length > p.tip_length * 0.5
    keep &= on_plate | (length >= 3 * p.tip_length)    # a pillar from the model onto itself shorter than three tips is just a blob
    pts, top, low, length, on_plate = pts[keep], top[keep], low[keep], length[keep], on_plate[keep]
    if len(pts):
        ok = ~_clipped(mesh, pts, top, low, np.minimum(p.tip_length, length * 0.5), p)
        pts, top, low, length, on_plate = pts[ok], top[ok], low[ok], length[ok], on_plate[ok]
    S = len(pts)
    if S == 0:
        return None, 0
    tl = np.minimum(p.tip_length, length * 0.5)
    inner = ~on_plate                                  # pillar standing on the model itself: a tip at both ends, no base
    bh = np.where(inner, tl, np.minimum(p.base_height, length * 0.25))
    shaft_top = top - tl
    rt = np.full(S, p.body_radius)                   # radius of the lower (trunk) part of each pillar
    tall = (length >= p.trunk_min_height) & (p.trunk_radius > p.body_radius) & on_plate
    if tall.any():                                    # never thicker than the room left by the nearest neighbour
        nn = cKDTree(pts[:, :2]).query(pts[:, :2], k=2)[0][:, 1] if S > 1 else np.full(S, 99.0)
        rt = np.where(tall, np.clip(0.5 * nn - 0.3, p.body_radius, p.trunk_radius), rt)
    zt = low + bh + (shaft_top - low - bh) * 0.5     # the trunk ends half way up the shaft, then tapers to the body
    prof = np.zeros((S, 5, 4))                       # per ring: x, y, z, radius
    prof[..., 0] = pts[:, 0, None]
    prof[..., 1] = pts[:, 1, None]
    prof[:, 0, 2] = np.where(inner, low - p.tip_sink, low)           # inner pillars: lower tip sinks into the model
    prof[:, 0, 3] = np.where(inner, p.tip_radius, p.base_radius)
    prof[:, 1, 2], prof[:, 1, 3] = low + bh, rt
    prof[:, 2, 2], prof[:, 2, 3] = zt, rt
    prof[:, 3, 2], prof[:, 3, 3] = shaft_top, p.body_radius
    prof[:, 4, 2], prof[:, 4, 3] = top + p.tip_sink, p.tip_radius   # tip sinks slightly into the model
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)
    verts = (prof[:, :, None, :3] + prof[:, :, None, 3:4] * ring[None, None]).reshape(-1, 3)
    i = np.arange(sides)
    j = (i + 1) % sides
    f = []
    for r in range(4):
        qa, qb, qc, qd = r * sides + i, r * sides + j, (r + 1) * sides + j, (r + 1) * sides + i
        f += [np.stack([qa, qb, qc], 1), np.stack([qa, qc, qd], 1)]
    f.append(np.stack([np.zeros(sides - 2, int), i[2:], i[1:-1]], 1))                       # bottom cap
    f.append(np.stack([np.full(sides - 2, 4 * sides), 4 * sides + i[1:-1], 4 * sides + i[2:]], 1))  # top cap
    f = np.concatenate(f)
    faces = ((np.arange(S) * 5 * sides)[:, None, None] + f[None]).reshape(-1, 3)
    sup = trimesh.Trimesh(verts, faces, process=False)
    if p.cross_radius > 0:
        br = _braces(mesh, pts, low, bh, shaft_top, p)
        if br is not None:
            sup = trimesh.util.concatenate([sup, br])
    if p.raft_height > 0 and on_plate.any():
        sup = trimesh.util.concatenate([_raft(pts[on_plate, :2], p), sup])
    return sup, S


def build_supports_each(meshes, p: SupportParams, sides: int = 8):
    """Supports built object by object: every object gets its own pillars AND its own raft, so the rafts of
    different objects are never merged into a single slab. Returns (support_mesh | None, total count)."""
    parts, total = [], 0
    for m in meshes:
        sup, n = build_supports(m, p, sides)
        if sup is not None:
            parts.append(sup)
            total += n
    return (trimesh.util.concatenate(parts) if parts else None), total


PILLAR_VERTS = 40                  # a pillar from build_supports: 5 rings of 8 vertices (the raft is one bigger piece)
OLD_PILLAR_VERTS = 32              # pillars of supports saved before the trunk rings


def split_pillars(sup: trimesh.Trimesh):
    """Find the single pillars inside a support mesh (they are separate shells of PILLAR_VERTS vertices).
    Returns (label per vertex, ids of the pillar shells, bottom points, top points)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    V, f = len(sup.vertices), np.asarray(sup.faces)
    g = coo_matrix((np.ones(2 * len(f)), (np.r_[f[:, 0], f[:, 1]], np.r_[f[:, 1], f[:, 2]])), shape=(V, V))
    nc, lab = connected_components(g, directed=False)
    v = np.asarray(sup.vertices, float)
    cnt = np.bincount(lab, minlength=nc)
    lo, hi = np.full((nc, 3), np.inf), np.full((nc, 3), -np.inf)
    np.minimum.at(lo, lab, v)
    np.maximum.at(hi, lab, v)
    ids = np.flatnonzero(((cnt == PILLAR_VERTS) | (cnt == OLD_PILLAR_VERTS)) & ((hi[:, :2] - lo[:, :2]).max(axis=1) < 4.0))
    c = (lo[ids] + hi[ids]) / 2
    return lab, ids, np.c_[c[:, :2], lo[ids, 2]], np.c_[c[:, :2], hi[ids, 2]]


def drop_pillars(sup: trimesh.Trimesh, lab: np.ndarray, shells) -> trimesh.Trimesh:
    """The support mesh without the pillar shells `shells` (ids from split_pillars), and without the braces that
    were attached to them."""
    shells = np.asarray(list(shells))
    v, f = np.asarray(sup.vertices, float), np.asarray(sup.faces)
    drop = np.isin(np.arange(lab.max() + 1), shells)
    if len(shells) and (np.bincount(lab) == BRACE_VERTS).any():
        gone = np.array([(v[lab == s_].min(0)[:2] + v[lab == s_].max(0)[:2]) / 2 for s_ in shells])
        for c in np.flatnonzero(np.bincount(lab) == BRACE_VERTS):
            idx = np.flatnonzero(lab == c)
            ends = np.stack([v[idx[:BRACE_SIDES], :2].mean(0), v[idx[BRACE_SIDES:], :2].mean(0)])
            if (np.linalg.norm(ends[:, None] - gone[None], axis=2) < 0.3).any():
                drop[c] = True
    out = trimesh.Trimesh(v, f[~drop[lab[f[:, 0]]]], process=False)
    out.remove_unreferenced_vertices()
    return out
