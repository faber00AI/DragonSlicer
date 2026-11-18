"""Print orientation advisor (resin): tries many orientations and ranks them.

Per candidate it measures the overhang area that needs supports, the isolated low points (islands), the largest layer
cross-section (a proxy for the peel force) and the height. The numbers are normalised across the candidates and mixed
with fixed weights, lower score = better. Only the rotation is changed: shape and scale are untouched."""
import numpy as np
import trimesh

from . import modeling as md

WEIGHTS = dict(support=0.40, islands=0.20, peel=0.30, height=0.10)


def _fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def candidate_downs(mesh, n_sphere=48, n_facets=20, merge_deg=4.0):
    """Unit vectors that may point 'down' (towards the plate): the biggest flat facets of the hull plus an even sphere."""
    ds = [np.array([0.0, 0.0, -1.0])]                                     # current orientation first (the baseline)
    try:
        hull = mesh.convex_hull
        order = np.argsort(-np.asarray(hull.facets_area))[:n_facets]
        ds += [np.asarray(hull.facets_normal[i], float) for i in order]
    except Exception:
        pass
    ds += list(_fibonacci(n_sphere))
    out = []
    cos_min = np.cos(np.radians(merge_deg))
    for d in ds:
        d = d / np.linalg.norm(d)
        if all(float(d @ o) < cos_min for o in out):
            out.append(d)
    return out


def _section_areas(V, F, N, levels):
    """Cross-section area of a closed mesh at each z in `levels` (holes count negative), vectorised."""
    tri = V[F]                                            # (f, 3, 3)
    z = tri[:, :, 2]
    zlo, zhi = z.min(axis=1), z.max(axis=1)
    out = np.zeros(len(levels))
    for k, lv in enumerate(levels):
        m = (zlo < lv) & (zhi > lv)
        if not m.any():
            continue
        t, n = tri[m], N[m]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        e = ((a, b), (b, c), (c, a))
        cross, pts = [], []
        for p, q in e:
            dz = q[:, 2] - p[:, 2]
            ok = (p[:, 2] - lv) * (q[:, 2] - lv) < 0
            s = np.where(ok, (lv - p[:, 2]) / np.where(dz == 0, 1, dz), 0)
            cross.append(ok)
            pts.append(p + s[:, None] * (q - p))
        P = np.where(cross[0][:, None], pts[0], pts[1])
        Q = np.where(cross[2][:, None], pts[2], pts[1])
        d = Q[:, :2] - P[:, :2]
        right = np.c_[d[:, 1], -d[:, 0]]
        sign = np.where((right * n[:, :2]).sum(axis=1) >= 0, 1.0, -1.0)
        out[k] = (sign * 0.5 * (P[:, 0] * Q[:, 1] - Q[:, 0] * P[:, 1])).sum()
    return out


def _evaluate(small, R, crit_deg):
    V = small.vertices @ R.T
    N = small.face_normals @ R.T
    F = small.faces
    A = small.area_faces
    zmin = V[:, 2].min()
    height = float(V[:, 2].max() - zmin)
    fz = V[F][:, :, 2].mean(axis=1)
    down = N[:, 2] < -np.sin(np.radians(crit_deg))
    on_plate = (N[:, 2] < -0.99) & (fz < zmin + 0.3)
    support_area = float(A[down & ~on_plate].sum())
    # islands: vertices lower than all their neighbours that are not resting on the plate
    ed = small.edges_unique
    za, zb = V[ed[:, 0], 2], V[ed[:, 1], 2]
    has_lower = np.zeros(len(V), bool)
    has_lower[ed[za > zb, 0]] = True
    has_lower[ed[zb > za, 1]] = True
    islands = int(((~has_lower) & (V[:, 2] > zmin + 0.5)).sum())
    levels = np.linspace(zmin + 0.1, zmin + height - 0.1, 24) if height > 0.4 else np.array([zmin + height / 2])
    peel = float(np.abs(_section_areas(V, F, N, levels)).max())
    return dict(support=support_area, islands=islands, peel=peel, height=height)


def analyse(mesh, crit_deg=30.0, max_faces=8000):
    """Rank orientations of `mesh` (world mesh of the object). Returns a list of dicts, best first; each has `R`
    (3x3 rotation to apply on top of the current one), the raw metrics and `score`. The first item of `baseline`
    in the result tuple is the current orientation's metrics."""
    small = mesh
    if len(mesh.faces) > max_faces:
        try:
            small = md.simplify(mesh, max_faces)
        except Exception:
            small = mesh
    small = trimesh.Trimesh(small.vertices - small.centroid, small.faces, process=False)
    res = []
    for d in candidate_downs(small):
        R = trimesh.geometry.align_vectors(d, [0.0, 0.0, -1.0])[:3, :3]
        m = _evaluate(small, R, crit_deg)
        m["R"] = R
        res.append(m)
    base = dict(res[0])
    mx = {k: max(r[k] for r in res) or 1.0 for k in WEIGHTS}
    for r in res:
        r["score"] = sum(w * r[k] / mx[k] for k, w in WEIGHTS.items())
    res.sort(key=lambda r: r["score"])
    return res, base
