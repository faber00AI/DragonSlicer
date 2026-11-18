"""Alignment pins for the two halves of a cut: round pins and matching holes, alternating between the halves.

Pins are placed inside the cross-section of the cut, as far from the outline (and from each other) as possible; each
pin is only kept when the hole it needs fits entirely inside the material of the other half."""
import numpy as np
import trimesh

from . import modeling as md

OVERLAP = 0.6        # a pin starts this far inside its own half, so it is fused to it
BOTTOM = 0.8         # holes are this much deeper than the pin is long: the halves close flat


def _frame(n):
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _inside_and_distance(P, segs):
    """Even-odd inside test and distance to the outline for 2D points P (k,2) against segments (m,2,2)."""
    a, b = segs[:, 0], segs[:, 1]
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    inside = np.zeros(len(P), bool)
    dist = np.full(len(P), np.inf)
    for s in range(0, len(P), 400):
        p = P[s:s + 400]
        py, px = p[:, 1:2], p[:, 0:1]
        straddle = (a[:, 1] > py) != (b[:, 1] > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = a[:, 0] + (py - a[:, 1]) * ab[:, 0] / np.where(ab[:, 1] == 0, 1, ab[:, 1])
        inside[s:s + 400] = ((straddle & (xi > px)).sum(axis=1) % 2) == 1
        t = np.clip(((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / ab2, 0, 1)
        dx = a[:, 0] + t * ab[:, 0] - px
        dy = a[:, 1] + t * ab[:, 1] - py
        dist[s:s + 400] = np.sqrt(dx * dx + dy * dy).min(axis=1)
    return inside, dist


def pin_sites(mesh, n, offset, radius, margin=1.5, max_sites=30):
    """Candidate pin centres (3D, on the cut plane) ordered by preference: the roomiest spot first, then each next one
    as far as possible from those already listed. Returns [] if the section is too thin for any pin."""
    n = np.asarray(n, float)
    origin = n * offset
    segs3 = trimesh.intersections.mesh_plane(mesh, n, origin)
    if segs3 is None or len(segs3) == 0:
        return []
    u, v = _frame(n)
    segs = np.stack([np.c_[(segs3[:, :, :] - origin) @ u].reshape(-1, 2, 1)[:, :, 0],
                     np.c_[(segs3[:, :, :] - origin) @ v].reshape(-1, 2, 1)[:, :, 0]], axis=2)
    lo, hi = segs.reshape(-1, 2).min(axis=0), segs.reshape(-1, 2).max(axis=0)
    step = max(0.5, float(max(hi - lo)) / 110)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + step, step), np.arange(lo[1], hi[1] + step, step))
    P = np.c_[gx.ravel(), gy.ravel()]
    inside, dist = _inside_and_distance(P, segs)
    ok = inside & (dist >= radius + margin)
    if not ok.any():
        return []
    P, dist = P[ok], dist[ok]
    chosen = [int(dist.argmax())]
    near = np.linalg.norm(P - P[chosen[0]], axis=1)
    while len(chosen) < max_sites:
        k = int(near.argmax())
        if near[k] < 2.5 * radius * 2:                    # keep pins at least 2.5 diameters apart
            break
        chosen.append(k)
        near = np.minimum(near, np.linalg.norm(P - P[k], axis=1))
    return [origin + u * P[k, 0] + v * P[k, 1] for k in chosen]


def _cyl(radius, z0, z1, n, point):
    c = trimesh.creation.cylinder(radius=radius, height=z1 - z0, sections=32)
    c.apply_translation([0, 0, (z0 + z1) / 2])
    T = trimesh.geometry.align_vectors([0, 0, 1], n)
    T[:3, 3] = point
    c.apply_transform(T)
    return c


def _pin(radius, length, n, point, direction):
    """Round pin standing on the plane, pointing along direction*n, with a narrower tip so it enters easily."""
    s = direction
    body = _cyl(radius, *sorted((-s * OVERLAP, s * (length - 1.0))), n, point)
    tip = _cyl(radius * 0.72, *sorted((s * (length - 1.2), s * length)), n, point)
    return trimesh.boolean.union([body, tip], engine=md.ENGINE)


def _hole(radius, depth, n, point, direction):
    s = direction
    return _cyl(radius, *sorted((-s * 0.5, s * depth)), n, point)


def _fits(mesh, radius, depth, n, point, direction):
    """True when a hole of this size drilled from `point` along direction*n lies entirely in the material of `mesh`."""
    u, v = _frame(n)
    pts = [point + direction * n * d for d in np.linspace(0.3, depth, 4)]
    ring = [p + r * (u * np.cos(a) + v * np.sin(a)) for p in pts for r in (radius,) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    try:
        return bool(mesh.contains(np.array(pts + ring)).all())
    except Exception:
        return True


def add_pins(pos, neg, n, offset, count=4, diameter=4.0, clearance=0.2, margin=1.5, sites=None):
    """Add alignment pins/holes to the two halves of a cut. Returns (pos, neg, placed) with `placed` = number of pins."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    r, length = diameter / 2, max(5.0, diameter * 1.5)
    depth = length + BOTTOM
    if sites is None:
        sites = pin_sites(pos, n, offset, r + clearance, margin=margin)
    placed = 0
    pos_add, pos_cut, neg_add, neg_cut = [], [], [], []
    for p in sites:
        if placed >= count:
            break
        if placed % 2 == 0:       # pin on the + half (points into -n) and hole in the - half
            if not _fits(neg, r + clearance, depth, n, p, -1):
                continue
            pos_add.append(_pin(r, length, n, p, -1))
            neg_cut.append(_hole(r + clearance, depth, n, p, -1))
        else:                     # pin on the - half (points into +n) and hole in the + half
            if not _fits(pos, r + clearance, depth, n, p, +1):
                continue
            neg_add.append(_pin(r, length, n, p, +1))
            pos_cut.append(_hole(r + clearance, depth, n, p, +1))
        placed += 1
    if pos_add:
        pos = md._clean(trimesh.boolean.union([pos] + pos_add, engine=md.ENGINE)) or pos
    if neg_add:
        neg = md._clean(trimesh.boolean.union([neg] + neg_add, engine=md.ENGINE)) or neg
    if pos_cut:
        pos = md._clean(trimesh.boolean.difference([pos] + pos_cut, engine=md.ENGINE)) or pos
    if neg_cut:
        neg = md._clean(trimesh.boolean.difference([neg] + neg_cut, engine=md.ENGINE)) or neg
    return pos, neg, placed
