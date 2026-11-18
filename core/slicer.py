"""MSLA slicer: mesh -> one 8-bit mask per layer."""
import numpy as np
import trimesh

from .machine import Machine


def crossings(segs: np.ndarray, W: int, H: int):
    """Scanline crossings of 2D segments (pixel coords).
    Returns (rows, cols, dirs) sorted by (row, x); dirs is +1 where the segment, as given, runs towards +y, else -1.
    Consecutive crossings of a row bound filled spans (parity), or are weighted by dirs (winding)."""
    if len(segs) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int8)
    x0, y0, x1, y1 = segs[:, 0, 0], segs[:, 0, 1], segs[:, 1, 0], segs[:, 1, 1]
    ymin, ymax = np.minimum(y0, y1), np.maximum(y0, y1)
    rlo = np.clip(np.ceil(ymin - 0.5), 0, H).astype(np.int64)
    rhi = np.clip(np.ceil(ymax - 0.5), 0, H).astype(np.int64)
    cnt = np.maximum(rhi - rlo, 0)
    total = int(cnt.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int8)
    seg = np.repeat(np.arange(len(segs)), cnt)
    start = np.cumsum(cnt) - cnt
    rows = rlo[seg] + (np.arange(total) - np.repeat(start, cnt))
    dy = y1[seg] - y0[seg]
    t = (rows + 0.5 - y0[seg]) / np.where(dy == 0, 1, dy)
    xs = x0[seg] + t * (x1[seg] - x0[seg])
    dirs = np.where(dy > 0, 1, -1).astype(np.int8)
    order = np.lexsort((xs, rows))
    rows, xs, dirs = rows[order], xs[order], dirs[order]
    return rows, np.clip(np.ceil(xs - 0.5), 0, W).astype(np.int64), dirs


def _merge(st, en):
    nz = en > st
    st, en = st[nz], en[nz]
    if len(st) > 1:
        brk = st[1:] != en[:-1]                      # spans that touch are one run
        st = st[np.concatenate(([True], brk))]
        en = en[np.concatenate((brk, [True]))]
    return st, en


def filled_intervals(segs: np.ndarray, W: int, H: int, winding: bool = False):
    """Filled spans as (starts, ends) over the flattened image (row*W + col), merged across row ends.
    winding=False: even-odd parity; a row with an odd number of crossings (open mesh) drops its last crossing
    instead of leaking to the end of the row.
    winding=True: `segs` must be oriented (interior on a fixed side, see _orient); a pixel is filled where the
    winding number is not 0, so overlapping closed shells (supports touching the model or each other) are a
    union instead of cancelling out, and an inverted inner shell is a cavity. A layer whose rows do not balance
    (open or inconsistent mesh) falls back to parity."""
    rows, cols, dirs = crossings(segs, W, H)
    if len(rows) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    idx = np.arange(len(rows))
    first = np.searchsorted(rows, rows, side="left")
    last = np.searchsorted(rows, rows, side="right") - 1
    lin = rows * W + cols
    if winding:
        run = np.cumsum(dirs, dtype=np.int64)
        run = run - (run[first] - dirs[first])       # winding number after each crossing, per row
        if not (run[last] != 0).any():
            on = (run != 0) & (idx < last)
            return _merge(lin[on], lin[np.minimum(idx + 1, len(lin) - 1)][on])
    k = idx - first
    keep = ~(((last - first + 1) % 2 == 1) & (idx == last))
    lin, k = lin[keep], k[keep]
    return _merge(lin[k % 2 == 0], lin[k % 2 == 1])


def raster_even_odd(segs: np.ndarray, W: int, H: int, winding: bool = False) -> np.ndarray:
    """Fill the interior of 2D segments into a (H, W) uint8 mask (255 inside); see filled_intervals."""
    st, en = filled_intervals(segs, W, H, winding)
    if len(st) == 0:
        return np.zeros((H, W), np.uint8)
    diff = np.zeros(H * W + 1, np.int8)                  # spans are disjoint and merged: starts and ends are all distinct
    diff[st] = 1
    diff[en] = -1
    return (diff.cumsum(dtype=np.int8)[:H * W].reshape(H, W) != 0).view(np.uint8) * np.uint8(255)


def layer_count(mesh: trimesh.Trimesh, layer_height: float) -> int:
    return max(1, int(np.ceil(mesh.bounds[1][2] / layer_height - 1e-6)))


def _to_pixels(xy: np.ndarray, m: Machine, W: int, H: int) -> np.ndarray:
    px = (xy[..., 0] + m.bed_x / 2) / m.bed_x * W
    py = (m.bed_y / 2 - xy[..., 1]) / m.bed_y * H
    return np.stack([px, py], axis=-1)


def _orient(world_xy: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Turn each section segment so the face's outward normal is on its right: closed outward-facing shells then all
    run the same way round, which is what the winding fill needs."""
    d = np.stack([-normals[:, 1], normals[:, 0]], 1)
    flip = np.einsum("ij,ij->i", world_xy[:, 1] - world_xy[:, 0], d) < 0
    out = world_xy.copy()
    out[flip] = world_xy[flip][:, ::-1]
    return out


class _Mesh:
    """Per-slice cache of the triangle arrays the section code needs."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.tri = np.asarray(mesh.vertices, np.float64)[np.asarray(mesh.faces)]        # (F, 3, 3)
        self.zmin, self.zmax = self.tri[:, :, 2].min(1), self.tri[:, :, 2].max(1)
        self.normals = np.asarray(mesh.face_normals)[:, :2]
        self.top = float(mesh.bounds[1][2]) - 1e-6


def _plane_heights(m: _Mesh, layer_height: float, part):
    return np.minimum([(i + 0.5) * layer_height for i in part], m.top)


def _cut(m: _Mesh, zp: np.ndarray):
    """Segments where the horizontal planes `zp` (ascending) cut the triangles, one list per plane.
    Only the triangles that span a plane are touched (cost ~ number of crossings, not faces x planes). A vertex lying
    exactly on a plane counts as below it, so every section is a closed loop. Returns [(xy (n,2,2), face (n,))]."""
    cand = np.flatnonzero((m.zmax > zp[0]) & (m.zmin <= zp[-1]))
    lo = np.searchsorted(zp, m.zmin[cand], side="left")             # first plane at or above the face's lowest vertex
    hi = np.searchsorted(zp, m.zmax[cand], side="left")             # first plane at or above its highest: not crossed
    cnt = np.maximum(hi - lo, 0)
    face = np.repeat(cand, cnt)
    start = np.cumsum(cnt) - cnt
    plane = np.repeat(lo, cnt) + np.arange(len(face)) - np.repeat(start, cnt)
    order = np.argsort(plane, kind="stable")
    face, plane = face[order], plane[order]
    tri = m.tri[face]                                                # (P, 3, 3)
    d = tri[:, :, 2] - zp[plane][:, None]
    above = d > 0
    nxt = [1, 2, 0]
    cross = above != above[:, nxt]                                   # edge i -> i+1 crosses the plane
    e = np.argsort(~cross, axis=1, kind="stable")[:, :2]             # the two crossing edges
    r = np.arange(len(face))[:, None]
    a, b = tri[r, e], tri[r, np.array(nxt)[e]]
    da, db = d[r, e], d[r, np.array(nxt)[e]]
    pts = (a + (b - a) * (da / (da - db))[..., None])[..., :2]       # (P, 2, 2)
    bounds = np.searchsorted(plane, np.arange(len(zp) + 1))
    return [(pts[bounds[k]:bounds[k + 1]], face[bounds[k]:bounds[k + 1]]) for k in range(len(zp))]


def _sections(m: _Mesh, machine, layer_height, part, W, H, flip_x=False, flip_y=False):
    """Yield the oriented pixel-space section segments of each layer in `part`. flip_x / flip_y mirror the image
    (used for printer files; the winding fill is indifferent to the resulting reversed orientation)."""
    for xy, fi in _cut(m, _plane_heights(m, layer_height, part)):
        if not len(xy):
            yield np.empty((0, 2, 2))
            continue
        px = _to_pixels(_orient(xy, m.normals[fi]), machine, W, H)
        if flip_x:
            px[..., 0] = W - px[..., 0]
        if flip_y:
            px[..., 1] = H - px[..., 1]
        yield px


def slice_layers(mesh: trimesh.Trimesh, machine: Machine, layer_height: float,
                 W: int | None = None, H: int | None = None, indices=None, chunk: int = 24):
    """Yield (layer_index, mask) for every requested layer. W/H allow downscaled previews."""
    W = W or machine.res_x
    H = H or machine.res_y
    n = layer_count(mesh, layer_height)
    idx = list(range(n)) if indices is None else list(indices)
    m = _Mesh(mesh)
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        for i, segs in zip(part, _sections(m, machine, layer_height, part, W, H)):
            yield i, raster_even_odd(segs, W, H, winding=True)


def slice_intervals(mesh: trimesh.Trimesh, machine: Machine, layer_height: float, indices, chunk: int = 24,
                    flip_x=False, flip_y=False):
    """Yield (layer_index, starts, ends): filled spans of each layer at full machine resolution."""
    W, H = machine.res_x, machine.res_y
    idx = list(indices)
    m = _Mesh(mesh)
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        for i, segs in zip(part, _sections(m, machine, layer_height, part, W, H, flip_x, flip_y)):
            yield (i, *filled_intervals(segs, W, H, winding=True))


def voxelize(mesh: trimesh.Trimesh, pitch: float, pad: int = 2):
    """Solid occupancy of a closed mesh on a cubic grid: (bool array [z, y, x], origin of voxel [0, 0, 0]'s corner).
    Same section and winding fill as the slicer, so cavities (inverted inner shells) come out empty."""
    lo = np.asarray(mesh.bounds[0], float) - pad * pitch
    n = np.ceil((np.asarray(mesh.bounds[1], float) + pad * pitch - lo) / pitch).astype(int)
    vol = np.zeros((n[2], n[1], n[0]), bool)
    m = _Mesh(mesh)
    zs = lo[2] + (np.arange(n[2]) + 0.5) * pitch
    for a in range(0, n[2], 32):
        for k, (xy, fi) in enumerate(_cut(m, zs[a:a + 32])):
            if len(xy):
                segs = (_orient(xy, m.normals[fi]) - lo[:2]) / pitch
                vol[a + k] = raster_even_odd(segs, int(n[0]), int(n[1]), winding=True) > 0
    return vol, lo
