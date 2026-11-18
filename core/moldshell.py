"""Contoured silicone mold: a shell that hugs the master at a fixed distance (the silicone thickness), split in two
halves, standing in a groove of a base plate, with a pour spout and air vents on the ceiling of the cavity.

Everything is built from distance fields on a voxel grid (scipy EDT + marching cubes), so any mesh works and the
smoothing is a morphological closing: gaps and dents narrower than 2*smooth are filled, which also avoids undercuts.
Frame: the master stands on the plane z = 0 (its lowest point), the base is below it. Meshes are returned in that frame."""
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure

from . import modeling as md
from .cutkeys import add_pins

RES = {"Veloce": 240, "Normale": 320, "Fine": 420}        # voxels along the longest side of the grid


@dataclass
class ShellParams:
    silicone: float = 10.0       # silicone thickness around the master
    smooth: float = 6.0          # closing radius: bigger = simpler shell, fewer undercuts
    wall: float = 5.0            # shell wall thickness
    clearance: float = 0.25      # play between shell rim and base groove (per side)
    base: float = 6.0            # base plate thickness
    groove: float = 3.0          # depth of the groove (the shell rim sinks this far)
    margin: float = 8.0          # base margin beyond the shell
    pocket: float = 2.0          # depth of the recess in the base that receives the master (0 = none)
    seam_w: float = 10.0         # width of the flat flange along the seam (surface for clamps and pins)
    seam_t: float = 3.0          # thickness of the flange on each half
    spout: float = 14.0          # pour spout diameter
    vents: bool = True           # air vents on the other highs of the ceiling
    vent_d: float = 2.5
    split_axis: str = "X"        # the shell is cut by the plane through the master's centre perpendicular to this axis
    pins: bool = True            # alignment pins between the halves
    pin_d: float = 2.5
    pin_n: int = 4
    res: str = "Normale"


def _voxelize(master, pitch):
    v = master.voxelized(pitch).fill()
    return np.asarray(v.matrix, bool), np.asarray(v.translation, float)


def _cyl(r, z0, z1, x, y):
    c = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=24)
    c.apply_translation([x, y, (z0 + z1) / 2])
    return c


def _mc(field, level, lo, pitch):
    """Closed surface of {field > level} (positive inside); the border of the array must be below `level`."""
    # level nudged off exact lattice values: a field that is exactly `level` on lattice points gives degenerate triangles
    verts, faces, _, _ = measure.marching_cubes(field, level=level + 0.0023, spacing=(pitch, pitch, pitch))
    m = trimesh.Trimesh(verts + lo, faces[:, ::-1], process=True)
    m.fix_normals()
    return m


def _peaks(H, pitch, radius_mm, min_prominence=2.0):
    """Plateaus that are local maxima of the cavity ceiling height map H (NaN = no cavity). [(row, col, z)] high first."""
    valid = ~np.isnan(H)
    Hf = np.where(valid, H, -1e9)
    r = max(1, int(round(radius_mm / pitch)))
    mx = ndi.maximum_filter(Hf, size=2 * r + 1)
    mn = ndi.minimum_filter(np.where(valid, H, 1e9), size=2 * r + 1)
    mask = valid & (Hf >= mx - 1e-6) & (Hf - mn >= min_prominence)
    lab, n = ndi.label(mask)
    out = []
    for k in range(1, n + 1):
        rr, cc = np.nonzero(lab == k)
        out.append((float(rr.mean()), float(cc.mean()), float(H[rr[0], cc[0]]), len(rr)))
    out.sort(key=lambda t: (-t[2], -t[3]))
    return out


def build(master, p: ShellParams) -> dict:
    """master: world mesh. Returns {'base', 'shells': [mesh, mesh], 'info'} (see module doc for the frame)."""
    if p.groove >= p.silicone:
        raise ValueError("L'incastro deve essere meno profondo dello spessore del silicone")
    if p.base <= p.groove + 1:
        raise ValueError("La base deve essere più spessa dell'incastro")
    if p.pocket > 0 and p.pocket + p.clearance > p.base - 1.5:
        raise ValueError("L'incastro del modello è troppo profondo per lo spessore della base")
    master = master.copy()
    master.apply_translation([0, 0, -master.bounds[0][2] - p.pocket])      # the master sinks `pocket` into the base
    ext = master.extents
    g, s, w, c = p.silicone, p.smooth, p.wall, p.clearance
    pad_xy = g + s + w + c + 6
    big = float(ext.max() + 2 * pad_xy)
    pitch = max(0.5, big / RES.get(p.res, 320))
    dims = np.array([ext[0] + 2 * pad_xy, ext[1] + 2 * pad_xy, ext[2] + 2 * (g + s + w + p.groove) + 12])
    while float(np.prod(dims / pitch)) > 3.5e7:                       # keep memory in check on big models
        pitch *= 1.1

    solid_small, lat0 = _voxelize(master, pitch)
    npad = int(np.ceil(pad_xy / pitch)) + 2
    zdn = int(np.ceil((p.groove + w + 3) / pitch)) + 2
    zup = int(np.ceil((g + s + w + 6) / pitch)) + 2
    sh = solid_small.shape
    shape = (sh[0] + 2 * npad, sh[1] + 2 * npad, sh[2] + zdn + zup)
    lo = lat0 - np.array([npad, npad, zdn]) * pitch
    solid = np.zeros(shape, bool)
    solid[npad:npad + sh[0], npad:npad + sh[1], zdn:zdn + sh[2]] = solid_small

    # signed distance of the master, then the smoothed cavity: closing = dilate by g+s, erode by s
    d_out = ndi.distance_transform_edt(~solid, sampling=pitch).astype(np.float32)
    A = d_out <= g + s
    cav = ndi.distance_transform_edt(A, sampling=pitch) > s
    cav |= solid
    del A
    z = (lo[2] + np.arange(shape[2]) * pitch).astype(np.float32)[None, None, :]
    sd_c = (ndi.distance_transform_edt(~cav, sampling=pitch) - ndi.distance_transform_edt(cav, sampling=pitch)).astype(np.float32)
    sd_c = ndi.gaussian_filter(sd_c, 0.7)

    pocket = None
    if p.pocket > 0:                                       # recess in the base: the master's own shape + play, open at the top
        f_pocket = np.minimum(c - d_out, 0.3 - z)
        f_pocket[:, :, -1] = -1
        f_pocket[0] = f_pocket[-1] = f_pocket[:, 0] = f_pocket[:, -1] = -1
        pocket = _mc(f_pocket, 0.0, lo, pitch)
        del f_pocket
    del d_out
    border = np.zeros(shape, bool)
    border[0] = border[-1] = border[:, 0] = border[:, -1] = border[:, :, 0] = border[:, :, -1] = True

    # ---- shell: 0 < sd_c < wall, plus a flat flange along the seam, above -groove ----
    axis = {"X": 0, "Y": 1}.get(p.split_axis.upper(), 0)
    n = np.zeros(3)
    n[axis] = 1.0
    off = float((master.bounds[0][axis] + master.bounds[1][axis]) / 2)
    coord = lo[axis] + np.arange(shape[axis], dtype=np.float32) * pitch
    cshape = [1, 1, 1]
    cshape[axis] = shape[axis]
    dplane = np.abs(coord.reshape(cshape) - off)
    h = p.seam_t
    region = np.maximum(np.minimum(w - sd_c, sd_c), np.minimum(np.minimum(w + p.seam_w - sd_c, sd_c), h - dplane))
    # pins: the flange stays as thin as asked; where a pin sits, a round boss thickens it just enough to hold the hole
    sites3, boss_h = [], 0.0
    if p.pins:
        pin_r, pin_len = p.pin_d / 2, max(5.0, p.pin_d * 1.5)
        need = pin_len + 0.8 + 1.0                                          # hole depth + a solid bottom
        rb = pin_r + 0.2 + 1.5
        i0 = int(round((off - lo[axis]) / pitch))
        sl = region[i0] if axis == 0 else region[:, i0]
        other = 1 - axis
        ca = lo[other] + np.arange(sl.shape[0]) * pitch
        cz = lo[2] + np.arange(sl.shape[1]) * pitch
        dist = ndi.distance_transform_edt(sl > 0, sampling=pitch)
        ok = (dist >= pin_r + 1.2) & (cz[None, :] >= rb + 0.5)
        ia, iz = np.nonzero(ok)
        if len(ia):
            P = np.c_[ca[ia], cz[iz]]
            dd0 = dist[ia, iz]
            chosen = [int(dd0.argmax())]
            near = np.linalg.norm(P - P[chosen[0]], axis=1)
            while len(chosen) < p.pin_n:
                k = int(near.argmax())
                if near[k] < 2 * rb + 4:
                    break
                chosen.append(k)
                near = np.minimum(near, np.linalg.norm(P - P[k], axis=1))
            for k in chosen:
                a_, z_ = P[k]
                site = np.zeros(3)
                site[axis], site[other], site[2] = off, a_, z_
                sites3.append(site)
        if sites3 and h < need:
            boss_h = need + 0.5
            Aco = (lo[other] + np.arange(shape[other], dtype=np.float32) * pitch)
            Zco = lo[2] + np.arange(shape[2], dtype=np.float32) * pitch
            for site in sites3:
                if axis == 0:
                    dd = np.hypot(Aco[None, :, None] - site[1], Zco[None, None, :] - site[2])
                else:
                    dd = np.hypot(Aco[:, None, None] - site[0], Zco[None, None, :] - site[2])
                boss = np.minimum(np.minimum(rb - dd, boss_h - dplane), sd_c)
                np.maximum(region, boss, out=region)
    f_shell = np.minimum(region, z + p.groove)
    f_shell[border] = -1
    shell = _mc(f_shell, 0.0, lo, pitch)

    # ---- spout and vents on the cavity ceiling ----
    cav_up = cav & (z >= 0)
    top_idx = np.where(cav_up.any(axis=2), shape[2] - 1 - np.argmax(cav_up[:, :, ::-1], axis=2), -1)
    H = np.where(top_idx >= 0, z[0, 0][np.clip(top_idx, 0, None)], np.nan)
    peaks = _peaks(H, pitch, 10.0)
    if not peaks:
        i, j = np.unravel_index(np.nanargmax(H), H.shape)
        peaks = [(float(i), float(j), float(H[i, j]), 1)]
    holes = []
    pos = lambda r_, c_: (lo[0] + r_ * pitch, lo[1] + c_ * pitch)
    sx, sy = pos(peaks[0][0], peaks[0][1])
    sz = peaks[0][2]
    top = float(shell.bounds[1][2]) + 1.0                     # holes go through everything: the seam flange can stand above the roof
    holes.append(_cyl(p.spout / 2, sz - 3, top, sx, sy))
    n_vents = 0
    if p.vents:
        for r_, c_, hz, _ in peaks[1:8]:
            x, y = pos(r_, c_)
            if np.hypot(x - sx, y - sy) < p.spout / 2 + 6:
                continue
            holes.append(_cyl(p.vent_d / 2, hz - 1.5, top, x, y))
            n_vents += 1
    shell = md._clean(trimesh.boolean.difference([shell] + holes, engine=md.ENGINE)) or shell

    # ---- split in two ----
    pa, pb = md.cut(shell, n, off)
    pins = 0
    if pa is not None and pb is not None and p.pins:
        pa, pb, pins = add_pins(pa, pb, n, off, p.pin_n, p.pin_d, 0.2, margin=0.8, sites=sites3)
    shells = [m for m in (pa, pb) if m is not None] or [shell]

    # ---- base: plate minus the groove that receives the shell rim ----
    zz = np.broadcast_to(z, shape)
    f_groove = np.minimum(region + c, np.minimum(zz + p.groove, 0.5 - zz))
    f_groove[border] = -1
    groove = _mc(f_groove, 0.0, lo, pitch)
    rim = shell.bounds
    gx0, gy0 = rim[0][:2] - p.margin
    gx1, gy1 = rim[1][:2] + p.margin
    plate = trimesh.creation.box((gx1 - gx0, gy1 - gy0, p.base))
    plate.apply_translation([(gx0 + gx1) / 2, (gy0 + gy1) / 2, -p.base / 2])
    base = md._clean(trimesh.boolean.difference([plate, groove] + ([pocket] if pocket is not None else []),
                                                engine=md.ENGINE)) or plate

    vox = pitch ** 3 / 1000.0
    sil = float((cav & (z >= 0)).sum() - (solid & (z >= 0)).sum()) * vox
    info = dict(seam_t=p.seam_t, boss_h=boss_h, pocket=p.pocket, silicone_ml=sil, master_ml=float(solid.sum()) * vox, vents=n_vents, pins=pins, pitch=pitch,
                spout_at=(float(sx), float(sy), float(sz)), parts=len(shells),
                shell=tuple(float(v) for v in shell.extents), base=tuple(float(v) for v in base.extents))
    return dict(base=base, shells=shells, info=info)
