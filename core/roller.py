"""Texture roller: a sleeve with a relief pattern wrapped around a cylinder, plus an optional core and two handles.

The pattern is a grey-scale tile (built-in, or any image) repeated N times around and M times along the roller.
White = relief. Raised: the pattern stands out of the roller (it presses into clay); Recessed: the pattern is grooved in.
The sleeve is built directly as a surface of revolution (no booleans), so it is exact and fast. Z is the roller's axis."""
import base64
import io
import json
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage as ndi

from . import modeling as md
from .machine import APP

USER = APP / "presets" / "roller_patterns.json"


@dataclass
class RollerParams:
    diameter: float = 36.0       # outer diameter (the top of the relief)
    length: float = 90.0         # length of the patterned sleeve
    depth: float = 1.2           # relief depth
    raised: bool = True          # True: pattern sticks out; False: pattern is grooved
    repeats_u: int = 9           # pattern repeats around
    repeats_v: int = 7           # pattern repeats along
    wall: float = 4.0            # sleeve wall thickness (under the relief)
    clearance: float = 0.3       # play between sleeve and core
    bore: float = 8.0            # diameter of the core's end stubs (and of the handle sockets)
    margin: float = 2.0          # smooth rim at both ends of the sleeve
    detail: int = 40             # samples per pattern repeat
    handles: bool = True         # also make the core and two handles
    handle_d: float = 28.0
    handle_len: float = 55.0
    angle: float = 0.0           # pattern rotation in degrees (snapped to the nearest angle that still closes around the roller)


# ---------------------------------------------------------------- patterns
def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _grid(n):
    u = (np.arange(n) + 0.5) / n
    return np.meshgrid(u, u)          # (u along columns, v along rows)


def _pdist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1 - d)


def _dots(n):
    u, v = _grid(n)
    d = np.minimum(np.hypot(_pdist(u, .25), _pdist(v, .25)), np.hypot(_pdist(u, .75), _pdist(v, .75)))
    return 1 - _smooth(d, .15, .2)


def _diamonds(n):
    u, v = _grid(n)
    return 1 - _smooth(np.abs(u - .5) + np.abs(v - .5), .32, .4)


def _stripes(n):
    u, _ = _grid(n)
    return .5 + .5 * np.tanh(6 * np.sin(2 * np.pi * u * 2))


def _waves(n):
    u, v = _grid(n)
    return .5 + .5 * np.tanh(7 * np.sin(2 * np.pi * (2 * v + .12 * np.sin(2 * np.pi * u))))


def _rings(n):
    u, v = _grid(n)
    d = np.minimum(np.hypot(_pdist(u, .25), _pdist(v, .25)), np.hypot(_pdist(u, .75), _pdist(v, .75)))
    return 1 - _smooth(np.abs(d - .17), .04, .08)


def _bricks(n):
    u, v = _grid(n)
    row = np.floor(v * 2)
    uu = (u + .5 * (row % 2)) % 1.0
    mortar = np.minimum(_smooth(.07 - (v * 2 % 1.0), -.02, .02), 1) + _smooth(.05 - uu, -.02, .02)
    return 1 - np.clip(mortar, 0, 1)


def _checker(n):
    u, v = _grid(n)
    return .5 + .5 * np.tanh(20 * np.sin(2 * np.pi * u) * np.sin(2 * np.pi * v))


def _fbm(n, power, seed):
    """Seamless random field in [0, 1]: white noise filtered in the frequency domain (periodic by construction)."""
    w = np.random.default_rng(seed).standard_normal((n, n))
    f = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :])
    f[0, 0] = 1.0
    F = np.fft.fft2(w) * f ** (-power)
    F[0, 0] = 0.0
    x = np.fft.ifft2(F).real
    return (x - x.min()) / (x.max() - x.min())


def _worley(u, v, cells, seed, jitter=0.85):
    """Periodic Worley noise at points (u, v): distance to the nearest and second nearest feature point, id of the nearest."""
    rng = np.random.default_rng(seed)
    px = (np.arange(cells)[None, :] + 0.5 + (rng.random((cells, cells)) - 0.5) * jitter) / cells
    py = (np.arange(cells)[:, None] + 0.5 + (rng.random((cells, cells)) - 0.5) * jitter) / cells
    ci, cj = np.floor((u % 1.0) * cells).astype(int), np.floor((v % 1.0) * cells).astype(int)
    f1 = np.full(u.shape, 9.0)
    f2 = np.full(u.shape, 9.0)
    ident = np.zeros(u.shape, int)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            gi, gj = ci + di, cj + dj
            x = px[gj % cells, gi % cells] + (gi // cells)
            y = py[gj % cells, gi % cells] + (gj // cells)
            d = np.hypot(u % 1.0 - x, v % 1.0 - y)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            ident = np.where(closer, (gj % cells) * cells + gi % cells, ident)
            f1 = np.where(closer, d, f1)
    return f1, f2, ident


def _stones(n):
    u, v = _grid(n)
    f1, f2, ident = _worley(u, v, 5, 11)
    level = 0.8 + 0.2 * np.random.default_rng(3).random(25)[ident]
    dome = 0.8 + 0.2 * (1 - _smooth(f1, 0.02, 0.13))
    grain = 0.94 + 0.06 * _fbm(n, 1.0, 5)
    return _smooth(f2 - f1, 0.008, 0.05) * level * dome * grain


def _lava(n):
    u, v = _grid(n)
    wu = u + 0.05 * (_fbm(n, 1.6, 21) - 0.5)
    wv = v + 0.05 * (_fbm(n, 1.6, 22) - 0.5)
    f1, f2, _ = _worley(wu, wv, 4, 7, 0.95)
    crack = 1 - _smooth(f2 - f1, 0.0, 0.055)
    rough = _fbm(n, 1.2, 23)
    return (1 - crack) * (0.55 + 0.45 * rough) * (0.85 + 0.15 * _smooth(f1, 0.0, 0.2))


def _grass(n):
    rng = np.random.default_rng(31)
    hgt = np.zeros((n, n), np.float32)
    ys = np.arange(n)
    xs = np.arange(n)[None, :]
    for _ in range(190):                     # blades stand along v (the roller axis), tip at the end of the stroke
        x0, y0 = rng.random() * n, rng.random() * n
        length = n * rng.uniform(0.28, 0.6)
        width = n * rng.uniform(0.012, 0.022)
        lean = rng.uniform(-0.09, 0.09) * n
        level = rng.uniform(0.55, 1.0)
        t = np.clip(((ys - y0) % n) / length, 0, 1.5)
        on = t <= 1.0
        cx = x0 + lean * t ** 2
        hw = width * np.clip(1 - t, 0, 1) ** 0.8 + 0.25
        d = np.abs((xs - cx[:, None] + n / 2) % n - n / 2)
        cov = np.clip(1.0 - (d - hw[:, None]) / 1.2, 0, 1) * on[:, None]
        hgt = np.maximum(hgt, (cov * (level * (0.6 + 0.4 * (1 - t))[:, None])).astype(np.float32))
    return hgt

BUILTIN = {"Punti": _dots, "Rombi": _diamonds, "Righe": _stripes, "Onde": _waves, "Anelli": _rings,
           "Mattoni": _bricks, "Scacchi": _checker, "Pietre": _stones, "Lava": _lava, "Erba": _grass}


def builtin_tile(name, n=128):
    return np.clip(BUILTIN[name](n), 0, 1).astype(np.float32)


def image_tile(data: bytes, max_side=256):
    """Grey-scale float tile [0, 1] from image bytes (PNG/JPG...). Transparent pixels count as black."""
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("L")
    im.thumbnail((max_side, max_side))
    return np.asarray(im, np.float32) / 255.0


def to_png(tile) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray((np.clip(tile, 0, 1) * 255).astype(np.uint8), "L").save(buf, "PNG")
    return buf.getvalue()


def prepare(tile, invert=False, threshold=0.0, mirror=False):
    """Tile -> relief tile in [0, 1]. mirror = reflect it 2x2 so its edges always meet (seamless)."""
    t = np.asarray(tile, np.float32)
    lo, hi = float(t.min()), float(t.max())
    if hi - lo > 1e-6:
        t = (t - lo) / (hi - lo)
    if invert:
        t = 1 - t
    if threshold > 0:
        t = ndi.gaussian_filter((t >= threshold).astype(np.float32), 0.8, mode="wrap")
    if mirror:
        t = np.block([[t, t[:, ::-1]], [t[::-1], t[::-1, ::-1]]])
    return t


def lattice(ru: int, angle: float):
    """Integer vector (m, n) whose direction is closest to `angle` (degrees) and whose length is close to ru.
    Rotating the pattern so that one turn around the roller advances it by exactly (m, n) cells keeps it seamless.
    Returns (m, n, effective angle in degrees)."""
    ru = max(int(ru), 1)
    a = float(angle) % 360.0
    if a == 0.0:
        return ru, 0, 0.0
    r = int(np.ceil(ru * 1.4)) + 1
    m, n = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    ln = np.hypot(m, n)
    ok = (ln >= ru * 0.7) & (ln <= ru * 1.4)
    ang = np.degrees(np.arctan2(n, m)) % 360.0
    err = np.abs((ang - a + 180.0) % 360.0 - 180.0)
    cost = np.where(ok, err + 20.0 * np.abs(ln - ru) / ru, np.inf)       # angle first, scale change as a mild penalty
    i = np.unravel_index(np.argmin(cost), cost.shape)
    return int(m[i]), int(n[i]), float(ang[i])


def pattern_coords(x, z, ru: int, angle: float):
    """(cu, cv) tile coordinates (in cells) for a point x cells around and z cells along the roller."""
    m, n, _ = lattice(ru, angle)
    return (m * x + n * z) / ru, (-n * x + m * z) / ru


def relief_field(tile, p: RollerParams):
    """Height field (rows along z, columns around) in [0, 1] sampled at the mesh nodes, with the smooth end rims."""
    nu, nv = p.repeats_u * p.detail, p.repeats_v * p.detail
    th, tw = tile.shape
    X, Z = np.meshgrid(np.arange(nu) / p.detail, np.arange(nv + 1) / p.detail)
    cu, cv = pattern_coords(X, Z, p.repeats_u, p.angle)
    H = ndi.map_coordinates(tile, [(cv % 1.0) * th, (cu % 1.0) * tw], order=1, mode="wrap")
    z = np.linspace(0, p.length, nv + 1)
    rim = ((z < p.margin) | (z > p.length - p.margin))
    H[rim, :] = 0.0
    return np.clip(H, 0, 1)


def sleeve_mesh(H, p: RollerParams):
    nz, nth = H.shape[0] - 1, H.shape[1]
    R_out = p.diameter / 2
    base = R_out - p.depth
    R = base + p.depth * H if p.raised else R_out - p.depth * H
    Ri = base - p.wall
    if Ri < p.bore / 2 + 1:
        raise ValueError("La manica è troppo sottile: riduci lo spessore della parete o la profondità, o aumenta il diametro")
    th = np.arange(nth) * 2 * np.pi / nth
    c, s = np.cos(th), np.sin(th)
    z = np.linspace(0, p.length, nz + 1)
    outer = np.stack([R * c[None, :], R * s[None, :], np.broadcast_to(z[:, None], R.shape)], axis=2).reshape(-1, 3)
    ib = np.c_[Ri * c, Ri * s, np.zeros(nth)]
    it = np.c_[Ri * c, Ri * s, np.full(nth, p.length)]
    V = np.vstack([outer, ib, it])
    nO = len(outer)
    i = np.arange(nth)
    i1 = (i + 1) % nth
    F = []
    j = np.arange(nz)[:, None]
    a, b = j * nth + i[None, :], j * nth + i1[None, :]
    cc, dd = (j + 1) * nth + i1[None, :], (j + 1) * nth + i[None, :]
    F += [np.stack([a, b, cc], -1).reshape(-1, 3), np.stack([a, cc, dd], -1).reshape(-1, 3)]
    for o, o1, n0, n1 in ((i, i1, nO + i, nO + i1), (nz * nth + i, nz * nth + i1, nO + nth + i, nO + nth + i1)):
        F += [np.stack([o, o1, n1], -1), np.stack([o, n1, n0], -1)]          # end annulus: outer rim -> inner ring
    ibi, ibi1 = nO + i, nO + i1
    iti, iti1 = nO + nth + i, nO + nth + i1
    F += [np.stack([ibi, ibi1, iti1], -1), np.stack([ibi, iti1, iti], -1)]
    m = trimesh.Trimesh(V, np.vstack(F), process=False)
    m.merge_vertices()
    m.fix_normals()
    return m


def _cyl(r, h, z0=0.0):
    c = trimesh.creation.cylinder(radius=r, height=h, sections=64)
    c.apply_translation([0, 0, z0 + h / 2])
    return c


def core_mesh(p: RollerParams):
    base = p.diameter / 2 - p.depth
    body_r = base - p.wall - p.clearance
    L = p.length + 2 * p.clearance
    stub = 14.0
    parts = [_cyl(body_r, L, 0), _cyl(p.bore / 2, stub + 1, -stub), _cyl(p.bore / 2, stub + 1, L - 1)]
    return trimesh.boolean.union(parts, engine=md.ENGINE)


def handle_mesh(p: RollerParams):
    base = p.diameter / 2 - p.depth
    collar = _cyl(base, 4.0, 0)
    grip = _cyl(p.handle_d / 2, p.handle_len, 4.0)
    h = trimesh.boolean.union([collar, grip], engine=md.ENGINE)
    sock = _cyl(p.bore / 2 + 0.15, 14.5, -0.01)
    return trimesh.boolean.difference([h, sock], engine=md.ENGINE)


def build(tile, p: RollerParams) -> dict:
    """tile: prepared relief tile. Returns {'parts': [(name, mesh)], 'info': {...}}."""
    if p.diameter <= 2 * p.depth + 2 * p.wall + p.bore + 2:
        raise ValueError("Diametro troppo piccolo per la profondità e la parete scelte")
    p = RollerParams(**{**p.__dict__})
    while (p.repeats_u * p.detail) * (p.repeats_v * p.detail) > 360_000 and p.detail > 12:   # keep the mesh manageable
        p.detail -= 4
    H = relief_field(tile, p)
    sleeve = sleeve_mesh(H, p)
    parts = [("manica", sleeve)]
    if p.handles:
        parts.append(("anima", core_mesh(p)))
        h = handle_mesh(p)
        parts += [("maniglia 1", h), ("maniglia 2", h.copy())]
    info = dict(faces=len(sleeve.faces), detail=p.detail, sleeve_ml=abs(float(sleeve.volume)) / 1000.0,
                circumference=float(np.pi * p.diameter), cell=(float(np.pi * p.diameter / p.repeats_u), float(p.length / p.repeats_v)),
                watertight=bool(sleeve.is_watertight))
    return dict(parts=parts, info=info)


# ---------------------------------------------------------------- saved patterns
def list_patterns() -> dict:
    if not USER.is_file():
        return {}
    try:
        return json.loads(USER.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_pattern(name: str, rec: dict):
    d = list_patterns()
    d[name] = rec
    USER.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=USER.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, USER)


def delete_pattern(name: str):
    d = list_patterns()
    if d.pop(name, None) is not None:
        USER.parent.mkdir(parents=True, exist_ok=True)
        USER.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def encode_image(tile) -> str:
    return base64.b64encode(to_png(tile)).decode("ascii")


def decode_image(b64: str):
    return image_tile(base64.b64decode(b64))
