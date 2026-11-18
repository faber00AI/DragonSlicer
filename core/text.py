"""3D text: a line of text from any installed font, extruded to a thickness, optionally wrapped round a cylinder."""
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTCollection, TTFont

CURVE_STEPS = 10                 # segments per Bezier curve of a glyph outline
MAX_DETAIL = 0.4                 # mm: longest edge once the text is bent (the flat caps must follow the cylinder)
_FONTS: dict | None = None


class TextError(Exception):
    pass


def _font_dirs():
    if sys.platform == "win32":
        yield Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        yield Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
    elif sys.platform == "darwin":
        yield Path("/System/Library/Fonts")
        yield Path("/Library/Fonts")
        yield Path.home() / "Library" / "Fonts"
    else:
        yield Path("/usr/share/fonts")
        yield Path.home() / ".fonts"
        yield Path.home() / ".local" / "share" / "fonts"


def fonts() -> dict[str, tuple[str, int]]:
    """{full font name: (file, index in the file)} for the fonts installed on this computer (read once)."""
    global _FONTS
    if _FONTS is None:
        found = {}
        for d in _font_dirs():
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*")):
                ext = p.suffix.lower()
                if ext not in (".ttf", ".otf", ".ttc", ".otc"):
                    continue
                try:
                    if ext in (".ttc", ".otc"):
                        fs = TTCollection(str(p), lazy=True).fonts
                    else:
                        fs = [TTFont(str(p), lazy=True)]
                    for i, f in enumerate(fs):
                        name = f["name"].getDebugName(4)
                        if name and ("glyf" in f or "CFF " in f or "CFF2" in f):
                            found.setdefault(name, (str(p), i))
                except Exception:
                    continue
        _FONTS = dict(sorted(found.items(), key=lambda kv: kv[0].lower()))
    return _FONTS


def default_font() -> str:
    names = fonts()
    for want in ("Arial", "Arial Nova", "Segoe UI", "DejaVu Sans", "Helvetica", "Liberation Sans"):
        if want in names:
            return want
    if not names:
        raise TextError("nessun font trovato sul computer")
    return next(iter(names))


def _open(font: str) -> TTFont:
    names = fonts()
    if font in names:
        path, idx = names[font]
    elif Path(font).is_file():                      # a font file given directly
        path, idx = font, 0
    else:
        close = [n for n in names if font.lower() in n.lower()]
        if not close:
            raise TextError(f"font {font!r} non trovato (vedi l'elenco dei font)")
        path, idx = names[close[0]]
    if path.lower().endswith((".ttc", ".otc")):
        return TTCollection(path).fonts[idx]
    return TTFont(path)


class _Outline(BasePen):
    """Collects the contours of a glyph as polylines (curves flattened)."""

    def __init__(self, glyphs):
        super().__init__(glyphs)
        self.contours: list[list[tuple[float, float]]] = []
        self._cur: list = []

    def _moveTo(self, p):
        self._cur = [p]

    def _lineTo(self, p):
        self._cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._cur[-1]
        for t in np.linspace(0, 1, CURVE_STEPS + 1)[1:]:
            u = 1 - t
            self._cur.append((u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
                              u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self._cur[-1]
        for t in np.linspace(0, 1, CURVE_STEPS + 1)[1:]:
            u = 1 - t
            self._cur.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))

    def _closePath(self):
        if len(self._cur) > 2:
            self.contours.append(self._cur)
        self._cur = []

    _endPath = _closePath


def _layout(text: str, font: TTFont, spacing: float):
    """(contours in em-normalised units per line, line widths, cap height, line height, characters the font lacks)."""
    upm = font["head"].unitsPerEm
    cmap = font.getBestCmap() or {}
    glyphs = font.getGlyphSet()
    hmtx = font["hmtx"]
    os2 = font["OS/2"] if "OS/2" in font else None
    cap = (getattr(os2, "sCapHeight", 0) or 0.7 * upm) / upm
    asc, desc = font["hhea"].ascent / upm, -font["hhea"].descent / upm
    line_h = max(asc + desc + font["hhea"].lineGap / upm, 1.0)
    lines, widths, missing = [], [], set()
    for line in text.split("\n"):
        x, contours = 0.0, []
        for ch in line:
            name = cmap.get(ord(ch))
            if name is None:
                if ch not in " \t":
                    missing.add(ch)
                x += 0.3 * spacing if ch != "\t" else 1.2
                continue
            pen = _Outline(glyphs)
            glyphs[name].draw(pen)
            for c in pen.contours:
                a = np.asarray(c, float) / upm
                a[:, 0] += x
                contours.append(a)
            x += hmtx[name][0] / upm * spacing
        lines.append(contours)
        widths.append(x)
    return lines, widths, cap, line_h, missing


def make_text(text: str, font: str | None = None, height: float = 10.0, thickness: float = 2.0, spacing: float = 1.0,
              radius: float = 0.0, inside: bool = False, align: str = "center") -> tuple[trimesh.Trimesh, list[str]]:
    """Text as a watertight solid, centred on the origin: letters read along +X, the thickness is along +Z (base at Z=0).
    height = height of the capital letters (mm); thickness = extrusion (mm); spacing = multiplies the letter advance.
    radius > 0 wraps the text round a cylinder of that radius whose axis is parallel to Y (the letters stand on its
    surface: outside, or on the inside of a ring when `inside`). Returns (mesh, [characters the font does not have])."""
    import manifold3d as mf

    text = text.strip("\n")
    if not text.strip():
        raise TextError("scrivi un testo")
    if height <= 0 or thickness <= 0:
        raise TextError("altezza e spessore devono essere maggiori di zero")
    f = _open(font or default_font())
    lines, widths, cap, line_h, missing = _layout(text, f, spacing)
    if not any(lines):
        raise TextError("il font scelto non ha nessuno di questi caratteri")
    k = height / cap                                         # em -> mm
    widest = max(widths)
    polys = []
    for i, (contours, w) in enumerate(zip(lines, widths)):
        dx = {"left": 0.0, "center": (widest - w) / 2, "right": widest - w}[align]
        for c in contours:
            p = c.copy()
            p[:, 0] += dx
            p[:, 1] -= i * line_h
            polys.append(p * k)
    cs = mf.CrossSection(polys, mf.FillRule.NonZero)         # overlapping contours merge, inner ones become holes
    if cs.is_empty():
        raise TextError("il testo non ha contorni")
    solid = mf.Manifold.extrude(cs, float(thickness))
    if radius > 0:
        solid = solid.refine_to_length(MAX_DETAIL)
    m = solid.to_mesh()
    v = np.asarray(m.vert_properties, float)[:, :3]
    t = np.asarray(m.tri_verts, np.int64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    v[:, 0] -= (lo[0] + hi[0]) / 2
    v[:, 1] -= (lo[1] + hi[1]) / 2
    if radius > 0:
        if (hi[0] - lo[0]) / radius > 1.8 * np.pi:
            raise TextError(f"il testo è troppo largo per un raggio di {radius:g} mm: aumenta il raggio o riduci il testo")
        if inside and radius <= thickness:
            raise TextError("con le lettere all'interno il raggio deve essere maggiore dello spessore")
        r = radius - v[:, 2] if inside else radius + v[:, 2]
        th = v[:, 0] / radius
        v = np.column_stack([r * np.sin(th), v[:, 1], r * np.cos(th)])
    out = trimesh.Trimesh(v, t, process=True)
    if out.volume < 0:
        out.invert()
    out.apply_translation([0, 0, -out.bounds[0][2]])
    return out, sorted(missing)
