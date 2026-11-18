"""ELEGOO .goo (V3.0) reader. Layout derived from real files written by SatelLite 1.0.2.29."""
import struct
from dataclasses import dataclass, field

PREVIEW_SMALL = (116, 116)
PREVIEW_BIG = (290, 290)

# (name, struct fmt) in file order, all big-endian, after the previews. Verified against real files.
HEADER_FIELDS = [
    ("layer_count", "I"), ("res_x", "H"), ("res_y", "H"), ("mirror_x", "B"), ("mirror_y", "B"),
    ("display_w", "f"), ("display_h", "f"), ("machine_z", "f"), ("layer_height", "f"), ("exposure", "f"),
    ("delay_mode", "B"), ("light_off_delay", "f"),
    ("bottom_wait_before_cure", "f"), ("bottom_wait_after_cure", "f"), ("bottom_wait_after_lift", "f"),
    ("wait_before_cure", "f"), ("wait_after_cure", "f"), ("wait_after_lift", "f"),
    ("bottom_exposure", "f"), ("bottom_layers", "I"),
    ("bottom_lift_dist", "f"), ("bottom_lift_speed", "f"), ("lift_dist", "f"), ("lift_speed", "f"),
    ("bottom_retract_dist", "f"), ("bottom_retract_speed", "f"), ("retract_dist", "f"), ("retract_speed", "f"),
    ("bottom_lift2_dist", "f"), ("bottom_lift2_speed", "f"), ("lift2_dist", "f"), ("lift2_speed", "f"),
    ("bottom_retract2_dist", "f"), ("bottom_retract2_speed", "f"), ("retract2_dist", "f"), ("retract2_speed", "f"),
    ("bottom_pwm", "H"), ("pwm", "H"), ("per_layer_settings", "B"),
    ("print_time", "I"), ("volume", "f"), ("material_g", "f"), ("material_cost", "f"),
]


def _cstr(b):
    return b.split(b"\0", 1)[0].decode("utf-8", "replace")


def parse_header(d: bytes) -> dict:
    h = {"version": _cstr(d[0:4]), "magic": d[4:12].hex(), "software": _cstr(d[12:44]),
         "software_version": _cstr(d[44:68]), "created": _cstr(d[68:92]), "machine": _cstr(d[92:124]),
         "machine_type": _cstr(d[124:156]), "profile": _cstr(d[156:188])}
    h["aa"], h["grey"], h["blur"] = struct.unpack_from(">HHH", d, 188)
    o = 194 + PREVIEW_SMALL[0] * PREVIEW_SMALL[1] * 2
    h["delim1"] = d[o:o + 2].hex()
    o += 2 + PREVIEW_BIG[0] * PREVIEW_BIG[1] * 2
    h["delim2"] = d[o:o + 2].hex()
    o += 2
    h["_fields_at"] = o
    for name, fmt in HEADER_FIELDS:
        (h[name],) = struct.unpack_from(">" + fmt, d, o)
        o += struct.calcsize(">" + fmt)
    h["_fields_end"] = o
    return h


if __name__ == "__main__":
    import sys, pprint
    with open(sys.argv[1], "rb") as f:
        hh = parse_header(f.read(400000))
    pprint.pprint(hh, sort_dicts=False)


# ---------------------------------------------------------------------------------------------
# Layer image RLE (binary images). Token = 1 header byte + 0..3 length bytes:
#   bits 7-6 type (0 empty, 3 full; 1 / 2 only occur in grey-level files and are not implemented)
#   bits 5-4 number of extra length bytes (big-endian), bits 3-0 low nibble of the run length
#   run length = (extra << 4) | nibble. Block = 0x55 + tokens + checksum, checksum = ~sum(tokens).
# ---------------------------------------------------------------------------------------------
import numpy as np

T_EMPTY, T_FULL = 0, 3


def encode_runs(mask: np.ndarray) -> bytes:
    """mask: (H, W) uint8, >=128 counts as exposed. Returns the token stream (no 0x55 / checksum)."""
    flat = (mask.reshape(-1) >= 128)
    n = flat.size
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change))
    lens = np.diff(np.concatenate((starts, [n])))
    vals = flat[starts]
    out = bytearray()
    for L, v in zip(lens.tolist(), vals.tolist()):
        t = (T_FULL if v else T_EMPTY) << 6
        hi = L >> 4
        if hi == 0:
            out.append(t | (L & 0xF))
        elif hi < 0x100:
            out += bytes((t | 0x10 | (L & 0xF), hi))
        elif hi < 0x10000:
            out += bytes((t | 0x20 | (L & 0xF), hi >> 8, hi & 0xFF))
        else:
            out += bytes((t | 0x30 | (L & 0xF), hi >> 16, (hi >> 8) & 0xFF, hi & 0xFF))
    return bytes(out)


def encode_intervals(starts: np.ndarray, ends: np.ndarray, n_pixels: int) -> bytes:
    """Token stream for an image given by its filled spans (no image needed). Identical output to
    encode_runs() on the equivalent mask."""
    if len(starts) == 0:
        runs, types = np.array([n_pixels], np.int64), np.array([T_EMPTY], np.int64)
    else:
        full = ends - starts
        gaps = np.empty(len(starts) + 1, np.int64)
        gaps[0] = starts[0]
        gaps[1:-1] = starts[1:] - ends[:-1]
        gaps[-1] = n_pixels - ends[-1]
        runs = np.empty(2 * len(starts) + 1, np.int64)
        runs[0::2], runs[1::2] = gaps, full
        types = np.empty_like(runs)
        types[0::2], types[1::2] = T_EMPTY, T_FULL
        nz = runs > 0
        runs, types = runs[nz], types[nz]
    hi = runs >> 4
    nb = (hi > 0).astype(np.int64) + (hi > 0xFF) + (hi > 0xFFFF)
    size = 1 + nb
    off = np.cumsum(size) - size
    out = np.empty(int(size.sum()), np.uint8)
    out[off] = (types << 6) | (nb << 4) | (runs & 0xF)
    for j in (1, 2, 3):
        sel = nb >= j
        out[off[sel] + j] = (hi[sel] >> (8 * (nb[sel] - j))) & 0xFF
    return out.tobytes()


def decode_runs(tokens: bytes, n_pixels: int) -> np.ndarray:
    out = np.zeros(n_pixels, np.uint8)
    i, pos, n = 0, 0, len(tokens)
    while i < n:
        b = tokens[i]; i += 1
        t, lb, v = b >> 6, (b >> 4) & 3, b & 0xF
        x = 0
        for e in tokens[i:i + lb]:
            x = (x << 8) | e
        i += lb
        L = (x << 4) | v
        if t == T_FULL:
            out[pos:pos + L] = 255
        elif t != T_EMPTY:
            raise ValueError(f"unsupported RLE token type {t}")
        pos += L
    return out


def wrap_layer_data(tokens: bytes) -> bytes:
    return b"\x55" + tokens + bytes(((~sum(tokens)) & 0xFF,))


# ---------------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------------
from dataclasses import dataclass
from datetime import datetime

MAGIC = bytes.fromhex("07000000444c5000")
FOOTER = b"\x00\x00\x00" + MAGIC
LAYER_DEF_FMT = ">H" + "f" * 15 + "H"          # 64 bytes
LAYER_DEF_FIELDS = ["pause", "pause_pos", "z", "exposure", "light_off", "wait_before", "wait_after", "wait_after_lift",
                    "lift_dist", "lift_speed", "lift2_dist", "lift2_speed",
                    "retract_dist", "retract_speed", "retract2_dist", "retract2_speed", "pwm"]


def _txt(s: str, n: int) -> bytes:
    return s.encode("utf-8")[: n - 1].ljust(n, b"\0")


def rgb565_be(img, size) -> bytes:
    """PIL image -> big-endian RGB565, letterboxed on black to `size`."""
    from PIL import Image
    img = img.convert("RGB")
    img.thumbnail(size)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    a = np.asarray(canvas, dtype=np.uint16)
    v = ((a[..., 0] >> 3) << 11) | ((a[..., 1] >> 2) << 5) | (a[..., 2] >> 3)
    return v.astype(">u2").tobytes()


def build_header(m: dict, small: bytes, big: bytes) -> bytes:
    """m: text fields + the numeric fields in HEADER_FIELDS (missing numerics default to 0)."""
    out = bytearray()
    out += _txt(m.get("version", "V3.0"), 4) if False else b"V3.0"
    out += MAGIC
    out += _txt(m.get("software", "ELEGOO SatelLite"), 32)
    out += _txt(m.get("software_version", "1.0.2.29"), 24)
    out += _txt(m.get("created", datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 24)
    out += _txt(m.get("machine", ""), 32)
    out += _txt(m.get("machine_type", m.get("machine", "")), 32)
    out += _txt(m.get("profile", ""), 32)
    out += struct.pack(">HHH", m.get("aa", 0), m.get("grey", 0), m.get("blur", 0))
    assert len(small) == PREVIEW_SMALL[0] * PREVIEW_SMALL[1] * 2 and len(big) == PREVIEW_BIG[0] * PREVIEW_BIG[1] * 2
    out += small + b"\r\n" + big + b"\r\n"
    for name, fmt in HEADER_FIELDS:
        out += struct.pack(">" + fmt, m.get(name, 0))
    out += m.get("currency", b"$\0\0\0\0\0\0\0")
    out += struct.pack(">I", len(out) + 7)                   # layer-def address = end of header
    out += struct.pack(">BH", m.get("gray_level", 0), m.get("transition_layers", 0))
    assert struct.unpack_from(">I", out, len(out) - 7)[0] == len(out)
    return bytes(out)


def pack_layer_def(ld: dict) -> bytes:
    return struct.pack(LAYER_DEF_FMT, *(ld.get(k, 0) for k in LAYER_DEF_FIELDS))


@dataclass
class GooLayer:
    ld: dict
    mask: "np.ndarray | None" = None      # (H, W) uint8
    tokens: bytes | None = None           # already-encoded stream (skips encoding)
    on_pixels: int = 0


def write_goo(path, meta: dict, small: bytes, big: bytes, layers, progress=None, cancelled=lambda: False) -> bool:
    """layers: iterable of GooLayer in order. meta['layer_count'] must be set. Patches stats at the end."""
    hdr = build_header(meta, small, big)
    offs = {}
    o = 194 + len(small) + 2 + len(big) + 2
    for name, fmt in HEADER_FIELDS:
        offs[name] = o
        o += struct.calcsize(">" + fmt)
    total_on = 0
    n = 0
    with open(path, "wb") as f:
        f.write(hdr)
        for L in layers:
            if cancelled():
                return False
            tok = L.tokens if L.tokens is not None else encode_runs(L.mask)
            total_on += int(np.count_nonzero(L.mask >= 128)) if L.mask is not None else L.on_pixels
            data = wrap_layer_data(tok)
            f.write(pack_layer_def(L.ld) + b"\r\n" + struct.pack(">I", len(data)) + data + b"\r\n")
            n += 1
            if progress:
                progress(n, meta["layer_count"])
        f.write(FOOTER)
        if n != meta["layer_count"]:
            raise ValueError(f"declared {meta['layer_count']} layers, wrote {n}")
        if "volume" not in meta or meta.get("_patch_stats"):
            px_mm2 = (meta["display_w"] / meta["res_x"]) * (meta["display_h"] / meta["res_y"])
            vol = total_on * px_mm2 * meta["layer_height"]
            f.seek(offs["volume"]); f.write(struct.pack(">f", vol))
            grams = vol / 1000.0 * meta.get("density", 1.1)
            f.seek(offs["material_g"]); f.write(struct.pack(">f", grams))
            f.seek(offs["material_cost"]); f.write(struct.pack(">f", vol / 1e6 * meta.get("cost_per_l", 0.0)))
    return True
