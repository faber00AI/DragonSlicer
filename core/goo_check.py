"""Independent structural validator for ELEGOO .goo files (does not use the writer)."""
import struct
from pathlib import Path

import numpy as np

from .goo import FOOTER, HEADER_FIELDS, LAYER_DEF_FIELDS, LAYER_DEF_FMT, MAGIC, PREVIEW_BIG, PREVIEW_SMALL, parse_header

LD_SIZE = struct.calcsize(LAYER_DEF_FMT)


def _runs_total(tokens: bytes):
    """Pixels covered by an RLE token stream, or raise ValueError on a malformed stream."""
    i, n, tot, full = 0, len(tokens), 0, 0
    while i < n:
        b = tokens[i]
        i += 1
        t, lb, v = b >> 6, (b >> 4) & 3, b & 0xF
        if t not in (0, 3):
            raise ValueError(f"token type {t}")
        if i + lb > n:
            raise ValueError("token cut")
        x = int.from_bytes(tokens[i:i + lb], "big") if lb else 0
        i += lb
        L = (x << 4) | v
        tot += L
        if t == 3:
            full += L
    return tot, full


def check(path, max_layers=None, quiet=False):
    """Returns (problems, info). problems == [] means the file is structurally valid."""
    problems = []
    p = Path(path)
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(400000)
        h = parse_header(head)
        if h["version"] != "V3.0":
            problems.append(f"version {h['version']!r}")
        if bytes.fromhex(h["magic"]) != MAGIC:
            problems.append(f"magic {h['magic']}")
        if h["delim1"] != "0d0a" or h["delim2"] != "0d0a":
            problems.append(f"preview delimiters {h['delim1']} {h['delim2']}")
        n_px = h["res_x"] * h["res_y"]
        end = h["_fields_end"]
        # after the numeric fields: currency(8) + layer-def address(4) + gray level(1) + transition layers(2)
        addr, = struct.unpack_from(">I", head, end + 8)
        if addr != end + 8 + 4 + 3:
            problems.append(f"layer-def address {addr} != {end + 15}")
        f.seek(addr)
        prev_z = 0.0
        count, bad_cs, bad_len, bad_def = 0, 0, 0, 0
        zs, exps, on = [], [], 0
        first_def = None
        total_layers = h["layer_count"]
        while count < total_layers:
            raw = f.read(LD_SIZE + 2 + 4)
            if len(raw) < LD_SIZE + 6:
                problems.append(f"file ends in layer {count}")
                break
            ld = dict(zip(LAYER_DEF_FIELDS, struct.unpack(LAYER_DEF_FMT, raw[:LD_SIZE])))
            if first_def is None:
                first_def = ld
            if raw[LD_SIZE:LD_SIZE + 2] != b"\r\n":
                bad_def += 1
            (dlen,) = struct.unpack(">I", raw[LD_SIZE + 2:])
            data = f.read(dlen + 2)
            if len(data) < dlen + 2 or data[dlen:] != b"\r\n":
                problems.append(f"layer {count}: data block / CRLF broken")
                break
            if data[0] != 0x55:
                bad_len += 1
            tok, cs = data[1:dlen - 1], data[dlen - 1]
            if (~sum(tok)) & 0xFF != cs:
                bad_cs += 1
            try:
                tot, full = _runs_total(tok)
                if tot != n_px:
                    problems.append(f"layer {count}: runs cover {tot} px, expected {n_px}")
                on += full
            except ValueError as e:
                problems.append(f"layer {count}: {e}")
            if ld["z"] + 1e-4 < prev_z:
                problems.append(f"layer {count}: z goes back ({ld['z']} < {prev_z})")
            prev_z = ld["z"]
            zs.append(ld["z"])
            exps.append(ld["exposure"])
            count += 1
            if max_layers and count >= max_layers:
                break
        tail = f.read()
        if not max_layers:
            if tail != FOOTER:
                problems.append(f"footer is {tail[:16].hex()}… ({len(tail)} bytes), expected {FOOTER.hex()}")
        if bad_cs:
            problems.append(f"{bad_cs} layers with a wrong checksum")
        if bad_len:
            problems.append(f"{bad_len} layers not starting with 0x55")
        if bad_def:
            problems.append(f"{bad_def} layer definitions without CRLF")
    info = dict(size=size, layers=count, declared=h["layer_count"], res=(h["res_x"], h["res_y"]),
                layer_height=h["layer_height"], z_last=zs[-1] if zs else None, on_pixels=on,
                header={k: h[k] for k, _ in HEADER_FIELDS}, text={k: h[k] for k in ("software", "software_version",
                        "machine", "machine_type", "profile", "created")}, first_layer_def=first_def)
    if zs:
        expect = h["layer_count"] * h["layer_height"]
        if abs(zs[-1] - expect) > h["layer_height"] * 1.01:
            problems.append(f"last z {zs[-1]:.3f} vs {expect:.3f}")
    return problems, info
