"""Silicone mold box: a base plate with a groove plus an open-top wall frame that sits in the groove.

The master stands on the base (glued down), the frame is dropped into the groove, silicone is poured from the open top.
After curing the frame is lifted off and the silicone block is cut open to take the master out.
All sizes in mm; every returned mesh is centred on x/y = 0 and rests on z = 0 (the caller places it)."""
from dataclasses import dataclass

import numpy as np
import trimesh

from .modeling import ENGINE


@dataclass
class MoldParams:
    silicone: float = 12.0      # silicone thickness around the master (sides)
    top: float = 12.0           # silicone above the highest point of the master
    wall: float = 3.0           # thickness of the printed frame walls
    clearance: float = 0.2      # play between frame and groove (per side)
    base: float = 5.0           # base plate thickness
    groove: float = 3.0         # depth of the groove (and how far the frame sinks into the base)
    flange: float = 6.0         # base plate margin outside the frame


def _box(sx, sy, z0, z1):
    b = trimesh.creation.box((sx, sy, z1 - z0))
    b.apply_translation([0, 0, (z0 + z1) / 2])
    return b


def _diff(a, *rest):
    return trimesh.boolean.difference([a, *rest], engine=ENGINE)


def build(master: trimesh.Trimesh, p: MoldParams) -> dict:
    """Return {'base': mesh, 'frame': mesh, 'info': {...}} for `master` (a mesh in world coordinates)."""
    ext = np.asarray(master.bounds[1] - master.bounds[0], float)
    if not np.all(ext > 0):
        raise ValueError("Il modello non ha volume")
    if p.groove >= p.base:
        raise ValueError("La scanalatura deve essere meno profonda dello spessore della base")
    ix, iy = ext[0] + 2 * p.silicone, ext[1] + 2 * p.silicone          # cavity (inner) size
    ox, oy = ix + 2 * p.wall, iy + 2 * p.wall                          # frame outer size
    cav_h = ext[2] + p.top                                              # silicone column height above the plate
    frame_h = cav_h + p.groove

    frame = _diff(_box(ox, oy, 0, frame_h), _box(ix, iy, -1, frame_h + 1))

    c = p.clearance
    plate = _box(ox + 2 * p.flange, oy + 2 * p.flange, 0, p.base)
    groove = _diff(_box(ox + 2 * c, oy + 2 * c, p.base - p.groove, p.base + 1),
                   _box(ix - 2 * c, iy - 2 * c, p.base - p.groove - 1, p.base + 2))
    base = _diff(plate, groove)

    vol = abs(float(master.volume)) if master.is_watertight else 0.0
    silicone_ml = (ix * iy * cav_h - vol) / 1000.0
    info = dict(cavity=(ix, iy, cav_h), frame=(ox, oy, frame_h), base=(ox + 2 * p.flange, oy + 2 * p.flange, p.base),
                master=tuple(ext), silicone_ml=silicone_ml, master_ml=vol / 1000.0,
                master_watertight=bool(master.is_watertight))
    return dict(base=base, frame=frame, info=info)
