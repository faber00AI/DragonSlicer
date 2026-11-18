"""Write a sliced job as a ZIP of PNG masks + job.json (printer-native format comes later)."""
import io
import json
import zipfile
from PIL import Image

from .machine import Machine, PrintSettings
from .slicer import slice_layers, slice_intervals, layer_count

PLACEHOLDER_MOTION = 0.05      # what ChituBox writes in every lift/retract field of a tilt-release .goo (0 = unset)


def export_zip(path, mesh, machine: Machine, ps: PrintSettings, progress=None, cancelled=lambda: False):
    n = layer_count(mesh, ps.layer_height)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for i, mask in slice_layers(mesh, machine, ps.layer_height):
            if cancelled():
                return False
            buf = io.BytesIO()
            Image.fromarray(mask, "L").save(buf, "PNG", compress_level=1)
            z.writestr(f"layers/{i + 1:05d}.png", buf.getvalue())
            if progress:
                progress(i + 1, n)
        job = {
            "machine": machine.name, "resolution": [machine.res_x, machine.res_y],
            "bed_mm": [machine.bed_x, machine.bed_y, machine.bed_z],
            "layers": n, "height_mm": float(mesh.bounds[1][2]), **ps.as_dict(),
            "estimated_seconds": ps.estimate_seconds(n),
        }
        z.writestr("job.json", json.dumps(job, indent=2))
    return True


def mesh_thumbnail(mesh, size, max_faces=40000):
    """Shaded three-quarter view of the part, used as the file thumbnail when the UI gives none."""
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (24, 24, 28))
    if mesh is None or len(mesh.faces) == 0:
        return img
    v = np.asarray(mesh.vertices, float) - mesh.bounds.mean(axis=0)
    a, b = np.radians(35), np.radians(-30)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    v = v @ (rx @ rz).T                               # x right, z up, y depth
    f = np.asarray(mesh.faces)
    if len(f) > max_faces:
        f = f[np.random.default_rng(0).choice(len(f), max_faces, replace=False)]
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.4, -0.7, 0.6]); light /= np.linalg.norm(light)
    shade = (0.35 + 0.65 * np.abs(n @ light) * 1.0).clip(0, 1)
    span = max(np.ptp(v[:, 0]), np.ptp(v[:, 2]), 1e-9)
    sc = 0.8 * min(size) / span
    cx, cy = size[0] / 2, size[1] / 2
    d = ImageDraw.Draw(img)
    for i in np.argsort(-tri[:, :, 1].mean(axis=1)):
        pts = [(cx + p[0] * sc, cy - p[2] * sc) for p in tri[i]]
        g = int(60 + 190 * shade[i])
        d.polygon(pts, fill=(g, int(g * 0.9), int(g * 0.8)))
    return img


def _motion(ps: PrintSettings, bottom: bool, prefix="") -> dict:
    """Lift/retract in the .goo field names. The retract is the mirror of the lift: the last `retract2_height` mm slow."""
    g = (lambda n: getattr(ps, ("bottom_" if bottom else "") + n))
    total = g("lift_height") + g("lift2_height")
    r2 = min(g("retract2_height"), total)
    return {prefix + "lift_dist": g("lift_height"), prefix + "lift_speed": g("lift_speed"),
            prefix + "lift2_dist": g("lift2_height"), prefix + "lift2_speed": g("lift2_speed"),
            prefix + "retract_dist": total - r2, prefix + "retract_speed": g("retract_speed"),
            prefix + "retract2_dist": r2, prefix + "retract2_speed": g("retract2_speed")}


def export_goo(path, mesh, machine: Machine, ps: PrintSettings, preview=None, progress=None,
               cancelled=lambda: False, software="DragonSlicer"):
    """Write an ELEGOO .goo (V3.0) job. Exposure, waits and LED power come from `ps`. Lift/retract are written only with
    ps.embed_motion; otherwise they stay 0 like SatelLite does and the printer's own machine profile drives motion."""
    from PIL import Image
    from .goo import GooLayer, rgb565_be, write_goo, PREVIEW_SMALL, PREVIEW_BIG

    n = layer_count(mesh, ps.layer_height)
    img = preview if preview is not None else mesh_thumbnail(mesh, PREVIEW_BIG)
    meta = dict(
        software=software, software_version="0.1", machine=machine.name, machine_type=machine.name,
        profile=ps.resin, layer_count=n, res_x=machine.res_x, res_y=machine.res_y,
        mirror_x=machine.mirror_y, mirror_y=machine.mirror_x,     # as ChituBox: the profile's flags swapped (Saturn 4 Ultra: 0,1 -> 1,0)
        display_w=machine.bed_x, display_h=machine.bed_y, machine_z=machine.bed_z,
        layer_height=ps.layer_height, exposure=ps.exposure, delay_mode=1, light_off_delay=ps.light_off_delay,
        bottom_wait_before_cure=ps.bottom_wait_before_cure, bottom_wait_after_cure=ps.bottom_wait_after_cure,
        bottom_wait_after_lift=ps.bottom_wait_after_lift, wait_before_cure=ps.wait_before_cure,
        wait_after_cure=ps.wait_after_cure, wait_after_lift=ps.wait_after_lift,
        bottom_exposure=ps.bottom_exposure, bottom_layers=ps.bottom_layers,
        bottom_pwm=ps.bottom_pwm, pwm=ps.pwm, print_time=int(ps.estimate_seconds(n)),
        transition_layers=ps.transition_layers, aa=4, grey=3, blur=2,      # ChituBox's values (layer images stay 1-bit)
        density=ps.density, cost_per_l=ps.cost, _patch_stats=True,
    )
    if ps.embed_motion:
        meta.update(_motion(ps, True, "bottom_"))
        meta.update(_motion(ps, False))
    else:                                       # like ChituBox: a tiny placeholder, the printer's own profile drives motion
        meta.update({p + k: PLACEHOLDER_MOTION for p in ("bottom_", "")
                     for k in ("lift_dist", "lift_speed", "retract_dist", "retract_speed")})

    from .goo import encode_intervals
    npx = machine.res_x * machine.res_y

    def layer_def(i):
        b = i < ps.bottom_layers
        g = (lambda n: getattr(ps, ("bottom_" if b else "") + n))
        ld = dict(pause=0, pause_pos=(i + 1) * ps.layer_height, z=(i + 1) * ps.layer_height,   # ChituBox: pause_pos = layer z
                  exposure=ps.layer_exposure(i), light_off=g("light_off_delay"),
                  wait_before=g("wait_before_cure"), wait_after=g("wait_after_cure"), wait_after_lift=g("wait_after_lift"),
                  pwm=ps.bottom_pwm if b else ps.pwm)
        if ps.embed_motion:
            ld.update({k: v for k, v in _motion(ps, b).items()})
        else:
            ld.update(lift_dist=PLACEHOLDER_MOTION, lift_speed=PLACEHOLDER_MOTION,
                      retract_dist=PLACEHOLDER_MOTION, retract_speed=PLACEHOLDER_MOTION)
        return ld

    def layers():
        # no image is ever built: RLE tokens come straight from the scanline spans
        for i, st, en in slice_intervals(mesh, machine, ps.layer_height, range(n),
                                         flip_x=bool(machine.mirror_x), flip_y=bool(machine.mirror_y)):
            yield GooLayer(layer_def(i), tokens=encode_intervals(st, en, npx), on_pixels=int((en - st).sum()))

    return write_goo(path, meta, rgb565_be(img, PREVIEW_SMALL), rgb565_be(img, PREVIEW_BIG), layers(),
                     progress=progress, cancelled=cancelled)
