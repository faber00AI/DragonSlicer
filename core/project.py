"""Headless project: the same scene -> supports -> .goo pipeline as the UI, driven by files (for the CLI and the MCP server).

A project is a folder: project.json (machine, print settings, supports, object transforms) + meshes/*.ply
(imported, already repaired / simplified / scaled). Every function returns a JSON-serialisable dict.
"""
import json
import re
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh

from . import modeling as md
from . import resins
from .machine import PrintSettings, load_machines, load_support_presets
from .scene import SceneObject
from .supports import SupportParams, build_supports, build_supports_each

GAP = 3.0                      # mm between parts when arranging
SMALL_MM = 5.0                 # smaller than this on import: the file is probably in cm / inches / metres
MODEL_EXTS = {".stl", ".obj", ".3mf", ".ply", ".off"}
TRANSFORM = ("x", "y", "rx", "ry", "rz", "sx", "sy", "sz")


class ProjectError(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------------------------
def _file(d) -> Path:
    return Path(d) / "project.json"


def _load(d) -> dict:
    f = _file(d)
    if not f.exists():
        raise ProjectError(f"{d} is not a project (run `new` first)")
    return json.loads(f.read_text(encoding="utf-8"))


def _save(d, p: dict):
    _file(d).write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")


def machine_by_name(name: str):
    ms = load_machines()
    key = _norm(name)
    for m in ms:
        if _norm(m.name) == key:
            return m
    close = [m.name for m in ms if key in _norm(m.name)]
    raise ProjectError(f"unknown machine {name!r}" + (f"; did you mean {close}?" if close else "; see `machines`"))


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("elegoo", ""))


def _machine(p):
    return machine_by_name(p["machine"])


def _objects(d, p) -> list[SceneObject]:
    out = []
    for o in p["objects"]:
        mesh = trimesh.load(Path(d) / o["mesh"], force="mesh", process=False)
        lift = o["lift"] if "lift" in o else (p["supports"]["lift"] if p["supports"]["enabled"] else 0.0)   # "lift": saved by the UI
        out.append(SceneObject(o["name"], mesh, **{k: o[k] for k in TRANSFORM}, lift=lift))
    return out


def _print_settings(p) -> PrintSettings:
    ps = PrintSettings.from_machine(_machine(p))
    for k, v in p["print"].items():
        setattr(ps, k, v)
    return ps


def _support_params(p) -> SupportParams:
    s = p["supports"]
    d = dict(load_support_presets()[s["preset"]])
    d.update(s["overrides"])
    return SupportParams.from_dict(d)


def _describe(d, p) -> dict:
    m = _machine(p)
    objs = _objects(d, p)
    items = []
    for i, (o, rec) in enumerate(zip(objs, p["objects"])):
        b = o.bounds()
        items.append(dict(index=i, name=o.name, faces=len(o.mesh.faces), watertight=bool(o.mesh.is_watertight),
                          **{k: round(rec[k], 4) for k in TRANSFORM},
                          size_mm=[round(float(v), 3) for v in b[1] - b[0]],
                          min_mm=[round(float(v), 3) for v in b[0]], max_mm=[round(float(v), 3) for v in b[1]]))
    return dict(project=str(Path(d).resolve()), machine=m.name, bed_mm=[m.bed_x, m.bed_y, m.bed_z],
                print=_print_settings(p).as_dict(), supports=p["supports"], objects=items)


# ---------------------------------------------------------------------------------------------
# project, objects
# ---------------------------------------------------------------------------------------------
def machines() -> dict:
    return dict(machines=[dict(name=m.name, bed_mm=[m.bed_x, m.bed_y, m.bed_z], resolution=[m.res_x, m.res_y])
                          for m in load_machines()], support_presets=list(load_support_presets()))


def new(d, machine="ELEGOO Saturn 4 Ultra") -> dict:
    m = machine_by_name(machine)
    d = Path(d)
    (d / "meshes").mkdir(parents=True, exist_ok=True)
    p = dict(machine=m.name, print={}, objects=[],
             supports=dict(enabled=False, preset="Tree_Medium", overrides={}, lift=0.0))
    _save(d, p)
    return _describe(d, p)


def info(d) -> dict:
    return _describe(d, _load(d))


def add(d, path, scale=None, repair=True, simplify=None, name=None) -> dict:
    """Import a model. `scale` multiplies the mesh (e.g. 10 for cm); a model < 5 mm in every direction is reported, not guessed."""
    p = _load(d)
    src = Path(path)
    if src.is_dir():
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in MODEL_EXTS)
    else:
        files = [src]
    if not files:
        raise ProjectError(f"no model files in {path}")
    notes = []
    for f in files:
        mesh = trimesh.load(f, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ProjectError(f"{f.name}: no valid mesh")
        if simplify:
            n0 = len(mesh.faces)
            mesh = md.simplify(mesh, int(simplify))
            notes.append(f"{f.name}: simplified {n0} -> {len(mesh.faces)} faces")
        if scale:
            mesh.apply_scale(float(scale))
            notes.append(f"{f.name}: scaled x{scale:g}")
        elif mesh.extents.max() < SMALL_MM:
            notes.append(f"WARNING {f.name}: largest side is {mesh.extents.max():.3f} mm - probably cm/inch/m; re-add with scale=10 / 25.4 / 1000")
        if repair and not mesh.is_watertight:
            mesh, rep = md.repair(mesh)
            b, a = rep["before"], rep["after"]
            notes.append(f"{f.name}: repaired ({rep['steps']}) open edges {b['open_edges']} -> {a['open_edges']}, "
                         f"non-manifold {b['nonmanifold_edges']} -> {a['nonmanifold_edges']}")
        if not mesh.is_watertight:
            notes.append(f"WARNING {f.name}: still not watertight")
        _append(d, p, mesh, name or f.name, name or f.stem)
    _save(d, p)
    r = _describe(d, p)
    r["notes"] = notes
    return r


def _append(d, p, mesh, name, stem):
    """Store `mesh` in the project folder and list it as a new object (not saved: the caller saves p)."""
    stem = re.sub(r"[^\w.-]", "_", stem)
    n = sum(1 for o in p["objects"] if o["mesh"].startswith(f"meshes/{stem}"))
    rel = f"meshes/{stem}{'' if n == 0 else '_' + str(n)}.ply"
    mesh.export(Path(d) / rel)
    p["objects"].append(dict(name=name, mesh=rel, x=0.0, y=0.0, rx=0.0, ry=0.0, rz=0.0, sx=1.0, sy=1.0, sz=1.0))


def add_text(d, text, font=None, height=10.0, thickness=2.0, spacing=1.0, radius=0.0, inside=False, align="center",
             name=None) -> dict:
    """Add a 3D text object (see core.text.make_text)."""
    from . import text as tx
    p = _load(d)
    try:
        mesh, missing = tx.make_text(text, font, height, thickness, spacing, radius, inside, align)
    except tx.TextError as e:
        raise ProjectError(str(e))
    label = name or "Testo " + " ".join(text.split())[:24]
    _append(d, p, mesh, label, "testo")
    p["objects"][-1]["text"] = dict(text=text, font=font, height=height, thickness=thickness, spacing=spacing, radius=radius,
                                    inside=inside, align=("center", "left", "right").index(align))   # lets the GUI edit it
    _save(d, p)
    r = _describe(d, p)
    r["notes"] = [f"characters missing from the font: {''.join(missing)}"] if missing else []
    return r


def object_mesh(d, index):
    """The world mesh of object `index` (transform applied)."""
    p = _load(d)
    objs = _objects(d, p)
    if not 0 <= index < len(objs):
        raise ProjectError(f"no object with index {index} (there are {len(objs)})")
    return objs[index].world_mesh()


def remove(d, index) -> dict:
    p = _load(d)
    _index(p, index)
    del p["objects"][index]
    _save(d, p)
    return _describe(d, p)


def _index(p, i):
    if not 0 <= i < len(p["objects"]):
        raise ProjectError(f"object index {i} out of range (0..{len(p['objects']) - 1})")


def set_object(d, index, x=None, y=None, rx=None, ry=None, rz=None, scale=None, sx=None, sy=None, sz=None,
               size_x=None, size_y=None, size_z=None, lay_flat=False) -> dict:
    """Move / rotate (degrees) / scale an object. `size_*` set the final size in mm along the world axis
    (proportional unless several are given). The part always sits on the plate."""
    p = _load(d)
    _index(p, index)
    rec = p["objects"][index]
    for k, v in dict(x=x, y=y, rx=rx, ry=ry, rz=rz, sx=sx, sy=sy, sz=sz).items():
        if v is not None:
            rec[k] = float(v)
    if scale is not None:
        rec["sx"] = rec["sy"] = rec["sz"] = float(scale)
    if lay_flat:
        o = _objects(d, p)[index]
        o.lay_flat()
        o.mesh.export(Path(d) / rec["mesh"])
        rec["rx"] = rec["ry"] = rec["rz"] = 0.0
    sizes = {k: v for k, v in dict(x=size_x, y=size_y, z=size_z).items() if v is not None}
    if sizes:
        cur = SceneObject(rec["name"], _objects(d, p)[index].mesh, **{k: rec[k] for k in TRANSFORM}).bounds()
        ext = cur[1] - cur[0]
        axes = "xyz"
        if len(sizes) == 1:
            ax, v = next(iter(sizes.items()))
            k = v / ext[axes.index(ax)]
            rec["sx"] *= k; rec["sy"] *= k; rec["sz"] *= k
        else:
            # per-axis: scale the local axis that best matches each world axis (exact for 0/90 degree rotations)
            R = np.abs(SceneObject(rec["name"], _objects(d, p)[index].mesh, rx=rec["rx"], ry=rec["ry"], rz=rec["rz"]).rotation())
            for ax, v in sizes.items():
                j = int(np.argmax(R[axes.index(ax)]))
                rec["s" + axes[j]] *= v / ext[axes.index(ax)]
    _save(d, p)
    return _describe(d, p)



def hollow(d, index, wall=2.0, drain_diameter=3.0, drain_holes=True) -> dict:
    """Hollow object `index` (shell `wall` mm thick) and, with drain_holes, open a drain at the lowest point of each cavity and
    a vent at the highest. The object keeps its place; its rotation and scale are baked into the new mesh."""
    p = _load(d)
    _index(p, index)
    rec = p["objects"][index]
    world = _objects(d, p)[index].world_mesh()
    try:
        mesh, info = md.hollow(world, float(wall), float(drain_diameter), bool(drain_holes))
    except ValueError as e:
        raise ProjectError(str(e))
    b = mesh.bounds
    mesh.export(Path(d) / rec["mesh"])
    rec.update(x=float((b[0][0] + b[1][0]) / 2), y=float((b[0][1] + b[1][1]) / 2),
               rx=0.0, ry=0.0, rz=0.0, sx=1.0, sy=1.0, sz=1.0)
    p["supports"].pop("mesh", None)
    _save(d, p)
    r = _describe(d, p)
    r["hollow"] = info
    return r


def place_objects(objs, m, gap=GAP):
    """Lay every object flat side by side on the plate of machine `m` (shelf packing), centred on the bed. Sets .x / .y."""
    order = sorted(range(len(objs)), key=lambda i: -np.prod(np.ptp(objs[i].bounds()[:, :2], axis=0)))
    sizes = {i: np.ptp(objs[i].bounds()[:, :2], axis=0) for i in order}
    rows, cur, w_cur = [], [], 0.0
    for i in order:
        w = sizes[i][0]
        if cur and w_cur + gap + w > m.bed_x - 2 * gap:
            rows.append(cur)
            cur, w_cur = [], 0.0
        cur.append(i)
        w_cur += (gap if len(cur) > 1 else 0) + w
    rows.append(cur)
    heights = [max(sizes[i][1] for i in r) for r in rows]
    total = sum(heights) + gap * (len(rows) - 1)
    y = total / 2
    for r, h in zip(rows, heights):
        x = -(sum(sizes[i][0] for i in r) + gap * (len(r) - 1)) / 2
        for i in r:
            o = objs[i]
            o.x = o.y = 0.0
            o.invalidate()
            b = o.bounds()                                   # centre of the footprint, not of the mesh
            cx, cy = (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2
            o.x = float(x + sizes[i][0] / 2 - cx)
            o.y = float(y - h / 2 - cy)
            o.invalidate()
            x += sizes[i][0] + gap
        y -= h + gap


def arrange(d, gap=GAP) -> dict:
    """Lay every object flat side by side on the plate (shelf packing), centred on the bed."""
    p = _load(d)
    objs = _objects(d, p)
    place_objects(objs, _machine(p), gap)
    for o, rec in zip(objs, p["objects"]):
        rec["x"], rec["y"] = o.x, o.y
    _save(d, p)
    return _describe(d, p)


# ---------------------------------------------------------------------------------------------
# print settings, supports
# ---------------------------------------------------------------------------------------------
def set_print(d, **kv) -> dict:
    p = _load(d)
    ok = PrintSettings.__dataclass_fields__
    for k, v in kv.items():
        if v is None:
            continue
        if k not in ok:
            raise ProjectError(f"unknown print setting {k!r}; valid: {sorted(ok)}")
        p["print"][k] = type(getattr(PrintSettings(), k))(v)
    _save(d, p)
    return _describe(d, p)


def _resin_call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except resins.ResinError as e:
        raise ProjectError(str(e))


def list_resins(d, query="", source=None, limit=60) -> dict:
    """Resin profiles available for the project's printer (the user's own first), optionally filtered by name/type."""
    m = _machine(_load(d))
    q = (query or "").lower()
    rows = [r for r in resins.list_profiles(m.name)
            if (not q or q in r["name"].lower() or q in r["type"].lower()) and (not source or r["source"] == source)]
    keys = ("layer_height", "bottom_layers", "exposure", "bottom_exposure", "lift_height", "lift_speed")
    return dict(machine=m.name, total=len(rows), shown=min(limit, len(rows)),
                profiles=[dict(name=r["name"], source=r["source"], type=r["type"],
                               **{k: r["settings"].get(k) for k in keys}) for r in rows[:limit]])


def apply_resin(d, name) -> dict:
    """Load a resin profile into the project's print settings (replaces them all)."""
    p = _load(d)
    m = _machine(p)
    ps = _resin_call(lambda: resins.to_settings(m, resins.get_profile(m.name, name)))
    p["print"] = ps.as_dict()
    _save(d, p)
    return _describe(d, p)


def save_resin(d, name, overwrite=True) -> dict:
    """Save the project's current print settings as a user resin profile for this printer."""
    p = _load(d)
    _resin_call(resins.save_user_profile, p["machine"], name, _print_settings(p), overwrite)
    p["print"]["resin"] = name.strip()
    _save(d, p)
    return dict(saved=name.strip(), machine=p["machine"], file=str(resins.USER))


def delete_resin(d, name) -> dict:
    p = _load(d)
    _resin_call(resins.delete_user_profile, p["machine"], name)
    return dict(deleted=name, machine=p["machine"])


def supports(d, enable=True, preset=None, **overrides) -> dict:
    """Turn automatic supports on/off (parameters: critical_angle, spacing, tip_radius, body_radius, lift_height, raft_height, raft_hole, ...)."""
    p = _load(d)
    s = p["supports"]
    if preset:
        if preset not in load_support_presets():
            raise ProjectError(f"unknown preset {preset!r}; valid: {list(load_support_presets())}")
        s["preset"] = preset
    s["overrides"].update({k: float(v) for k, v in overrides.items() if v is not None})
    s.pop("mesh", None)                                      # supports saved by the UI: regenerated from the parameters now
    s["enabled"] = bool(enable)
    sp = _support_params(p)
    s["lift"] = sp.lift_height if enable else 0.0
    for o in p["objects"]:
        o.pop("lift", None)                                  # per-object lifts saved by the UI no longer apply
    _save(d, p)
    r = _describe(d, p)
    if enable and p["objects"]:
        try:
            sup, n = build_supports_each([o.world_mesh() for o in _objects(d, p) if o.fits(_machine(p))], sp)      # only objects inside the print area
            r["support_count"] = n
        except ProjectError:
            r["support_count"] = 0
    return r


def _world(d, p, with_supports=True):
    m = _machine(p)
    parts = [o.world_mesh() for o in _objects(d, p) if o.fits(m)]      # objects outside the print area are ignored
    if not parts:
        raise ProjectError("the project has no objects inside the print area")
    mesh = trimesh.util.concatenate(parts)
    if with_supports and p["supports"]["enabled"]:
        sup, _ = build_supports_each(parts, _support_params(p))
        if sup is not None:
            mesh = trimesh.util.concatenate([mesh, sup])
    return mesh


# ---------------------------------------------------------------------------------------------
# .dsl: the whole project in one file
# ---------------------------------------------------------------------------------------------
DSL_EXT = ".dsl"
DSL_VERSION = 2                # 2: "extra_plates" (more build plates; the top-level objects/supports are plate 1)


def mesh_files(p: dict) -> list:
    """Every mesh file a project dict refers to: objects and saved supports of plate 1, then of the extra plates."""
    out = [o["mesh"] for o in p["objects"]] + ([p["supports"]["mesh"]] if p["supports"].get("mesh") else [])
    for pl in p.get("extra_plates", []):
        out += [o["mesh"] for o in pl["objects"]] + ([pl["supports"]["mesh"]] if pl["supports"].get("mesh") else [])
    return out


def write_dsl(out, p: dict, meshes: dict) -> Path:
    """Write project dict `p` and its meshes ({"meshes/x.ply": ply bytes}) as one .dsl file (a ZIP)."""
    out = Path(out)
    if out.suffix.lower() != DSL_EXT:
        out = out.with_suffix(DSL_EXT)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("dsl.json", json.dumps(dict(format="dragonslicer-project", version=DSL_VERSION if p.get("extra_plates") else 1)))
        z.writestr("project.json", json.dumps(p, indent=2, ensure_ascii=False))
        for rel, data in meshes.items():
            z.writestr(rel, data)
    return out


def read_dsl(src) -> tuple[dict, dict]:
    """Read a .dsl file: (project dict, {"meshes/x.ply": ply bytes}); raises ProjectError if it is not a valid one."""
    src = Path(src)
    if not src.is_file():
        raise ProjectError(f"{src} not found")
    try:
        z = zipfile.ZipFile(src)
        meta = json.loads(z.read("dsl.json"))
        p = json.loads(z.read("project.json"))
    except (zipfile.BadZipFile, KeyError, ValueError):
        raise ProjectError(f"{src.name} is not a valid .dsl file")
    if meta.get("format") not in ("dragonslicer-project", "dragonslide-project") or meta.get("version", 0) > DSL_VERSION:  # old name: files saved before the rename
        raise ProjectError(f"{src.name}: unsupported .dsl format {meta}")
    meshes = {}
    for m in mesh_files(p):                                  # supports["mesh"]: supports saved by the UI
        if not re.fullmatch(r"meshes/[\w.-]+\.ply", m) or m not in z.namelist():
            raise ProjectError(f"{src.name}: bad or missing mesh {m!r}")
        meshes[m] = z.read(m)
    return p, meshes


def pack(d, out) -> dict:
    """Write the project folder as a single .dsl file (a ZIP of dsl.json + project.json + meshes/*.ply)."""
    p = _load(d)
    out = write_dsl(out, p, {rel: (Path(d) / rel).read_bytes() for rel in mesh_files(p)})
    return dict(file=str(out.resolve()), objects=len(p["objects"]), plates=1 + len(p.get("extra_plates", [])), machine=p["machine"],
                size_mb=round(out.stat().st_size / 1e6, 2))


def unpack(src, d) -> dict:
    """Restore a .dsl file - or a SatelLite .goo.3mf - into the project folder `d` (must not already hold a project)."""
    if _file(d).exists():
        raise ProjectError(f"{d} already holds a project; pick another folder")
    notes = []
    if Path(src).suffix.lower() == ".3mf":
        from .satellite import read_satellite
        p, meshes, notes = read_satellite(src)
    else:
        p, meshes = read_dsl(src)
    d = Path(d)
    (d / "meshes").mkdir(parents=True, exist_ok=True)
    for rel, data in meshes.items():
        (d / rel).write_bytes(data)
    _save(d, p)
    r = _describe(d, p)
    if notes:
        r["notes"] = notes
    return r


def export_3mf(d, out) -> dict:
    """Write the objects as a standard 3MF (no supports), placed like SatelLite does: origin at a corner of the plate, parts on Z=0."""
    from .threemf import write_3mf
    p = _load(d)
    m = _machine(p)
    parts = []
    for o in _objects(d, p):
        w = o.world_mesh().copy()
        w.apply_translation([m.bed_x / 2, m.bed_y / 2, -o.lift])
        parts.append((o.name, w))
    if not parts:
        raise ProjectError("the project has no objects")
    out = write_3mf(out, parts)
    return dict(file=str(out.resolve()), objects=len(parts), machine=m.name, size_mb=round(out.stat().st_size / 1e6, 2))


# ---------------------------------------------------------------------------------------------
# check, export
# ---------------------------------------------------------------------------------------------
def check(d) -> dict:
    """Does the job fit the printer? Reports problems and the estimate; changes nothing."""
    from .slicer import layer_count
    p = _load(d)
    m = _machine(p)
    ps = _print_settings(p)
    objs = _objects(d, p)
    problems = []
    if not objs:
        problems.append("no objects")
    for i, o in enumerate(objs):
        b = o.bounds()
        if b[0][0] < -m.bed_x / 2 or b[1][0] > m.bed_x / 2 or b[0][1] < -m.bed_y / 2 or b[1][1] > m.bed_y / 2:
            problems.append(f"#{i} {o.name}: outside the plate, ignored by supports and slicing (x {b[0][0]:.1f}..{b[1][0]:.1f}, y {b[0][1]:.1f}..{b[1][1]:.1f}); try `arrange`")
        if b[1][2] > m.bed_z:                              # bounds() already include the lift
            problems.append(f"#{i} {o.name}: too tall, ignored by supports and slicing ({b[1][2]:.1f} mm > {m.bed_z:g})")
        if not o.mesh.is_watertight:
            problems.append(f"#{i} {o.name}: mesh not watertight (the slice may have holes)")
    live = [i for i, o in enumerate(objs) if o.fits(m)]       # what is actually supported and sliced
    for ai, i in enumerate(live):
        for j in live[ai + 1:]:
            a, b = objs[i].bounds(), objs[j].bounds()
            if (a[0][:2] < b[1][:2]).all() and (b[0][:2] < a[1][:2]).all():
                problems.append(f"#{i} and #{j}: footprints overlap (bounding boxes)")
    r = dict(ok=not problems, problems=problems)
    if not live and objs:
        problems.append("no objects inside the print area")
        r["ok"] = False
    if live:
        top = max(float(objs[i].bounds()[1][2]) for i in live)      # bounds() already include the lift
        n = max(1, int(np.ceil(top / ps.layer_height - 1e-6)))
        r.update(layers=n, height_mm=round(top, 3), estimated_minutes=round(ps.estimate_seconds(n) / 60, 1))
    return r


def export(d, out, progress=None) -> dict:
    """Slice the project and write an ELEGOO .goo, then validate it."""
    from .export import export_goo
    from .goo_check import check as goo_check
    p = _load(d)
    c = check(d)
    # objects outside the plate / too tall are ignored (reported as warnings); nothing left to print is an error
    blocking = [x for x in c["problems"] if x in ("no objects", "no objects inside the print area")]
    if blocking:
        raise ProjectError("; ".join(blocking))
    out = str(out)
    mesh = _world(d, p)
    ok = export_goo(out, mesh, _machine(p), _print_settings(p), progress=progress)
    if not ok:
        raise ProjectError("export cancelled")
    problems, info = goo_check(out)
    return dict(file=str(Path(out).resolve()), valid=not problems, problems=problems[:10], layers=info["layers"],
                size_mb=round(info["size"] / 1e6, 2), warnings=[x for x in c["problems"] if x not in blocking])
