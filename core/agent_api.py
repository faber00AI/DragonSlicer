"""The tool surface for agents: one list of typed functions, exposed both as CLI sub-commands (dragonslicer_cli.py)
and as MCP tools (dragonslicer_mcp.py). Every function returns a JSON-serialisable dict and raises ProjectError on a
problem the caller can fix. Parameters named `confirm` guard everything that moves the printer."""
import sys

from . import library as lib
from . import printer_ops as po
from . import project as pj
from .goo_check import check as _goo_check
from .project import ProjectError  # noqa: F401  (re-exported)

DEFAULT_PROJECT = "dragonslicer_project"


def _progress(i, n):
    if n and (i == n or i % max(1, n // 20) == 0):
        print(f"  {i}/{n}", file=sys.stderr, flush=True)


def list_machines() -> dict:
    """List printer profiles (name, plate size, resolution) and the support presets."""
    return pj.machines()


def new_project(project: str = DEFAULT_PROJECT, machine: str = "ELEGOO Saturn 4 Ultra") -> dict:
    """Create an empty project folder for the given printer profile."""
    return pj.new(project, machine)


def project_info(project: str = DEFAULT_PROJECT) -> dict:
    """Show the machine, print settings, supports and every object with its exact size/position in mm."""
    return pj.info(project)


def add_model(path: str, project: str = DEFAULT_PROJECT, scale: float | None = None, repair: bool = True,
              simplify: int | None = None, name: str | None = None) -> dict:
    """Import a model (STL/OBJ/3MF/PLY/OFF, or a folder of them). Repairs open meshes. `scale` multiplies it
    (10 = cm to mm, 25.4 = inch to mm); `simplify` reduces the triangle count to that target."""
    return pj.add(project, path, scale=scale, repair=repair, simplify=simplify, name=name)


def list_fonts() -> dict:
    """List the fonts installed on this computer that add_text can use (full names, e.g. 'Arial Bold')."""
    from . import text as tx
    return dict(default=tx.default_font(), fonts=list(tx.fonts()))


def add_text(text: str, project: str = DEFAULT_PROJECT, font: str | None = None, height: float = 10.0,
             thickness: float = 2.0, spacing: float = 1.0, radius: float = 0.0, inside: bool = False,
             align: str = "center", name: str | None = None) -> dict:
    """Add a 3D text (a backslash-n in the text = new line). `height` = capital letter height in mm, `thickness` = extrusion in mm,
    `font` = name from list_fonts (default Arial), `spacing` multiplies the letter spacing. radius > 0 wraps the text round a
    cylinder of that radius (axis along Y, letters on the outside, or on the inside of a ring with inside=true); 0 = flat.
    align: left/center/right."""
    return pj.add_text(project, text.replace(chr(92) + "n", chr(10)), font, height, thickness, spacing, radius, inside, align, name)


def remove_object(index: int, project: str = DEFAULT_PROJECT) -> dict:
    """Remove the object with this index (see project_info)."""
    return pj.remove(project, index)


def set_object(index: int, project: str = DEFAULT_PROJECT, x: float | None = None, y: float | None = None,
               rx: float | None = None, ry: float | None = None, rz: float | None = None,
               scale: float | None = None, size_x: float | None = None, size_y: float | None = None,
               size_z: float | None = None, lay_flat: bool = False) -> dict:
    """Place an object: x/y centre in mm (plate centre = 0,0), rx/ry/rz rotation in degrees, uniform `scale`, or the
    final size in mm along X/Y/Z (one size = proportional). lay_flat rests the largest flat face on the plate."""
    return pj.set_object(project, index, x=x, y=y, rx=rx, ry=ry, rz=rz, scale=scale, size_x=size_x, size_y=size_y,
                         size_z=size_z, lay_flat=lay_flat)


def hollow_object(index: int, project: str = DEFAULT_PROJECT, wall: float = 2.0, drain_diameter: float = 3.0,
                  drain_holes: bool = True) -> dict:
    """Hollow an object to a shell of `wall` mm (saves resin, less suction). With drain_holes a drain is opened at the lowest
    point of each cavity and a vent at the highest, so uncured resin can run out. Do it before set_supports."""
    return pj.hollow(project, index, wall, drain_diameter, drain_holes)


def arrange(project: str = DEFAULT_PROJECT, gap: float = 3.0) -> dict:
    """Pack all objects side by side on the plate, centred, with `gap` mm between them."""
    return pj.arrange(project, gap)


def set_print(project: str = DEFAULT_PROJECT, layer_height: float | None = None, bottom_layers: int | None = None,
              transition_layers: int | None = None, exposure: float | None = None, bottom_exposure: float | None = None,
              light_off_delay: float | None = None, bottom_light_off_delay: float | None = None,
              wait_before_cure: float | None = None, bottom_wait_before_cure: float | None = None,
              wait_after_cure: float | None = None, bottom_wait_after_cure: float | None = None,
              wait_after_lift: float | None = None, bottom_wait_after_lift: float | None = None,
              lift_height: float | None = None, lift_speed: float | None = None,
              bottom_lift_height: float | None = None, bottom_lift_speed: float | None = None,
              lift2_height: float | None = None, lift2_speed: float | None = None,
              bottom_lift2_height: float | None = None, bottom_lift2_speed: float | None = None,
              retract_speed: float | None = None, bottom_retract_speed: float | None = None,
              retract2_height: float | None = None, retract2_speed: float | None = None,
              bottom_retract2_height: float | None = None, bottom_retract2_speed: float | None = None,
              pwm: int | None = None, bottom_pwm: int | None = None, resin: str | None = None,
              resin_type: str | None = None, density: float | None = None, cost: float | None = None,
              embed_motion: bool | None = None) -> dict:
    """Change print settings (seconds, mm, mm/min). Unset values stay as they are. bottom_* applies to the bottom layers,
    *2 is the second lift/retract stage. embed_motion=true writes lift/retract into the .goo (default: the printer's own
    profile drives motion, as SatelLite does). To load a whole resin profile use apply_resin instead."""
    kv = {k: v for k, v in locals().items() if k != "project"}
    return pj.set_print(project, **kv)


def list_resins(project: str = DEFAULT_PROJECT, query: str = "", source: str | None = None, limit: int = 60) -> dict:
    """List the resin profiles for the project's printer: the user's own ('user') and SatelLite's library ('library').
    `query` filters by name/type (e.g. 'water', 'ABS-Like'), `source` by 'user' or 'library'."""
    return pj.list_resins(project, query, source, limit)


def apply_resin(name: str, project: str = DEFAULT_PROJECT) -> dict:
    """Load a resin profile by exact name (see list_resins) into the project: replaces all the print settings."""
    return pj.apply_resin(project, name)


def save_resin(name: str, project: str = DEFAULT_PROJECT, overwrite: bool = True) -> dict:
    """Save the project's current print settings as a user resin profile (per printer) so it can be reused later."""
    return pj.save_resin(project, name, overwrite)


def delete_resin(name: str, project: str = DEFAULT_PROJECT) -> dict:
    """Delete a user resin profile (the library ones cannot be deleted)."""
    return pj.delete_resin(project, name)


def _lib(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except lib.LibraryError as e:
        raise ProjectError(str(e))


def list_library() -> dict:
    """List the personal model library (OBJ files in the library folder of DragonSlicer)."""
    return dict(folder=str(lib.DIR), models=lib.list_models())


def save_to_library(name: str, path: str | None = None, index: int | None = None, project: str = DEFAULT_PROJECT,
                    overwrite: bool = False) -> dict:
    """Store a model in the personal library as OBJ, either from a model file (`path`) or from an object of the project
    (`index`, as placed: rotation and scale included). Nothing is repaired or rescaled."""
    if (path is None) == (index is None):
        raise ProjectError("give exactly one of path or index")
    if path is not None:
        stored = _lib(lib.add_file, path, name, overwrite)
    else:
        stored = _lib(lib.save_mesh, name, pj.object_mesh(project, index), overwrite)
    return dict(saved=stored, folder=str(lib.DIR))


def add_from_library(name: str, project: str = DEFAULT_PROJECT, repair: bool = False) -> dict:
    """Add a library model to the project (not repaired by default: the library keeps models as saved)."""
    _lib(lib.load, name)
    return pj.add(project, str(lib._path(name)), repair=repair, name=name)


def delete_from_library(name: str) -> dict:
    """Remove a model from the personal library."""
    _lib(lib.delete, name)
    return dict(deleted=name)


def set_supports(project: str = DEFAULT_PROJECT, enable: bool = True, preset: str | None = None,
                 critical_angle: float | None = None, spacing: float | None = None, tip_radius: float | None = None,
                 body_radius: float | None = None, lift_height: float | None = None,
                 raft_height: float | None = None, raft_hole: float | None = None) -> dict:
    """Enable/disable automatic pillar supports (presets Tree_Small / Tree_Medium / Tree_Large) with a raft of raft_height mm
    on the plate (0 = no raft). raft_hole > 0 perforates the raft with square holes of that size between the
    pillars (0 = solid raft). Returns the support count."""
    return pj.supports(project, enable, preset, critical_angle=critical_angle, spacing=spacing, tip_radius=tip_radius,
                       body_radius=body_radius, lift_height=lift_height, raft_height=raft_height, raft_hole=raft_hole)


def check_project(project: str = DEFAULT_PROJECT) -> dict:
    """Check that everything fits the plate and height, parts do not overlap, meshes are closed; layer count and time."""
    return pj.check(project)


def export_project(out: str, project: str = DEFAULT_PROJECT) -> dict:
    """Save the whole project (machine, settings, supports, objects and their meshes) as one .dsl file."""
    return pj.pack(project, out)


def export_3mf(out: str, project: str = DEFAULT_PROJECT) -> dict:
    """Save the objects as a standard 3MF (geometry and positions only: no supports, resin or printer settings)."""
    return pj.export_3mf(project, out)


def import_project(path: str, project: str = DEFAULT_PROJECT) -> dict:
    """Open a .dsl file, or a SatelLite project (.goo.3mf: parts and positions, printer, resin; supports are regenerated),
    into a new project folder (the folder must not already hold a project)."""
    return pj.unpack(path, project)


def export_goo(out: str, project: str = DEFAULT_PROJECT) -> dict:
    """Slice and write the .goo file, then validate it (structure, checksums, pixel coverage). Takes a while on big jobs."""
    return pj.export(project, out, progress=_progress)


def validate_goo(path: str) -> dict:
    """Validate any .goo file without the project (structure, per-layer checksums, footer)."""
    problems, info = _goo_check(path)
    return dict(valid=not problems, problems=problems[:20], layers=info["layers"], declared_layers=info["declared"],
                resolution=list(info["res"]), layer_height=round(info["layer_height"], 4), size_mb=round(info["size"] / 1e6, 2),
                text=info["text"])


def find_printers(timeout: float = 3.0) -> dict:
    """Search the local network for ELEGOO printers (UDP broadcast on port 3000)."""
    return po.discover(timeout)


def printer_status(ip: str) -> dict:
    """Printer state: idle/printing, current file and layer."""
    return po.status(ip)


def printer_files(ip: str) -> dict:
    """List the files stored on the printer."""
    return po.files(ip)


def printer_upload(ip: str, path: str, remote_name: str | None = None, project: str | None = None) -> dict:
    """Upload a .goo to the printer (does NOT start the print). With `project`, refuses a printer of a different model."""
    return po.upload(ip, path, remote_name, project, progress=_progress)


def send_project(ip: str, project: str = DEFAULT_PROJECT, name: str | None = None, out: str | None = None) -> dict:
    """Export the project to .goo, validate it and upload it to the printer. Does NOT start the print."""
    return po.send(project, ip, name, out, progress=_progress)


def printer_start(ip: str, filename: str, confirm: bool = False) -> dict:
    """START A PRINT of a file already on the printer. Refuses unless confirm=True: ask the user first, every time."""
    return po.start(ip, filename, confirm)


def printer_pause(ip: str, confirm: bool = False) -> dict:
    """Pause the current print (needs confirm=True)."""
    return po.pause(ip, confirm)


def printer_resume(ip: str, confirm: bool = False) -> dict:
    """Resume a paused print (needs confirm=True)."""
    return po.resume(ip, confirm)


def printer_stop(ip: str, confirm: bool = False) -> dict:
    """STOP the current print for good (needs confirm=True; ask the user first)."""
    return po.stop(ip, confirm)


def printer_delete(ip: str, filename: str, confirm: bool = False) -> dict:
    """Delete a file from the printer's storage (needs confirm=True)."""
    return po.delete(ip, filename, confirm)


TOOLS = [list_machines, new_project, project_info, add_model, add_text, list_fonts, remove_object, set_object, hollow_object, arrange, set_print,
         list_resins, apply_resin, save_resin, delete_resin, list_library, save_to_library, add_from_library,
         delete_from_library,
         set_supports, check_project, export_project, export_3mf, import_project, export_goo, validate_goo, find_printers, printer_status, printer_files,
         printer_upload, send_project, printer_start, printer_pause, printer_resume, printer_stop, printer_delete]
