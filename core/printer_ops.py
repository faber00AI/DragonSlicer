"""Printer operations as plain functions returning JSON-serialisable dicts (shared by the CLI and the MCP server).

Anything that moves the printer (start / pause / resume / stop / delete) needs confirm=True: the caller (agent) must
have asked the user first.
"""
import time
from pathlib import Path

from . import printer_link as pl
from .project import ProjectError, _load, _machine, _norm, export


def _need(confirm, what):
    if not confirm:
        raise ProjectError(f"{what} needs confirm=True - ask the user before doing it")


def _printer(ip) -> pl.Printer:
    p = pl.probe(ip, timeout=3)
    if p is None:
        raise ProjectError(f"no printer answered at {ip} (port 3000/UDP); is it on and on the same network?")
    return p


def _state(st):
    pi = (st or {}).get("PrintInfo", {})
    ms = (st or {}).get("CurrentStatus", [])
    return dict(machine_status=[pl.MACHINE_STATUS.get(x, x) for x in ms], print_status=pl.PRINT_STATUS.get(pi.get("Status"), pi.get("Status")),
                file=pi.get("Filename"), layer=pi.get("CurrentLayer"), total_layers=pi.get("TotalLayer"),
                busy=bool(ms and ms != [0]) or pi.get("Status") not in (None, 0, 8, 9))


def discover(timeout=3.0) -> dict:
    return dict(printers=[dict(ip=p.ip, name=p.name, machine=p.machine, firmware=p.firmware, protocol=p.protocol) for p in pl.discover(timeout)])


def status(ip) -> dict:
    p = _printer(ip)
    with pl.Link(p) as link:
        return dict(ip=ip, name=p.name, machine=p.machine, firmware=p.firmware, **_state(link.status()))


def files(ip) -> dict:
    p = _printer(ip)
    with pl.Link(p) as link:
        return dict(files=[f.get("name") for f in link.files()])


def upload(ip, path, remote_name=None, project=None, progress=None) -> dict:
    """Send a .goo to the printer's storage (does not start anything). With `project`, refuses a different printer model."""
    path = Path(path)
    if not path.is_file():
        raise ProjectError(f"{path} does not exist")
    p = _printer(ip)
    if project:
        want, got = _norm(_machine(_load(project)).name), _norm(p.machine or "")
        if got and want != got:
            raise ProjectError(f"the project is sliced for {_machine(_load(project)).name} but the printer is {p.machine}")
    name = remote_name or path.name
    md5 = pl.upload(p, path, name, progress=progress)
    time.sleep(1.5)
    with pl.Link(p) as link:
        present = any(f.get("name", "").endswith("/" + name) or f.get("name") == name for f in link.files())
    return dict(uploaded=name, bytes=path.stat().st_size, md5=md5, on_printer=present)


def start(ip, filename, confirm=False) -> dict:
    """Start printing a file that is already on the printer. Refuses if the printer is busy or the file is missing."""
    _need(confirm, "starting a print")
    p = _printer(ip)
    with pl.Link(p) as link:
        st = _state(link.status())
        if st["busy"]:
            raise ProjectError(f"the printer is busy: {st}")
        names = [f.get("name", "") for f in link.files()]
        match = next((n for n in names if n == filename or n.endswith("/" + filename)), None)
        if match is None:
            raise ProjectError(f"{filename} is not on the printer; upload it first")
        link.start_print(match.removeprefix("/local/"))
        time.sleep(2)
        return dict(started=match, **_state(link.status()))


def _control(ip, action, confirm):
    _need(confirm, action)
    with pl.Link(_printer(ip)) as link:
        getattr(link, action)()
        time.sleep(1)
        return dict(done=action, **_state(link.status()))


def pause(ip, confirm=False):
    return _control(ip, "pause", confirm)


def resume(ip, confirm=False):
    return _control(ip, "resume", confirm)


def stop(ip, confirm=False):
    return _control(ip, "stop", confirm)


def delete(ip, filename, confirm=False) -> dict:
    _need(confirm, "deleting a file")
    with pl.Link(_printer(ip)) as link:
        names = [f.get("name", "") for f in link.files()]
        match = next((n for n in names if n == filename or n.endswith("/" + filename)), None)
        if match is None:
            raise ProjectError(f"{filename} is not on the printer")
        link.delete(match)
        time.sleep(1.5)
        return dict(deleted=match, still_there=any(f.get("name") == match for f in link.files()))


def send(project, ip, name=None, out=None, progress=None) -> dict:
    """export + upload in one step (never starts the print)."""
    import tempfile
    out = Path(out) if out else Path(tempfile.mkdtemp()) / "job.goo"
    ex = export(project, out)
    if not ex["valid"]:
        raise ProjectError(f"the exported file is not valid: {ex['problems']}")
    up = upload(ip, out, name or (Path(project).resolve().name + ".goo"), project=project, progress=progress)
    return dict(export=ex, upload=up)
