"""Resin profiles: the SatelLite library (presets/resin_library.json, per printer) plus the user's own profiles
(presets/resins_user.json, per printer). A profile is a name plus the PrintSettings fields."""
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from .machine import APP, ROOT, PrintSettings

LIBRARY = ROOT / "presets" / "resin_library.json"
USER = APP / "presets" / "resins_user.json"
FIELDS = tuple(PrintSettings.__dataclass_fields__)


class ResinError(Exception):
    pass


@lru_cache(maxsize=1)
def _library() -> dict:
    return json.loads(LIBRARY.read_text(encoding="utf-8")) if LIBRARY.is_file() else {}


def _user() -> dict:
    if not USER.is_file():
        return {}
    try:
        return json.loads(USER.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ResinError(f"{USER.name} non è un JSON valido: {e}")


def _write_user(d: dict):
    USER.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=USER.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, USER)


def list_profiles(machine: str) -> list[dict]:
    """[{name, source: 'user'|'library', type, color, settings}], the user's first."""
    out = [dict(name=n, source="user", type=s.get("resin_type", ""), color="", settings=s)
           for n, s in sorted(_user().get(machine, {}).items(), key=lambda kv: kv[0].lower())]
    out += [dict(name=p["name"], source="library", type=p["type"], color=p["color"], settings=p["settings"])
            for p in _library().get(machine, [])]
    return out


def get_profile(machine: str, name: str) -> dict:
    for p in list_profiles(machine):
        if p["name"] == name:
            return p
    raise ResinError(f"profilo resina {name!r} non trovato per {machine}")


def to_settings(machine, profile: dict) -> PrintSettings:
    """The machine defaults overlaid with the profile (the profile may be partial)."""
    ps = PrintSettings.from_machine(machine)
    for k, v in profile["settings"].items():
        if k in FIELDS:
            setattr(ps, k, type(getattr(ps, k))(v))
    ps.resin = profile["name"]
    return ps


def save_user_profile(machine: str, name: str, ps: PrintSettings, overwrite: bool = True):
    name = name.strip()
    if not name:
        raise ResinError("il profilo ha bisogno di un nome")
    if len(name) > 31:
        raise ResinError("nome troppo lungo (massimo 31 caratteri: è il limite del campo nel file .goo)")
    d = _user()
    if not overwrite and name in d.get(machine, {}):
        raise ResinError(f"esiste già un profilo utente {name!r}")
    s = ps.as_dict()
    s["resin"] = name
    d.setdefault(machine, {})[name] = s
    _write_user(d)


def export_user_profiles(path) -> int:
    """Write every user profile (all printers) to `path` as JSON; returns how many."""
    d = _user()
    Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return sum(len(v) for v in d.values())


def import_user_profiles(data: dict) -> int:
    """Merge {machine: {name: settings}} into the user profiles (same names are overwritten); returns how many."""
    d = _user()
    n = 0
    for machine, profs in data.items():
        if not isinstance(profs, dict):
            continue
        for name, s in profs.items():
            if isinstance(s, dict):
                d.setdefault(machine, {})[name] = s
                n += 1
    _write_user(d)
    return n


def delete_user_profile(machine: str, name: str):
    d = _user()
    if name not in d.get(machine, {}):
        raise ResinError(f"{name!r} non è un profilo utente (quelli della libreria non si cancellano)")
    del d[machine][name]
    if not d[machine]:
        del d[machine]
    _write_user(d)
