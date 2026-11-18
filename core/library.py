"""Personal model library: .obj files in ./library (name = file name). Whatever is added is stored as OBJ, as it is
(no repair, no rescaling), so a model comes back from the library exactly as it went in."""
import re
from pathlib import Path

import numpy as np
import trimesh

from .machine import APP, FROZEN, ROOT

DIR = APP / "library"
if FROZEN and not DIR.exists():                  # first run of the portable exe: start from the bundled library
    import shutil
    try:
        shutil.copytree(ROOT / "library", DIR)
    except OSError:
        pass
MODEL_EXTS = {".stl", ".obj", ".3mf", ".ply", ".off"}


class LibraryError(Exception):
    pass


def _clean(name: str) -> str:
    n = re.sub(r'[\/:*?"<>|\x00-\x1f]', "_", name.strip()).strip(". ")
    if not n:
        raise LibraryError("il modello ha bisogno di un nome")
    if len(n) > 80:
        raise LibraryError("nome troppo lungo (massimo 80 caratteri)")
    return n


def _path(name: str) -> Path:
    return DIR / (_clean(name) + ".obj")


def names() -> list[str]:
    return sorted((p.stem for p in DIR.glob("*.obj")), key=str.lower) if DIR.is_dir() else []


def list_models() -> list[dict]:
    """[{name, size_kb}] (the mesh itself is only read when used)."""
    return [dict(name=n, size_kb=round((DIR / (n + ".obj")).stat().st_size / 1024, 1)) for n in names()]


def load(name: str) -> trimesh.Trimesh:
    p = _path(name)
    if not p.is_file():
        raise LibraryError(f"il modello {name!r} non è nella libreria")
    m = trimesh.load(p, force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise LibraryError(f"{p.name}: file non valido")
    return m


def save_mesh(name: str, mesh: trimesh.Trimesh, overwrite: bool = False) -> str:
    """Store a mesh (already in mm). It is centred on X/Y and rests on Z=0, like every part dropped on the plate."""
    if mesh is None or len(mesh.faces) == 0:
        raise LibraryError("mesh vuota")
    p = _path(name)
    if p.exists() and not overwrite:
        raise LibraryError(f"esiste già un modello {p.stem!r} nella libreria")
    m = mesh.copy()
    b = m.bounds
    m.apply_translation([-(b[0][0] + b[1][0]) / 2, -(b[0][1] + b[1][1]) / 2, -b[0][2]])
    DIR.mkdir(exist_ok=True)
    m.export(p, file_type="obj")
    return p.stem


def add_file(path, name: str | None = None, overwrite: bool = False) -> str:
    src = Path(path)
    if src.suffix.lower() not in MODEL_EXTS:
        raise LibraryError(f"{src.name}: formato non supportato ({', '.join(sorted(MODEL_EXTS))})")
    try:
        m = trimesh.load(src, force="mesh")
    except Exception as e:
        raise LibraryError(f"{src.name}: {e}")
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise LibraryError(f"{src.name}: nessuna mesh valida")
    return save_mesh(name or src.stem, m, overwrite)


def delete(name: str):
    p = _path(name)
    if not p.is_file():
        raise LibraryError(f"il modello {name!r} non è nella libreria")
    p.unlink()


def rename(old: str, new: str):
    a, b = _path(old), _path(new)
    if not a.is_file():
        raise LibraryError(f"il modello {old!r} non è nella libreria")
    if b.exists() and a != b:
        raise LibraryError(f"esiste già un modello {b.stem!r}")
    a.rename(b)
