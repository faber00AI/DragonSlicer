"""Auto-update from GitHub Releases: checks the latest release, downloads the new exe(s) and swaps them after the app closes."""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from .version import GITHUB_REPO, __version__

CRLF, LF = chr(13) + chr(10), chr(10)
API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
HEADERS = {"User-Agent": "DragonSlicer-updater", "Accept": "application/vnd.github+json"}


def _vt(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4]) or (0,)


def check(timeout=8):
    """Returns {version, notes, url, assets:{name: url}} when a newer release exists, else None. Raises on network errors."""
    req = urllib.request.Request(API, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    tag = d.get("tag_name", "")
    if _vt(tag) <= _vt(__version__):
        return None
    return {"version": tag.lstrip("vV"), "notes": d.get("body") or "", "page": d.get("html_url", ""),
            "assets": {a["name"]: a["browser_download_url"] for a in d.get("assets", [])}}


def can_self_update():
    return bool(getattr(sys, "frozen", False)) and os.name == "nt"


def download(url, progress=None):
    dest = Path(tempfile.mkdtemp(prefix="dsupdate_")) / Path(url).name
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done * 100 // total)
    return dest


def apply_and_restart(new_files):
    """new_files: {downloaded_path: target_path}. Spawns a batch that keeps retrying to move each file over its target until this
    process (and the PyInstaller bootloader that locks the exe) has exited, then relaunches the app."""
    exe = Path(sys.executable)
    lines = ["@echo off", "set /a n=0"]
    for i, (src, dst) in enumerate(new_files.items()):
        lines += [f":retry{i}",
                  f'move /Y "{src}" "{dst}" >nul 2>&1 && goto done{i}',
                  "set /a n+=1",
                  f"if %n% GEQ 180 goto fail",
                  "ping -n 2 127.0.0.1 >nul",
                  f"goto retry{i}",
                  f":done{i}", "set /a n=0"]
    lines += ["set PYINSTALLER_RESET_ENVIRONMENT=1",
              f'start "" "{exe}"',
              ":fail",
              '(goto) 2>nul & del "%~f0"']
    bat = Path(tempfile.mkdtemp(prefix="dsupdate_")) / "update.bat"
    with open(bat, "w", encoding="mbcs", newline=CRLF) as f:         # one CRLF, not CR CR LF
        f.write(LF.join(lines) + LF)
    env = {k: v for k, v in os.environ.items() if not k.startswith("_PYI") and k != "_MEIPASS2"}   # the new exe must unpack by itself
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    subprocess.Popen(["cmd", "/c", str(bat)], env=env, creationflags=0x08000000 | 0x00000200, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
