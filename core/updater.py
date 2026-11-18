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
    """new_files: {downloaded_path: target_path}. Spawns a batch that waits for this process to exit, swaps the files, relaunches the app."""
    exe = Path(sys.executable)
    lines = ["@echo off", "setlocal"]
    lines.append(f'set PID={os.getpid()}')
    lines.append(":wait")
    lines.append('tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul && (timeout /t 1 /nobreak >nul & goto wait)')
    for src, dst in new_files.items():
        lines.append(f':try_{abs(hash(str(dst)))}')
        lines.append(f'move /Y "{src}" "{dst}" >nul 2>&1 || (timeout /t 1 /nobreak >nul & goto try_{abs(hash(str(dst)))})')
    lines.append(f'start "" "{exe}"')
    lines.append('(goto) 2>nul & del "%~f0"')
    bat = Path(tempfile.mkdtemp(prefix="dsupdate_")) / "update.bat"
    bat.write_text("\r\n".join(lines) + "\r\n", encoding="mbcs")
    subprocess.Popen(["cmd", "/c", str(bat)], creationflags=0x08000000 | 0x00000008, close_fds=True)
