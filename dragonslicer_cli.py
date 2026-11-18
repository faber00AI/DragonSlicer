"""Command line for the DragonSlicer pipeline (every command prints one JSON document; progress goes to stderr).

    python dragonslicer_cli.py new_project --project job1 --machine "Saturn 4 Ultra"
    python dragonslicer_cli.py add_model C:\\models\\figure.stl --project job1
    python dragonslicer_cli.py set_object 0 --project job1 --size_z 60
    python dragonslicer_cli.py arrange --project job1
    python dragonslicer_cli.py set_supports --project job1 --preset Tree_Medium
    python dragonslicer_cli.py check_project --project job1
    python dragonslicer_cli.py send_project 192.168.1.49 --project job1 --name figure.goo
    python dragonslicer_cli.py printer_start 192.168.1.49 figure.goo --confirm     # only after the user agrees

`python dragonslicer_cli.py --help` lists every command; `<command> --help` its options.
"""
import argparse
import inspect
import json
import os
import sys
import types
import typing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.agent_api import TOOLS, ProjectError  # noqa: E402


def _kind(ann):
    if typing.get_origin(ann) in (typing.Union, types.UnionType):
        ann = next(a for a in typing.get_args(ann) if a is not type(None))
    return ann


def build():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    for fn in TOOLS:
        sp = sub.add_parser(fn.__name__, help=(fn.__doc__ or "").split("\n")[0], description=fn.__doc__)
        for name, p in inspect.signature(fn).parameters.items():
            kind = _kind(p.annotation)
            if p.default is inspect.Parameter.empty:
                sp.add_argument(name, type=kind)
            elif kind is bool:
                sp.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=p.default)
            else:
                sp.add_argument("--" + name, type=kind, default=p.default)
    return ap


def main(argv=None):
    a = vars(build().parse_args(argv))
    cmd = a.pop("cmd")
    fn = next(f for f in TOOLS if f.__name__ == cmd)
    try:
        out = fn(**a)
    except ProjectError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 2
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        return 1
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
