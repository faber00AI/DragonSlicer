"""Validate .goo files and, with two files, show how their headers differ.

    python tools/check_goo.py job.goo
    python tools/check_goo.py ours.goo real_from_satellite.goo      # also prints header / layer-definition differences
"""
import pprint
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.goo_check import check  # noqa: E402


def main(paths):
    infos = []
    for p in paths:
        problems, info = check(p)
        print(f"\n== {p}  ({info['size'] / 1e6:.1f} MB, {info['layers']}/{info['declared']} layers, "
              f"{info['res'][0]}x{info['res'][1]}, {info['layer_height']} mm)")
        print("   " + " | ".join(f"{k}={v}" for k, v in info["text"].items()))
        print("   VALID" if not problems else "   PROBLEMS:")
        for x in problems[:20]:
            print("     -", x)
        infos.append(info)
    if len(infos) == 2:
        a, b = infos
        print("\n-- header fields that differ (first file vs second) --")
        for k in a["header"]:
            if a["header"][k] != b["header"][k]:
                print(f"   {k}: {a['header'][k]}  |  {b['header'][k]}")
        print("-- first layer definition (first vs second) --")
        for k in a["first_layer_def"]:
            if a["first_layer_def"][k] != b["first_layer_def"][k]:
                print(f"   {k}: {a['first_layer_def'][k]}  |  {b['first_layer_def'][k]}")


if __name__ == "__main__":
    main(sys.argv[1:])
