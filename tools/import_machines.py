"""Import printer profiles from SatelLite's officialPlatform.ini into machines/*.json.

    python tools/import_machines.py                 # ELEGOO only, default SatelLite install
    python tools/import_machines.py --all           # every brand in the file
    python tools/import_machines.py --brand Anycubic --src path/to/officialPlatform.ini

The file is a QSettings INI with Qt @Variant floats, so it is read with QSettings itself.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QSettings

DEFAULT_SRC = r"C:\Program Files\ELEGOO SatelLite\resources\machines\printer\officialPlatform.ini"
OUT = Path(__file__).resolve().parent.parent / "machines"

# profile key -> (output key, cast)
DEFAULTS = {
    "Printer/LayerThickness": ("layer_height", float),
    "Printer/BottomLayerCount": ("bottom_layers", int),
    "Printer/ExposureTime": ("exposure", float),
    "Printer/BottomExposureTime": ("bottom_exposure", float),
    "Printer/LightoffDelay": ("light_off_delay", float),
    "Printer/LiftingDis": ("lift_height", float),
    "Printer/LiftingSpeed": ("lift_speed", float),          # mm/min
    "Printer/BackSpeed": ("retract_speed", float),          # mm/min
    "Printer/BottomLiftDis": ("bottom_lift_height", float),
    "Printer/BottomLiftSpeed": ("bottom_lift_speed", float),
    "Printer/BottomRetractSpeed": ("bottom_retract_speed", float),
    "Printer/TranslateLayerCount": ("transition_layers", int),
    "Advanced/LightPwm": ("pwm", int),
    "Advanced/BottomLightPwm": ("bottom_pwm", int),
}


def num(v, cast=float):
    return cast(float(v))


def flag(v):
    return 1 if str(v).lower() in ("true", "1") else 0


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--brand", default="ELEGOO")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not Path(a.src).is_file():
        sys.exit(f"file non trovato: {a.src}")
    QCoreApplication([])
    s = QSettings(str(Path(a.src).resolve()), QSettings.IniFormat)
    ids = s.value("platformIds") or []
    OUT.mkdir(exist_ok=True)
    n = 0
    for i in ids:
        brand = str(s.value(f"{i}/Brand"))
        if not a.all and brand.lower() != a.brand.lower():
            continue
        g = lambda k, d=None: s.value(f"{i}/{k}", d)
        defaults = {}
        for k, (out, cast) in DEFAULTS.items():
            v = g(k)
            if v not in (None, ""):
                defaults[out] = num(v, cast)
        m = {
            "name": str(g("Name")), "brand": brand,
            "bed_x": round(num(g("XLength")), 3), "bed_y": round(num(g("YWidth")), 3), "bed_z": num(g("ZHeight")),
            "res_x": num(g("ResX"), int), "res_y": num(g("ResY"), int),
            "mirror_x": flag(g("MirrorX")), "mirror_y": flag(g("MirrorY")),
            "pixel_um": [num(g("PixX")), num(g("PixY"))],
            "format": str(g("UVtoolsFormat") or ""),
            "density": round(num(g("Resin/ResinDensity", 1.1)), 3),
            "platform_id": int(i),
            "defaults": defaults,
        }
        (OUT / f"{slug(m['name'])}.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        n += 1
        print(f"{m['name']:<28} {m['bed_x']:g}x{m['bed_y']:g}x{m['bed_z']:g} mm  {m['res_x']}x{m['res_y']}  "
              f"mirror({m['mirror_x']},{m['mirror_y']})  {m['format'] or '-'}")
    print(f"\n{n} profili scritti in {OUT}")


if __name__ == "__main__":
    main()
