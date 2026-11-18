"""Import SatelLite's resin library (resources/resin/material/*.data, JSON) into presets/resin_library.json.

    python tools/import_resins.py [--src "C:\Program Files\ELEGOO SatelLite\resources\resin\material"]

Output: {machine name: [{"name", "company", "type", "color", "settings": {PrintSettings fields}}]}.
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_SRC = r"C:\Program Files\ELEGOO SatelLite\resources\resin\material"
OUT = Path(__file__).resolve().parent.parent / "presets" / "resin_library.json"

# SatelLite key (inside vd_printer_printer_para / _advanced_para / _resin_para) -> PrintSettings field
PRINTER = {
    "thickness": "layer_height", "tanslate_layer_count": "transition_layers",
    "bottom_layer": "bottom_layers", "exposure_time": "exposure", "bottom_exposure_time": "bottom_exposure",
    "lightoff_delay": "light_off_delay", "bottom_lightoff_delay": "bottom_light_off_delay",
    "wait_after_cure": "wait_after_cure",
    "lift_dis": "lift_height", "lift_speed": "lift_speed", "bootom_lift_dis": "bottom_lift_height",
    "bottom_lift_speed": "bottom_lift_speed",
    "slow_lift_dis": "lift2_height", "slow_lift_speed": "lift2_speed",
    "slow_bottom_lift_dis": "bottom_lift2_height", "slow_bottom_lift_speed": "bottom_lift2_speed",
    "back_speed": "retract_speed", "bottom_back_speed": "bottom_retract_speed",
    "slow_back_dis": "retract2_height", "slow_back_speed": "retract2_speed",
    "slow_bottom_back_dis": "bottom_retract2_height", "slow_bottom_back_speed": "bottom_retract2_speed",
}
PRINTER_FULL = {  # these keys have no "para_" in the middle
    "vd_printer_printer_bottom_wait_before_lift": "bottom_wait_after_cure",
    "vd_printer_printer_bottom_wait_after_lift": "bottom_wait_after_lift",
    "vd_printer_printer_bottom_wait_after_retract": "bottom_wait_before_cure",
}
ADVANCED = {"vd_printer_advanced_para_led_power": "pwm", "vd_printer_advanced_para_bottom_led_power": "bottom_pwm"}
RESIN = {"vd_printer_resin_para_type": "resin_type", "vd_printer_resin_para_density": "density",
         "vd_printer_resin_para_cost": "cost"}


def convert(d):
    v = d["vd_printer"]["vd_active_printer"]
    s = {}
    for k, f in PRINTER.items():
        for full in (f"vd_printer_printer_para_{k}", f"vd_printer_printer_{k}"):
            if full in v["vd_printer_printer_para"]:
                s[f] = v["vd_printer_printer_para"][full]
                break
    for full, f in PRINTER_FULL.items():
        if full in v["vd_printer_printer_para"]:
            s[f] = v["vd_printer_printer_para"][full]
    for full, f in ADVANCED.items():
        s[f] = v["vd_printer_advanced_para"][full]
    for full, f in RESIN.items():
        s[f] = v["vd_printer_resin_para"][full]
    s["layer_height"] = round(s["layer_height"], 4)
    s["density"] = round(s["density"], 3)
    s["resin"] = str(v.get("vd_resine_profilename", ""))
    ex = v["cloudResinExtraParam"]
    return ex["cloudResinPrinterName"], dict(name=s["resin"], company=str(v.get("vd_resine_company", "")),
                                             type=s["resin_type"], color=ex.get("cloudResinColor", ""), settings=s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    a = ap.parse_args()
    src = Path(a.src)
    if not src.is_dir():
        sys.exit(f"cartella non trovata: {src}")
    lib, bad = {}, 0
    for f in sorted(src.glob("*.data"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        try:
            machine, prof = convert(json.loads(f.read_text(encoding="utf-8")).get("platitem", {}))
        except Exception:
            bad += 1
            continue
        lib.setdefault(machine, []).append(prof)
    for v in lib.values():
        v.sort(key=lambda p: p["name"].lower())
    OUT.write_text(json.dumps(lib, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{sum(map(len, lib.values()))} profili per {len(lib)} stampanti ({bad} scartati) -> {OUT}")


if __name__ == "__main__":
    main()
