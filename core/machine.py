import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
ROOT = Path(sys._MEIPASS) if FROZEN else Path(__file__).resolve().parent.parent     # read-only resources: machines, presets
APP = Path(sys.executable).parent if FROZEN else ROOT        # writable data (library, user resins, settings): beside the exe


@dataclass
class Machine:
    name: str = "ELEGOO Saturn 4 Ultra"
    bed_x: float = 218.88
    bed_y: float = 122.88
    bed_z: float = 220.0
    res_x: int = 11520
    res_y: int = 5120
    mirror_x: int = 0
    mirror_y: int = 1
    brand: str = "ELEGOO"
    format: str = "goo"
    density: float = 1.1
    pixel_um: list = field(default_factory=list)
    platform_id: int = 0
    defaults: dict = field(default_factory=dict)

    @property
    def px_size(self) -> float:
        return self.bed_x / self.res_x


@dataclass
class PrintSettings:
    """Same parameter set as SatelLite's resin profile. Speeds mm/min, distances mm, times s.
    Stage 1 = first lift segment, stage 2 ("lift2") = the second segment. The retract goes down in two stages as well:
    the last `retract2_height` mm at `retract2_speed`, the rest at `retract_speed`. A bottom_* value applies to the bottom layers."""
    layer_height: float = 0.05
    transition_layers: int = 5
    bottom_layers: int = 5
    exposure: float = 2.5
    bottom_exposure: float = 30.0
    light_off_delay: float = 1.0
    bottom_light_off_delay: float = 1.0
    wait_before_cure: float = 0.0        # ChituBox "rest before release" (goo wait_before_cure)
    bottom_wait_before_cure: float = 0.0
    wait_after_cure: float = 0.0         # ChituBox "rest after release" (goo wait_after_cure)
    bottom_wait_after_cure: float = 0.0
    wait_after_lift: float = 0.0         # ChituBox "rest after retract" (goo wait_after_lift)
    bottom_wait_after_lift: float = 0.0
    lift_height: float = 5.0
    lift_speed: float = 65.0
    bottom_lift_height: float = 5.0
    bottom_lift_speed: float = 65.0
    lift2_height: float = 0.0
    lift2_speed: float = 0.0
    bottom_lift2_height: float = 0.0
    bottom_lift2_speed: float = 0.0
    retract_speed: float = 150.0
    bottom_retract_speed: float = 150.0
    retract2_height: float = 0.0
    retract2_speed: float = 0.0
    bottom_retract2_height: float = 0.0
    bottom_retract2_speed: float = 0.0
    pwm: int = 255
    bottom_pwm: int = 255
    resin: str = "Generic"
    resin_type: str = ""
    density: float = 1.1                 # g/ml
    cost: float = 0.0                    # price per litre of resin
    embed_motion: bool = False          # write lift/retract into the .goo (SatelLite leaves them 0: the printer's own profile drives motion)

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PrintSettings":
        ps = cls()
        for k, v in d.items():
            if k in cls.__dataclass_fields__:
                setattr(ps, k, type(getattr(ps, k))(v))
        return ps

    @classmethod
    def from_machine(cls, m: Machine) -> "PrintSettings":
        ps = cls.from_dict({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.defaults.items()})
        ps.density = m.density
        return ps

    def stage_times(self, bottom: bool) -> float:
        """Seconds of one layer cycle without the exposure itself (lift, retract, waits, light-off delay)."""
        g = (lambda n: getattr(self, ("bottom_" if bottom else "") + n))
        lift1, lift2 = g("lift_height"), g("lift2_height")
        total = lift1 + lift2
        r2 = min(g("retract2_height"), total)
        legs = ((lift1, g("lift_speed")), (lift2, g("lift2_speed")), (total - r2, g("retract_speed")), (r2, g("retract2_speed")))
        move = sum(d / v * 60 for d, v in legs if d > 0 and v > 0)     # a 0 speed means the printer's own setting applies
        return move + g("light_off_delay") + g("wait_before_cure") + g("wait_after_cure") + g("wait_after_lift")

    def layer_exposure(self, i: int) -> float:
        """Exposure of layer i (0-based): bottom layers, then a linear ramp over `transition_layers` down to the normal
        exposure (50, 42, 34, 26, 18, 10, 2 for 5 transition layers like ChituBox), then the normal exposure."""
        if i < self.bottom_layers:
            return self.bottom_exposure
        j = i - self.bottom_layers
        n = self.transition_layers
        if j < n:
            return self.bottom_exposure - (self.bottom_exposure - self.exposure) * (j + 1) / (n + 1)
        return self.exposure

    def estimate_seconds(self, layers: int) -> float:
        nb = min(self.bottom_layers, layers)
        nt = min(self.transition_layers, layers - nb)
        ramp = sum(self.layer_exposure(nb + j) for j in range(nt))
        return (nb * (self.bottom_exposure + self.stage_times(True)) + ramp + nt * self.stage_times(False)
                + (layers - nb - nt) * (self.exposure + self.stage_times(False)))


def load_machines() -> list[Machine]:
    out = []
    for f in sorted((ROOT / "machines").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d.pop("note", None)
        out.append(Machine(**{k: v for k, v in d.items() if k in Machine.__dataclass_fields__}))
    out.sort(key=lambda m: (m.brand, m.name))
    return out or [Machine()]


def load_support_presets() -> dict:
    return json.loads((ROOT / "presets" / "supports.json").read_text(encoding="utf-8"))
