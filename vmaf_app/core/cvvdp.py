"""ColorVideoVDP (CVVDP) settings: the display it models, presets, and Vship's config.

CVVDP predicts how visible the differences between two videos are to a person
watching a particular display from a particular distance in a particular room,
so its score depends on that display as much as on the videos: on the same
Beekeeper frames an AV1 encode scored 9.76 JOD on a 4K HDR monitor and 9.28 on
a 24-inch 1080p one. The display is therefore part of what a CVVDP score
means -- it goes into the cache identity -- and is described here by its
physical properties, never by a name, so renaming a preset changes nothing.

The built-in presets are the official ColorVideoVDP display models
(pycvvdp/vvdp_data/display_models.json, v0.5.7). Where an official model
leaves a property out, it is filled with the value the official tool itself
uses when unspecified (contrast 1000:1, reflectivity 0.005, exposure 1).

The score is calculated on the GPU by Vship only: the official CPU
implementation needs PyTorch (977 MB installed) and scored a 4K frame in
1.5 s, against 40-78 frames a second with Vship. Vship agreed with the
official implementation to 0.016 JOD on the Beekeeper pair.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

#: The key Vship's config file defines the display under. Any name that is not
#: one of Vship's built-in models works; every property is given explicitly.
VSHIP_MODEL_KEY = "videometricslab_display"


@dataclass(frozen=True, slots=True)
class CvvdpDisplay:
    """A display as CVVDP models it."""

    width: int = 3840
    height: int = 2160
    diagonal_inches: float = 30.0
    viewing_distance_m: float = 0.7472
    peak_luminance: float = 200.0     # cd/m^2 (nits)
    contrast: float = 1000.0          # peak : black, e.g. 1000 for 1000:1
    ambient_lux: float = 250.0        # light falling on the screen
    reflectivity: float = 0.005       # share of ambient light the screen reflects
    exposure: float = 1.0
    hdr: bool = False

    def validated(self) -> CvvdpDisplay:
        """Raises ValueError for a display CVVDP cannot model."""
        numbers = (self.width, self.height, self.diagonal_inches, self.viewing_distance_m,
                   self.peak_luminance, self.contrast, self.ambient_lux, self.reflectivity, self.exposure)
        # Only reachable from a hand-edited settings file. Infinity passed
        # the checks below and then made the cache key's JSON encoding
        # raise; text or a list raised TypeError.
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                   for value in numbers):
            raise ValueError("Every display value must be a finite number.")
        if self.width < 16 or self.height < 16:
            raise ValueError("The display resolution must be at least 16 x 16.")
        for label, value in (("screen size", self.diagonal_inches), ("viewing distance", self.viewing_distance_m),
                             ("peak brightness", self.peak_luminance), ("contrast", self.contrast),
                             ("exposure", self.exposure)):
            if not value > 0:
                raise ValueError(f"The display's {label} must be greater than zero.")
        if self.ambient_lux < 0 or not 0 <= self.reflectivity < 1:
            raise ValueError("Ambient light cannot be negative and reflectivity must be between 0 and 1.")
        return self

    @property
    def height_m(self) -> float:
        """The screen's height in metres, from its diagonal and aspect ratio."""
        aspect = self.width / self.height
        return self.diagonal_inches * 0.0254 / (1 + aspect * aspect) ** 0.5

    @property
    def distance_in_heights(self) -> float:
        """The viewing distance as a multiple of the screen's height."""
        return self.viewing_distance_m / self.height_m

    def identity(self) -> dict:
        """What a score depends on: the physical properties, rounded so
        that a value typed or read back as 0.7472 is the same display."""
        return {
            "resolution": [int(self.width), int(self.height)],
            "diagonal_inches": round(float(self.diagonal_inches), 4),
            "viewing_distance_m": round(float(self.viewing_distance_m), 4),
            "peak_luminance": round(float(self.peak_luminance), 4),
            "contrast": round(float(self.contrast), 4),
            "ambient_lux": round(float(self.ambient_lux), 4),
            "reflectivity": round(float(self.reflectivity), 6),
            "exposure": round(float(self.exposure), 4),
            "hdr": bool(self.hdr),
        }

    def describe(self) -> str:
        """One line for tooltips: '30" 3840x2160 SDR, 200 nits, 250 lux, 0.75 m'."""
        return (f'{self.diagonal_inches:g}" {self.width}x{self.height} {"HDR" if self.hdr else "SDR"}, '
                f"{self.peak_luminance:g} nits, {self.ambient_lux:g} lux, {self.viewing_distance_m:.2f} m "
                f"({self.distance_in_heights:.1f} x screen height)")


@dataclass(frozen=True, slots=True)
class CvvdpSettings:
    """Everything that changes a CVVDP score besides the two videos."""

    display: CvvdpDisplay = CvvdpDisplay()
    #: Scale the compared frames up to fill the display, aspect kept (Vship's
    #: resizeToDisplay). Off is the official tool's default: frames are shown
    #: pixel for pixel, so a 1080p picture covers a quarter of a 4K display.
    resize_to_display: bool = False

    def spec_parameters(self) -> tuple[tuple[str, object], ...]:
        return (("display", self.display.identity()), ("resize_to_display", bool(self.resize_to_display)))

    def to_dict(self) -> dict:
        return {"display": asdict(self.display), "resize_to_display": self.resize_to_display}

    @classmethod
    def from_dict(cls, data: dict) -> CvvdpSettings:
        known = set(CvvdpDisplay.__dataclass_fields__)
        display = CvvdpDisplay(**{k: v for k, v in dict(data.get("display", {})).items() if k in known})
        return cls(display.validated(), bool(data.get("resize_to_display", False)))

    def same_as(self, other: CvvdpSettings) -> bool:
        return self.spec_parameters() == other.spec_parameters()

    @classmethod
    def from_spec_parameters(cls, parameters) -> CvvdpSettings:
        """The settings a CVVDP request was made with -- the rounded values
        of its identity, so what is scored is exactly what is cached."""
        values = dict(parameters)
        display = dict(values["display"])
        width, height = display.pop("resolution")
        return cls(CvvdpDisplay(width=width, height=height, **display).validated(),
                   bool(values["resize_to_display"]))


def write_vship_config(display: CvvdpDisplay, path: Path) -> Path:
    """Vship reads a custom display from a JSON file (not a string), in the
    official display_models.json format; every property is set so none falls
    back to a Vship default."""
    model = {
        "name": "VideoMetricsLab display",
        "colorspace": "HDR" if display.hdr else "SDR",
        "resolution": [int(display.width), int(display.height)],
        "viewing_distance_meters": float(display.viewing_distance_m),
        "diagonal_size_inches": float(display.diagonal_inches),
        "max_luminance": float(display.peak_luminance),
        "contrast": float(display.contrast),
        "E_ambient": float(display.ambient_lux),
        "k_refl": float(display.reflectivity),
        "exposure": float(display.exposure),
        "source": "VideoMetricsLab",
    }
    path.write_text(json.dumps({VSHIP_MODEL_KEY: model}, indent=1), encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class CvvdpPreset:
    """A named display. Only `settings.display` is the preset: whether the
    video is scaled to fill the display is each video's own setting, which
    choosing or saving a preset leaves alone."""

    name: str
    settings: CvvdpSettings
    builtin: bool = False
    description: str = ""


def _official(name: str, description: str, **display) -> CvvdpPreset:
    return CvvdpPreset(name, CvvdpSettings(CvvdpDisplay(**display)), builtin=True, description=description)


#: The official ColorVideoVDP display models that fit comparing video encodes
#: (the VR headsets and specific phones/tablets are left out).
BUILTIN_PRESETS: tuple[CvvdpPreset, ...] = (
    _official("30-inch 4K monitor, office", "Official 'standard_4k' -- the official tool's default display.",
              width=3840, height=2160, diagonal_inches=30, viewing_distance_m=0.7472,
              peak_luminance=200, contrast=1000, ambient_lux=250),
    _official("24-inch 1080p monitor, office", "Official 'standard_fhd'.",
              width=1920, height=1080, diagonal_inches=24, viewing_distance_m=0.6,
              peak_luminance=200, contrast=1000, ambient_lux=250),
    _official("30-inch 4K HDR monitor, dim room", "Official 'standard_hdr_pq' (HLG uses the same display).",
              width=3840, height=2160, diagonal_inches=30, viewing_distance_m=0.7472,
              peak_luminance=1500, contrast=1_000_000, ambient_lux=10, hdr=True),
    _official("30-inch 4K HDR monitor, dark room", "Official 'standard_hdr_dark'.",
              width=3840, height=2160, diagonal_inches=30, viewing_distance_m=0.7472,
              peak_luminance=1500, contrast=1_000_000, ambient_lux=0, hdr=True),
    _official("65-inch 4K HDR TV, 1000 nits, living room", "Official '65inch_hdr_pq_1Knit'.",
              width=3840, height=2160, diagonal_inches=65, viewing_distance_m=1.98,
              peak_luminance=1000, contrast=1_000_000, ambient_lux=5, hdr=True),
    _official("65-inch 4K HDR TV, 2000 nits, living room", "Official '65inch_hdr_pq_2Knit'.",
              width=3840, height=2160, diagonal_inches=65, viewing_distance_m=1.98,
              peak_luminance=2000, contrast=1_000_000, ambient_lux=5, hdr=True),
    _official("65-inch 4K HDR TV, 4000 nits, living room", "Official '65inch_hdr_pq_4knit'.",
              width=3840, height=2160, diagonal_inches=65, viewing_distance_m=1.98,
              peak_luminance=4000, contrast=1_000_000, ambient_lux=5, hdr=True),
    _official("6-inch phone", "Official 'standard_phone' (contrast unspecified there: 1000:1, the official default).",
              width=2400, height=1080, diagonal_inches=6, viewing_distance_m=0.4,
              peak_luminance=500, contrast=1000, ambient_lux=250),
)

#: The default for every video until the user saves a preset of their own.
DEFAULT_PRESET = BUILTIN_PRESETS[0]


def presets(user_presets: list[dict]) -> list[CvvdpPreset]:
    """Built-in presets, then the user's (from Settings.cvvdp_presets). A
    malformed saved preset is skipped rather than stopping the app."""
    found = list(BUILTIN_PRESETS)
    for data in user_presets:
        try:
            found.append(CvvdpPreset(str(data["name"]), CvvdpSettings.from_dict(data["settings"])))
        except (KeyError, TypeError, ValueError, AttributeError):
            # AttributeError: "settings" not a mapping, e.g. {"settings": [1]},
            # which used to escape and stop the app at startup.
            continue
    return found


def preset_named(name: str, user_presets: list[dict]) -> CvvdpPreset | None:
    return next((preset for preset in presets(user_presets) if preset.name == name), None)


def matching_preset(settings: CvvdpSettings, user_presets: list[dict],
                    preferred: str = "") -> CvvdpPreset | None:
    """The preset whose display these settings use, if any -- user presets first, since a
    user preset saved from a built-in should show under the user's name.

    `preferred` is the preset the video was given. It wins while its display
    still matches: of two presets with the same values, the later one's name
    was shown, whichever the user had chosen."""
    candidates = presets(user_presets)
    if preferred:
        for preset in candidates:
            if preset.name == preferred and preset.settings.display.identity() == settings.display.identity():
                return preset
    for preset in reversed(candidates):
        if preset.settings.display.identity() == settings.display.identity():
            return preset
    return None


def with_display(settings: CvvdpSettings, **changes) -> CvvdpSettings:
    return replace(settings, display=replace(settings.display, **changes).validated())


def default_settings(user_presets: list[dict], default_name: str) -> CvvdpSettings:
    """The settings a newly added video starts with: the preset chosen in
    Settings, or the built-in default if that preset is gone or unset."""
    preset = preset_named(default_name, user_presets) if default_name else None
    # A preset is a display; scaling to fill it is each video's own choice
    # and starts off, the official default.
    return CvvdpSettings((preset or DEFAULT_PRESET).settings.display)


def with_user_preset(user_presets: list[dict], name: str, settings: CvvdpSettings) -> list[dict]:
    """The saved-preset list with `name` added, or replaced if it exists.
    Built-in names are refused so a built-in preset always means what it says."""
    name = name.strip()
    if not name:
        raise ValueError("A preset needs a name.")
    if any(preset.name == name for preset in BUILTIN_PRESETS):
        raise ValueError(f"\"{name}\" is a built-in preset; choose another name.")
    settings.display.validated()
    kept = [data for data in user_presets if data.get("name") != name]
    return [*kept, {"name": name, "settings": CvvdpSettings(settings.display).to_dict()}]


def without_user_preset(user_presets: list[dict], name: str) -> list[dict]:
    return [data for data in user_presets if data.get("name") != name]
