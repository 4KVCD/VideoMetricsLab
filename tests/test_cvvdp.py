"""CVVDP settings, presets, request identity, and the per-second timeline's storage."""
from __future__ import annotations

import json

import numpy as np
import pytest

from vmaf_app.core import cvvdp
from vmaf_app.core.cvvdp import (
    BUILTIN_PRESETS,
    DEFAULT_PRESET,
    CvvdpDisplay,
    CvvdpSettings,
    default_settings,
    with_display,
)
from vmaf_app.core.ffmpeg_request import (
    comparison_recipe_from_vmaf_options,
    metric_request_specs,
)
from vmaf_app.core.metric_cache import load_metric, metric_path, recipe_directory, store_metric
from vmaf_app.core.metric_results import MetricProvenance, SequenceMetricResult
from vmaf_app.core.models import VmafOptions

PROVENANCE = MetricProvenance("vship/cvvdp", "4.1", "gpu", "cvvdp-vship-gpu-v1", {"display": "x"})


def _timeline_result(score=9.3141):
    return SequenceMetricResult(
        "cvvdp", score, PROVENANCE,
        frame=[0, 24, 48], time=[0.0, 1.001, 2.002], values=[9.5, 8.25, np.nan],
    )


def _cache_directory(tmp_path):
    source, test = tmp_path / "a.mkv", tmp_path / "b.mkv"
    source.write_bytes(b"a")
    test.write_bytes(b"b")
    return recipe_directory(tmp_path, source, test, comparison_recipe_from_vmaf_options(VmafOptions()))


def test_default_preset_is_the_official_4k_office_monitor_for_every_video():
    display = DEFAULT_PRESET.settings.display
    assert (display.width, display.height, display.diagonal_inches) == (3840, 2160, 30)
    assert (display.peak_luminance, display.ambient_lux, display.hdr) == (200, 250, False)
    assert DEFAULT_PRESET.settings.resize_to_display is False
    assert default_settings([], "") == DEFAULT_PRESET.settings
    # A default naming a preset that was deleted falls back rather than failing.
    assert default_settings([], "gone") == DEFAULT_PRESET.settings


def test_distance_in_heights_matches_the_official_4k_geometry():
    # standard_4k: a 30" 16:9 screen is 0.3736 m tall; 0.7472 m is two heights.
    assert DEFAULT_PRESET.settings.display.distance_in_heights == pytest.approx(2.0, abs=1e-3)


def test_invalid_displays_are_refused(subtests):
    def check(change):
        with pytest.raises(ValueError):
            with_display(DEFAULT_PRESET.settings, **change)

    for change in [
        {"width": 8}, {"diagonal_inches": 0}, {"viewing_distance_m": -1}, {"peak_luminance": 0},
        {"contrast": 0}, {"ambient_lux": -1}, {"reflectivity": 1.0}, {"exposure": 0},
    ]:
        with subtests.test(change=change):
            check(change)


def test_settings_round_trip_through_a_dict_and_ignore_unknown_fields():
    settings = CvvdpSettings(CvvdpDisplay(1920, 1080, 24, 0.6, 200, 1000, 250, 0.005, 1.0, False), True)
    data = settings.to_dict()
    data["display"]["from_a_future_version"] = 1
    assert CvvdpSettings.from_dict(data) == settings


def test_vship_display_json_sets_every_display_property():
    display = BUILTIN_PRESETS[2].settings.display
    model = json.loads(cvvdp.vship_display_json(display))[cvvdp.VSHIP_MODEL_KEY]
    assert model["colorspace"] == "HDR" and model["resolution"] == [3840, 2160]
    assert model["max_luminance"] == 1500 and model["contrast"] == 1_000_000 and model["E_ambient"] == 10
    assert {"viewing_distance_meters", "diagonal_size_inches", "k_refl", "exposure"} <= model.keys()


def test_cvvdp_request_is_full_coverage_and_its_identity_follows_the_display():
    (spec,) = metric_request_specs(VmafOptions(), ("cvvdp",))
    assert spec.coverage.mode == "full" and spec.coverage.step == 1
    assert spec.implementation_compatibility_id == "cvvdp-vship-gpu-v1"
    # Subsampling VMAF does not subsample CVVDP (it models motion over time).
    (sampled,) = metric_request_specs(VmafOptions(n_subsample=5), ("cvvdp",))
    assert sampled.coverage.step == 1
    (changed,) = metric_request_specs(VmafOptions(), ("cvvdp",), with_display(DEFAULT_PRESET.settings, ambient_lux=10))
    (resize,) = metric_request_specs(VmafOptions(), ("cvvdp",), CvvdpSettings(DEFAULT_PRESET.settings.display, True))
    identities = {repr(s.identity_dict()) for s in (spec, changed, resize)}
    assert len(identities) == 3
    # A value read back with float noise is the same display.
    noisy = with_display(DEFAULT_PRESET.settings, viewing_distance_m=0.74720000001)
    (same,) = metric_request_specs(VmafOptions(), ("cvvdp",), noisy)
    assert same.identity_dict() == spec.identity_dict()


def test_timeline_round_trips_through_the_metric_cache(tmp_path):
    directory = _cache_directory(tmp_path)
    (spec,) = metric_request_specs(VmafOptions(), ("cvvdp",))
    store_metric(directory, _timeline_result(), spec)
    loaded = load_metric(directory, spec)
    assert isinstance(loaded, SequenceMetricResult) and loaded.score == pytest.approx(9.3141)
    np.testing.assert_array_equal(loaded.frame, [0, 24, 48])
    np.testing.assert_array_equal(loaded.time, [0.0, 1.001, 2.002])
    assert loaded.values[:2].tolist() == [9.5, 8.25] and np.isnan(loaded.values[2])
    assert metric_path(directory, spec).exists()
