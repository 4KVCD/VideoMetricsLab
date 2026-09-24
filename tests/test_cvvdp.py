"""CVVDP settings, presets, request identity, and the per-second timeline's storage."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from vmaf_app.core import cvvdp
from vmaf_app.core.cvvdp import (
    BUILTIN_PRESETS,
    DEFAULT_PRESET,
    CvvdpDisplay,
    CvvdpSettings,
    default_settings,
    matching_preset,
    presets,
    with_display,
    with_user_preset,
    without_user_preset,
    write_vship_config,
)
from vmaf_app.core.ffmpeg_request import (
    analysis_request_from_vmaf_options,
    comparison_recipe_from_vmaf_options,
    metric_request_specs,
)
from vmaf_app.core.metric_cache import load_metric, metric_path, recipe_directory, store_metric
from vmaf_app.core.metric_results import MetricProvenance, MetricResultSet, SequenceMetricResult
from vmaf_app.core.metrics import METRIC_BY_KEY, SEQUENCE_METRICS, MetricDirection, MetricKind
from vmaf_app.core.models import FrameScores, VmafOptions
from vmaf_app.core.settings import Settings

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


def test_cvvdp_is_a_higher_is_better_sequence_metric_on_a_0_to_10_axis():
    definition = METRIC_BY_KEY["cvvdp"]
    assert definition.kind is MetricKind.SEQUENCE and definition in SEQUENCE_METRICS
    assert definition.direction is MetricDirection.HIGHER_IS_BETTER
    assert definition.fixed_y_max == 10.0 and definition.backend_id == "perceptual"


def test_default_preset_is_the_official_4k_office_monitor_for_every_video():
    display = DEFAULT_PRESET.settings.display
    assert (display.width, display.height, display.diagonal_inches) == (3840, 2160, 30)
    assert (display.peak_luminance, display.ambient_lux, display.hdr) == (200, 250, False)
    assert DEFAULT_PRESET.settings.resize_to_display is False
    assert default_settings([], "") == DEFAULT_PRESET.settings
    # A default naming a preset that was deleted falls back rather than failing.
    assert default_settings([], "gone") == DEFAULT_PRESET.settings


def test_builtin_presets_are_valid_and_distinct():
    names = [preset.name for preset in BUILTIN_PRESETS]
    assert len(set(names)) == len(names)
    identities = {json.dumps(preset.settings.spec_parameters()) for preset in BUILTIN_PRESETS}
    assert len(identities) == len(BUILTIN_PRESETS)
    for preset in BUILTIN_PRESETS:
        preset.settings.display.validated()


def test_distance_in_heights_matches_the_official_4k_geometry():
    # standard_4k: a 30" 16:9 screen is 0.3736 m tall; 0.7472 m is two heights.
    assert DEFAULT_PRESET.settings.display.distance_in_heights == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("change", [
    {"width": 8}, {"diagonal_inches": 0}, {"viewing_distance_m": -1}, {"peak_luminance": 0},
    {"contrast": 0}, {"ambient_lux": -1}, {"reflectivity": 1.0}, {"exposure": 0},
])
def test_invalid_displays_are_refused(change):
    with pytest.raises(ValueError):
        with_display(DEFAULT_PRESET.settings, **change)


def test_saving_a_user_preset_adds_or_replaces_it_and_it_can_become_the_default():
    tuned = with_display(DEFAULT_PRESET.settings, peak_luminance=350, ambient_lux=50)
    saved = with_user_preset([], "My monitor", tuned)
    assert default_settings(saved, "My monitor") == tuned
    assert [p.name for p in presets(saved)][-1] == "My monitor"
    brighter = with_display(tuned, peak_luminance=400)
    saved = with_user_preset(saved, "My monitor", brighter)
    assert len(saved) == 1 and default_settings(saved, "My monitor") == brighter
    assert without_user_preset(saved, "My monitor") == []


def test_user_presets_cannot_take_a_builtin_name_or_be_blank():
    with pytest.raises(ValueError):
        with_user_preset([], BUILTIN_PRESETS[0].name, DEFAULT_PRESET.settings)
    with pytest.raises(ValueError):
        with_user_preset([], "  ", DEFAULT_PRESET.settings)


def test_a_malformed_saved_preset_is_skipped():
    broken = [{"name": "bad"}, {"name": "worse", "settings": {"display": {"width": 0}}},
              {"name": "ok", "settings": CvvdpSettings().to_dict()}]
    assert [p.name for p in presets(broken)][len(BUILTIN_PRESETS):] == ["ok"]


def test_matching_preset_prefers_the_users_name_for_the_same_settings():
    saved = with_user_preset([], "Office", DEFAULT_PRESET.settings)
    assert matching_preset(DEFAULT_PRESET.settings, saved).name == "Office"
    assert matching_preset(DEFAULT_PRESET.settings, []).name == DEFAULT_PRESET.name
    assert matching_preset(with_display(DEFAULT_PRESET.settings, exposure=2.0), saved) is None


def test_settings_round_trip_through_a_dict_and_ignore_unknown_fields():
    settings = CvvdpSettings(CvvdpDisplay(1920, 1080, 24, 0.6, 200, 1000, 250, 0.005, 1.0, False), True)
    data = settings.to_dict()
    data["display"]["from_a_future_version"] = 1
    assert CvvdpSettings.from_dict(data) == settings


def test_vship_config_file_sets_every_display_property(tmp_path):
    display = BUILTIN_PRESETS[2].settings.display
    model = json.loads(write_vship_config(display, tmp_path / "d.json").read_text())[cvvdp.VSHIP_MODEL_KEY]
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


def test_cvvdp_request_carries_the_settings_through_analysis_requests():
    settings = with_display(DEFAULT_PRESET.settings, peak_luminance=600)
    request = analysis_request_from_vmaf_options(VmafOptions(), ("vmaf", "cvvdp"), cvvdp=settings)
    spec = next(spec for spec in request.metrics if spec.key == "cvvdp")
    assert dict(spec.parameters)["display"]["peak_luminance"] == 600


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


def test_sequence_result_without_a_timeline_still_round_trips(tmp_path):
    directory = _cache_directory(tmp_path)
    (spec,) = metric_request_specs(VmafOptions(), ("cvvdp",))
    store_metric(directory, SequenceMetricResult("cvvdp", 7.0, PROVENANCE), spec)
    loaded = load_metric(directory, spec)
    assert loaded.score == 7.0 and not loaded.has_timeline


def test_timeline_round_trips_through_saved_results(tmp_path):
    from tests.test_run_io import _sample_result
    from vmaf_app.core.run_io import load_run, save_run

    result = _sample_result()
    result.frames = FrameScores.empty()
    result.metric_results = MetricResultSet([_timeline_result()])
    save_run(result, tmp_path / "r.metrics.json")
    loaded, _ = load_run(tmp_path / "r.metrics.json")
    sequence = loaded.sequence_metric("cvvdp")
    assert sequence.score == pytest.approx(9.3141) and sequence.has_timeline
    np.testing.assert_array_equal(sequence.frame, [0, 24, 48])
    assert np.isnan(sequence.values[2])


def test_mismatched_timeline_lengths_are_refused():
    with pytest.raises(ValueError):
        SequenceMetricResult("cvvdp", 9.0, PROVENANCE, frame=[0, 1], time=[0.0], values=[9.0])


def test_settings_file_keeps_cvvdp_presets_and_default(tmp_path, monkeypatch):
    monkeypatch.setattr("vmaf_app.core.settings.settings_file", lambda: tmp_path / "settings.json")
    settings = Settings()
    assert settings.default_compute_cvvdp is False and settings.cvvdp_default_preset == ""
    settings.cvvdp_presets = with_user_preset([], "Mine", DEFAULT_PRESET.settings)
    settings.cvvdp_default_preset = "Mine"
    settings.default_compute_cvvdp = True
    assert settings.save() is None
    loaded = Settings.load()
    assert asdict(loaded)["cvvdp_presets"] == settings.cvvdp_presets
    assert default_settings(loaded.cvvdp_presets, loaded.cvvdp_default_preset) == DEFAULT_PRESET.settings
    assert loaded.default_compute_cvvdp is True
