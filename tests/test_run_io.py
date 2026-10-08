from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core.metric_results import (
    FrameMetricResult,
    MetricProvenance,
    MetricResultSet,
    SequenceMetricResult,
)
from vmaf_app.core.models import (
    ComparisonResult,
    CropBox,
    FrameScore,
    FrameScores,
    ResampleTarget,
    VideoInfo,
)
from vmaf_app.core.run_io import export_csv, load_run, save_run, unique_output_path


def _sample_result() -> ComparisonResult:
    src_info = VideoInfo(
        path=Path("source.mov"), width=3840, height=2160, fps=24000 / 1001,
        duration=10.0, nb_frames=240, codec_name="prores",
    )
    dist_info = VideoInfo(
        path=Path("distorted.mp4"), width=1920, height=1080, fps=24000 / 1001,
        duration=10.0, nb_frames=240, codec_name="h264",
    )
    frames = [FrameScore(frame=i, time=i / dist_info.fps, vmaf=90 + (i % 10)) for i in range(240)]
    return ComparisonResult(
        source=src_info.path, distorted=dist_info.path, frames=frames, fps=dist_info.fps,
        model="version=vmaf_v0.6.1",
        source_crop=CropBox(w=3840, h=1634, x=0, y=263),
        distorted_crop=CropBox(w=1920, h=817, x=0, y=131),
        source_info=src_info, distorted_info=dist_info,
    )


def test_save_and_load_round_trips_frames(tmp_path):
    result = _sample_result()
    result.model_choice = "version=vmaf_4k_v0.6.1"
    result.distorted_info.color_range = "tv"
    result.distorted_info.color_space = "bt2020nc"
    result.distorted_info.color_transfer = "smpte2084"
    result.distorted_info.color_primaries = "bt2020"
    result.scale_algorithm = "lanczos"
    result.resample_target = ResampleTarget(width=1920, label="1080p")
    result.compared_frame_count = 321
    out_path = tmp_path / "run.metrics.json"
    save_run(result, out_path, label="my-encode")

    loaded, label = load_run(out_path)
    assert label == "my-encode"
    assert len(loaded.frames) == len(result.frames)
    assert loaded.frames[10].vmaf == result.frames[10].vmaf
    assert loaded.source_crop == result.source_crop
    assert loaded.distorted_crop == result.distorted_crop
    assert loaded.source_info.width == result.source_info.width
    assert loaded.model == result.model
    assert loaded.model_choice == result.model_choice
    assert loaded.scale_algorithm == "lanczos"
    assert loaded.resample_target == ResampleTarget(width=1920, label="1080p")
    assert loaded.compared_frame_count == 321
    assert loaded.distorted_info.color_range == "tv"
    assert loaded.distorted_info.color_space == "bt2020nc"
    assert loaded.distorted_info.color_transfer == "smpte2084"
    assert loaded.distorted_info.color_primaries == "bt2020"


def test_save_and_load_preserves_scores_bit_for_bit(tmp_path):
    # Real ffmpeg scores are not round decimals. Rounding them to 6dp on save
    # landed between two float32 values, so a reloaded run no longer equalled
    # the one that produced it -- silently drifting every cached result.
    result = _sample_result()
    rng = np.random.default_rng(0)
    n = len(result.frames)
    result.frames = FrameScores(
        frame=result.frames.frame,
        time=result.frames.time,
        vmaf=rng.uniform(0, 100, n).astype(np.float32),
        psnr=rng.uniform(20, 60, n).astype(np.float32),
        ssim=rng.uniform(0, 1, n).astype(np.float32),
        xpsnr=rng.uniform(20, 60, n).astype(np.float32),
    )
    out_path = tmp_path / "run.metrics.json"
    save_run(result, out_path, label="exact")

    loaded, _ = load_run(out_path)
    for metric in ("vmaf", "psnr", "ssim", "xpsnr"):
        np.testing.assert_array_equal(
            loaded.frames.values(metric), result.frames.values(metric), err_msg=metric,
        )
    # Portable v2 keeps each metric's float64 timeline exactly rather than
    # forcing every metric through one rounded shared frame table.
    np.testing.assert_array_equal(loaded.frames.time, result.frames.time)


def test_export_csv_writes_header_and_all_rows(tmp_path):
    result = _sample_result()
    out_path = tmp_path / "run.csv"
    export_csv(result, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    # The first seven columns never move; VMAF v1 comes after them.
    assert lines[0] == "frame,time_s,vmaf,vmaf_neg,psnr,ssim,xpsnr,vmaf_v1,ssimulacra2,butteraugli"
    assert len(lines) == 1 + len(result.frames)


def test_infinite_xpsnr_round_trips_as_standards_compliant_json(tmp_path):
    import json

    result = _sample_result()
    result.frames = result.frames.with_values(
        "xpsnr", np.full(len(result.frames), np.inf, dtype=np.float32)
    )
    out_path = tmp_path / "perfect.metrics.json"
    save_run(result, out_path, label="perfect")

    # Reject JavaScript-style bare Infinity constants: the portable file
    # must remain valid JSON even though the underlying metric is infinite.
    json.loads(
        out_path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    loaded, _ = load_run(out_path)
    assert np.isposinf(loaded.frames.xpsnr).all()


def test_load_rejects_unknown_format_version(tmp_path):
    import json

    result = _sample_result()
    out_path = tmp_path / "unsupported.metrics.json"
    save_run(result, out_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    data["format_version"] = 999
    out_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported analysis result format version"):
        load_run(out_path)


def test_portable_results_preserve_independent_axes_sequence_metrics_and_provenance(tmp_path):
    provenance = MetricProvenance(
        "reference/cvvdp", "0.1", "cpu", "cvvdp-v1", {"display": "standard"}
    )
    vmaf = FrameMetricResult(
        "vmaf", [0, 2], [0.0, 0.1], [90.0, 91.0], provenance,
    )
    future = FrameMetricResult(
        "future_frame_metric", [0, 5, 10], [0.0, 0.25, 0.5],
        [1.0, np.nan, np.inf], provenance,
    )
    sequence = SequenceMetricResult("cvvdp", float("-inf"), provenance)
    result = _sample_result()
    result.frames = FrameScores.empty()
    result.metric_results = MetricResultSet([vmaf, future, sequence])

    out_path = tmp_path / "generic.metrics.json"
    save_run(result, out_path, label="generic")
    loaded, label = load_run(out_path)

    assert label == "generic"
    np.testing.assert_array_equal(loaded.frame_metric("vmaf").frame, [0, 2])
    np.testing.assert_array_equal(loaded.frame_metric("future_frame_metric").frame, [0, 5, 10])
    assert np.isnan(loaded.frame_metric("future_frame_metric").values[1])
    assert np.isposinf(loaded.frame_metric("future_frame_metric").values[2])
    assert np.isneginf(loaded.sequence_metric("cvvdp").score)
    assert loaded.sequence_metric("cvvdp").provenance == provenance
    # The current UI view keeps the registered metric it can display and does
    # not try to align an unknown independently sampled metric onto that axis.
    assert loaded.frames.vmaf.tolist() == [90.0, 91.0]
    assert not loaded.frames.has("future_frame_metric")


# ------------------------------------------------------ unique output paths


def test_a_file_already_on_disk_is_never_overwritten(tmp_path):
    (tmp_path / "movie.csv").write_text("existing", encoding="utf-8")

    path = unique_output_path(tmp_path, "movie", ".csv")

    assert path.name == "movie_2.csv"
    assert (tmp_path / "movie.csv").read_text(encoding="utf-8") == "existing"
