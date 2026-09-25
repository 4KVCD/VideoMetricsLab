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
    result.model_choice = "__builtin:vmaf_v1_3d0h"
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
    assert lines[0] == "frame,time_s,vmaf,vmaf_neg,psnr,ssim,xpsnr,ssimulacra2,butteraugli"
    assert len(lines) == 1 + len(result.frames)


def test_csv_export_keeps_genuine_zero_metric_values(tmp_path):
    # VMAF and SSIM really do bottom out at 0.0 for badly degraded frames.
    # Exporting with `value or ""` turned those into empty cells, making a
    # real score indistinguishable from "this metric wasn't computed".
    result = _sample_result()
    # frame 0 scores a legitimate 0.0 on every optional metric; frame 1 has
    # no PSNR at all (NaN = "not computed for this frame")
    result.frames = FrameScores(
        frame=np.array([0, 1], dtype=np.int32),
        time=np.array([0.0, 1 / 24], dtype=np.float64),
        vmaf=np.array([0.0, 50.0], dtype=np.float32),
        psnr=np.array([0.0, np.nan], dtype=np.float32),
        ssim=np.array([0.0, 0.5], dtype=np.float32),
        xpsnr=np.array([0.0, 30.0], dtype=np.float32),
    )
    out_path = tmp_path / "run.csv"
    export_csv(result, out_path)

    rows = out_path.read_text(encoding="utf-8").splitlines()
    zero_row = rows[1].split(",")
    assert zero_row[4:7] == ["0.0", "0.0", "0.0"]
    assert rows[2].split(",")[4] == ""  # None still exports as blank


def test_xpsnr_round_trips(tmp_path):
    result = _sample_result()
    xpsnr = np.full(len(result.frames), np.nan, dtype=np.float32)
    xpsnr[0] = 42.5
    result.frames = result.frames.with_values("xpsnr", xpsnr)
    out_path = tmp_path / "run.metrics.json"
    save_run(result, out_path, label="x")

    loaded, _ = load_run(out_path)
    assert loaded.frames[0].xpsnr == 42.5


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


def test_load_rejects_mismatched_frame_metric_arrays(tmp_path):
    import json

    result = _sample_result()
    out_path = tmp_path / "malformed.metrics.json"
    save_run(result, out_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    data["metric_results"][0]["values"] = data["metric_results"][0]["values"][:-1]
    out_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="equal lengths"):
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


def test_portable_file_is_strict_json_with_generic_special_values(tmp_path):
    import json

    provenance = MetricProvenance("test", "1", "cpu", "test-v1")
    result = _sample_result()
    result.frames = FrameScores.empty()
    result.metric_results = MetricResultSet([
        FrameMetricResult("future", [0], [0.0], [np.inf], provenance),
        SequenceMetricResult("sequence", float("nan"), provenance),
    ])
    out_path = tmp_path / "special.metrics.json"
    save_run(result, out_path)

    json.loads(
        out_path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    loaded, _ = load_run(out_path)
    assert np.isposinf(loaded.frame_metric("future").values[0])
    assert np.isnan(loaded.sequence_metric("sequence").score)


# ------------------------------------------------------ unique output paths

def test_a_second_run_with_the_same_label_gets_its_own_file(tmp_path):
    # Two encodes named movie.mp4 from different folders, or one file
    # compared twice under different options, both reduce to "movie".
    reserved: set[Path] = set()
    first = unique_output_path(tmp_path, "movie", ".csv", reserved)
    second = unique_output_path(tmp_path, "movie", ".csv", reserved)

    assert first.name == "movie.csv"
    assert second.name == "movie_2.csv"
    assert first != second


def test_reservations_hold_before_anything_is_written(tmp_path):
    # Within one export loop the earlier file may not exist on disk yet, so
    # checking only Path.exists() would hand out the same name twice.
    reserved: set[Path] = set()
    names = [unique_output_path(tmp_path, "movie", ".csv", reserved).name for _ in range(4)]

    assert names == ["movie.csv", "movie_2.csv", "movie_3.csv", "movie_4.csv"]
    assert not any((tmp_path / n).exists() for n in names)


def test_a_file_already_on_disk_is_never_overwritten(tmp_path):
    (tmp_path / "movie.csv").write_text("existing", encoding="utf-8")

    path = unique_output_path(tmp_path, "movie", ".csv")

    assert path.name == "movie_2.csv"
    assert (tmp_path / "movie.csv").read_text(encoding="utf-8") == "existing"


def test_characters_a_filename_cannot_carry_are_replaced(tmp_path):
    path = unique_output_path(tmp_path, "a/b:c*d", ".csv")
    assert path.name == "a_b_c_d.csv"


def test_a_label_with_nothing_usable_still_produces_a_name(tmp_path):
    assert unique_output_path(tmp_path, "///", ".csv").name == "___.csv"
    assert unique_output_path(tmp_path, "", ".csv").name == "run.csv"


def test_two_labels_that_sanitise_to_the_same_stem_do_not_collide(tmp_path):
    # "a b" and "a/b" both become "a_b" -- the collision appears only after
    # sanitising, so deduplicating the labels beforehand would miss it.
    reserved: set[Path] = set()
    first = unique_output_path(tmp_path, "a b", ".csv", reserved)
    second = unique_output_path(tmp_path, "a/b", ".csv", reserved)

    assert (first.name, second.name) == ("a_b.csv", "a_b_2.csv")



def test_csv_export_includes_ssimulacra2_and_butteraugli_on_their_own_frames(tmp_path):
    """The export had the five FFmpeg metrics hard-coded, so SSIMULACRA2 and
    Butteraugli were never written. Here SSIMULACRA2 covers every second
    frame: its cells are blank in between, never borrowed."""
    from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet

    result = _sample_result()
    n = len(result.frames)
    provenance = MetricProvenance("Vship/ssimulacra2", "5", "gpu", "ssimulacra2-vship-gpu-v1")
    every_second = np.arange(0, n, 2, dtype=np.int32)
    result.merge_metric_results(MetricResultSet([
        FrameMetricResult("ssimulacra2", every_second, every_second / 24.0,
                          np.full(len(every_second), 71.5, dtype=np.float32), provenance),
        FrameMetricResult("butteraugli", result.frames.frame, result.frames.time,
                          np.full(n, 0.0, dtype=np.float32), provenance),
    ]))
    out_path = tmp_path / "run.csv"
    export_csv(result, out_path)

    rows = [line.split(",") for line in out_path.read_text(encoding="utf-8").splitlines()]
    header = rows[0]
    assert header[-2:] == ["ssimulacra2", "butteraugli"]
    assert len(rows) == 1 + n
    s2, ba = header.index("ssimulacra2"), header.index("butteraugli")
    assert rows[1][s2] == "71.5" and rows[2][s2] == ""
    assert all(row[ba] == "0.0" for row in rows[1:])  # a genuine 0 is not blank


def test_csv_export_of_a_perceptual_only_result_has_its_values(tmp_path):
    from vmaf_app.core.metric_results import FrameMetricResult, MetricProvenance, MetricResultSet
    from vmaf_app.core.models import FrameScores

    result = _sample_result()
    result.frames = FrameScores.empty()
    result.metric_results = MetricResultSet([FrameMetricResult(
        "ssimulacra2", np.arange(3, dtype=np.int32), np.arange(3) / 24.0, np.array([80.0, 81.0, 82.0], dtype=np.float32),
        MetricProvenance("ssimulacra2", "0.12", "cpu", "ssimulacra2-libjxl-cpu-v1"),
    )])
    out_path = tmp_path / "run.csv"
    export_csv(result, out_path)
    rows = [line.split(",") for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [row[rows[0].index("ssimulacra2")] for row in rows[1:]] == ["80.0", "81.0", "82.0"]
    assert rows[1][2:7] == ["", "", "", "", ""]  # no FFmpeg metrics in this result


def test_csv_export_writes_cvvdp_per_second_and_for_the_whole_video(tmp_path):
    """CSV export had no CVVDP at all: a CVVDP-only video exported a file
    with just a header, and still reported "Export complete"."""
    from vmaf_app.core.metric_results import MetricProvenance, MetricResultSet, SequenceMetricResult
    from vmaf_app.core.models import FrameScores

    cvvdp = SequenceMetricResult(
        "cvvdp", 9.61, MetricProvenance("Vship/cvvdp", "5", "gpu", "cvvdp-vship-gpu-v1"),
        frame=[0, 24, 48], time=[0.0, 1.001, 2.002], values=[10.0, 9.5, 9.25],
    )
    result = _sample_result()
    result.merge_metric_results(MetricResultSet([cvvdp]))
    out_path = tmp_path / "with_vmaf.csv"
    export_csv(result, out_path)
    rows = [line.split(",") for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0][-2:] == ["cvvdp_second_jod", "cvvdp_video_jod"]
    second, video = rows[0].index("cvvdp_second_jod"), rows[0].index("cvvdp_video_jod")
    by_frame = {row[0]: row for row in rows[1:]}
    assert (by_frame["24"][second], by_frame["24"][video]) == ("9.5", "9.61")
    assert by_frame["1"][second] == "" and by_frame["1"][video] == ""
    assert len(rows) == 1 + len(result.frames)  # its seconds start on frames the table has

    alone = _sample_result()
    alone.frames = FrameScores.empty()
    alone.metric_results = MetricResultSet([cvvdp])
    export_csv(alone, tmp_path / "cvvdp_only.csv")
    rows = [line.split(",") for line in (tmp_path / "cvvdp_only.csv").read_text(encoding="utf-8").splitlines()]
    assert [row[0] for row in rows[1:]] == ["0", "24", "48"]
    assert [row[1] for row in rows[1:]] == ["0.000000", "1.001000", "2.002000"]
    assert [row[-2] for row in rows[1:]] == ["10.0", "9.5", "9.25"]

