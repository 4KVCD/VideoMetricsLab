from pathlib import Path

import numpy as np

from vmaf_app.core.models import (
    CropBox,
    FrameScore,
    FrameScores,
    ResampleTarget,
    VideoInfo,
    VmafRunResult,
    synthetic_resample_distorted_path,
)
from vmaf_app.core.run_io import export_csv, load_run, save_run, unique_output_path


def _sample_result() -> VmafRunResult:
    src_info = VideoInfo(
        path=Path("source.mov"), width=3840, height=2160, fps=24000 / 1001,
        duration=10.0, nb_frames=240, codec_name="prores",
    )
    dist_info = VideoInfo(
        path=Path("distorted.mp4"), width=1920, height=1080, fps=24000 / 1001,
        duration=10.0, nb_frames=240, codec_name="h264",
    )
    frames = [FrameScore(frame=i, time=i / dist_info.fps, vmaf=90 + (i % 10)) for i in range(240)]
    return VmafRunResult(
        source=src_info.path, distorted=dist_info.path, frames=frames, fps=dist_info.fps,
        model="version=vmaf_v0.6.1",
        source_crop=CropBox(w=3840, h=1634, x=0, y=263),
        distorted_crop=CropBox(w=1920, h=817, x=0, y=131),
        source_info=src_info, distorted_info=dist_info,
    )


def test_save_and_load_round_trips_frames(tmp_path):
    result = _sample_result()
    result.distorted_info.color_range = "tv"
    result.distorted_info.color_space = "bt2020nc"
    result.distorted_info.color_transfer = "smpte2084"
    result.distorted_info.color_primaries = "bt2020"
    result.scale_algorithm = "lanczos"
    result.resample_target = ResampleTarget(width=1920, label="1080p")
    result.compared_frame_count = 321
    out_path = tmp_path / "run.vmafrun.json"
    save_run(result, out_path, label="my-encode")

    loaded, label = load_run(out_path)
    assert label == "my-encode"
    assert len(loaded.frames) == len(result.frames)
    assert loaded.frames[10].vmaf == result.frames[10].vmaf
    assert loaded.source_crop == result.source_crop
    assert loaded.distorted_crop == result.distorted_crop
    assert loaded.source_info.width == result.source_info.width
    assert loaded.model == result.model
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
    out_path = tmp_path / "run.vmafrun.json"
    save_run(result, out_path, label="exact")

    loaded, _ = load_run(out_path)
    for metric in ("vmaf", "psnr", "ssim", "xpsnr"):
        np.testing.assert_array_equal(
            loaded.frames.values(metric), result.frames.values(metric), err_msg=metric,
        )
    # `time` is the one column deliberately rounded (to 6dp / 1us) to keep the
    # file small; that is far finer than anything displayed or searched on.
    np.testing.assert_allclose(loaded.frames.time, result.frames.time, atol=1e-6)


def test_export_csv_writes_header_and_all_rows(tmp_path):
    result = _sample_result()
    out_path = tmp_path / "run.csv"
    export_csv(result, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "frame,time_s,vmaf,psnr,ssim,xpsnr,vmaf_neg"
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
    assert zero_row[3] == "0.0" and zero_row[4] == "0.0" and zero_row[5] == "0.0"
    assert rows[2].split(",")[3] == ""  # None still exports as blank


def test_xpsnr_round_trips(tmp_path):
    result = _sample_result()
    xpsnr = np.full(len(result.frames), np.nan, dtype=np.float32)
    xpsnr[0] = 42.5
    result.frames = result.frames.with_values("xpsnr", xpsnr)
    out_path = tmp_path / "run.vmafrun.json"
    save_run(result, out_path, label="x")

    loaded, _ = load_run(out_path)
    assert loaded.frames[0].xpsnr == 42.5


def test_infinite_xpsnr_round_trips_as_standards_compliant_json(tmp_path):
    import json

    result = _sample_result()
    result.frames = result.frames.with_values(
        "xpsnr", np.full(len(result.frames), np.inf, dtype=np.float32)
    )
    out_path = tmp_path / "perfect.vmafrun.json"
    save_run(result, out_path, label="perfect")

    # Reject JavaScript-style bare Infinity constants: the portable file
    # must remain valid JSON even though the underlying metric is infinite.
    json.loads(
        out_path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    loaded, _ = load_run(out_path)
    assert np.isposinf(loaded.frames.xpsnr).all()


def test_loading_a_run_saved_before_xpsnr_existed_does_not_raise(tmp_path):
    # Simulates an old save file whose frame tuples are 5 elements (no xpsnr).
    import json
    result = _sample_result()
    out_path = tmp_path / "old_run.vmafrun.json"
    save_run(result, out_path, label="old")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    data["frames"] = [fr[:5] for fr in data["frames"]]
    out_path.write_text(json.dumps(data), encoding="utf-8")

    loaded, _ = load_run(out_path)
    assert loaded.frames[0].xpsnr is None


def test_old_resolution_run_recovers_recipe_from_its_synthetic_name(tmp_path):
    import json

    result = _sample_result()
    target = ResampleTarget(width=1920, label="1080p")
    result.distorted = synthetic_resample_distorted_path(result.source, target)
    result.distorted_info = result.source_info
    out_path = tmp_path / "old_resolution.vmafrun.json"
    save_run(result, out_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    data.pop("resample_target")
    data.pop("scale_algorithm")
    data.pop("compared_frame_count")
    out_path.write_text(json.dumps(data), encoding="utf-8")

    loaded, _ = load_run(out_path)

    assert loaded.resample_target == target
    assert loaded.scale_algorithm == "bicubic"
    assert loaded.compared_frame_count == int(loaded.frames.frame[-1]) + 1


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
