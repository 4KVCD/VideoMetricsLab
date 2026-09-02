from pathlib import Path

import numpy as np

from vmaf_app.core.models import CropBox, FrameScore, FrameScores, VideoInfo, VmafRunResult
from vmaf_app.core.run_io import export_csv, load_run, save_run


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


def test_export_csv_writes_header_and_all_rows(tmp_path):
    result = _sample_result()
    out_path = tmp_path / "run.csv"
    export_csv(result, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "frame,time_s,vmaf,psnr,ssim,xpsnr"
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
