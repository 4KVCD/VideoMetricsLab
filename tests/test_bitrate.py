import io
from pathlib import Path

import numpy as np
import pytest

from vmaf_app.core import bitrate as bitrate_module
from vmaf_app.core.bitrate import (
    BitrateData,
    analyze_video_bitrate,
    bitrate_summary,
    frame_plot,
    gop_plot,
    second_plot,
)
from vmaf_app.core.models import VideoInfo


def _data(start=0.0):
    return BitrateData(
        path=Path("video.mkv"),
        times=np.array([start, start + 0.5, start + 1.0]),
        durations=np.array([0.5, 0.5, 0.5]),
        sizes=np.array([1000, 1000, 2000]),
        keyframes=np.array([True, False, True]),
    )


def test_frame_view_is_one_encoded_size_per_video_packet():
    plot = frame_plot(_data())

    np.testing.assert_array_equal(plot.times, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(plot.values, [8.0, 8.0, 16.0])
    assert plot.axis_label == "Frame size (kbit)"


def test_second_view_uses_video_bytes_in_each_wall_clock_second():
    plot = second_plot(_data())

    # Step points duplicate each interval's beginning/end.
    np.testing.assert_array_equal(plot.times, [0.0, 1.0, 1.0, 2.0])
    np.testing.assert_array_equal(plot.values, [16.0, 16.0, 32.0, 32.0])


def test_second_view_splits_a_packet_that_crosses_a_boundary():
    data = BitrateData(
        path=Path("crossing.mkv"),
        times=np.array([0.75]), durations=np.array([0.5]),
        sizes=np.array([1000]), keyframes=np.array([True]),
    )

    plot = second_plot(data, adjust_start=False)

    # 500 bytes land on either side; each edge interval contains 0.25 s of
    # media, so both normalize to 16 kb/s rather than halving the rate.
    np.testing.assert_allclose(plot.values, [16.0] * 4)


def test_gop_view_groups_from_one_keyframe_to_the_next():
    plot = gop_plot(_data())

    np.testing.assert_array_equal(plot.times, [0.0, 1.0, 1.0, 1.5])
    np.testing.assert_array_equal(plot.values, [16.0, 16.0, 32.0, 32.0])


def test_summary_is_video_packet_rate_not_whole_file_size():
    summary = bitrate_summary(_data())

    assert summary.average_kbps == pytest.approx(4000 * 8 / 1.5 / 1000)
    assert summary.minimum_kbps == 16.0
    assert summary.maximum_kbps == 32.0


def test_adjust_start_time_aligns_the_first_interval_to_zero():
    data = _data(start=12.5)

    adjusted = second_plot(data, adjust_start=True)
    raw = second_plot(data, adjust_start=False)

    assert adjusted.times[0] == 0.0
    assert raw.times[0] == 12.0


def test_packet_scan_selects_only_video_and_sorts_decode_order_by_pts(
    tmp_path, monkeypatch
):
    path = tmp_path / "b-frames.mkv"
    path.write_bytes(b"video")
    info = VideoInfo(
        path=path, width=1920, height=1080, fps=30.0,
        duration=0.1, nb_frames=3, codec_name="h264",
    )

    class FakeProcess:
        pid = 321
        stdout = io.StringIO(
            "pts_time=0.000|duration_time=0.033|size=1000|pos=0|flags=K_\n"
            "pts_time=0.066|duration_time=0.033|size=300|pos=200|flags=__\n"
            "pts_time=0.033|duration_time=0.033|size=200|pos=100|flags=__\n"
        )
        stderr = io.StringIO("")

        def wait(self):
            return 0

    captured = []
    monkeypatch.setattr(bitrate_module, "ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(
        bitrate_module.proc_util,
        "popen",
        lambda command, **_kwargs: captured.append(command) or FakeProcess(),
    )

    data = analyze_video_bitrate(info)

    assert captured[0][captured[0].index("-select_streams") + 1] == "v:0"
    np.testing.assert_allclose(data.times, [0.0, 0.033, 0.066])
    np.testing.assert_array_equal(data.sizes, [1000, 200, 300])
    np.testing.assert_array_equal(data.keyframes, [True, False, False])
