"""LockedPresentation: GStreamer's GPU frames presented, zoomed by cropping."""
import pytest


def test_a_cropped_frame_carries_its_crop_and_the_original_stays_whole():
    """A zoomed GPU frame is shown by a GstVideoCropMeta on a copy of its
    buffer, put there through GStreamer's C API (the Python one has no way):
    the meta must hold the rectangle, and the frame kept for showing again
    must not get it."""
    pytest.importorskip("gi")
    from vmaf_app.core.gstreamer_playback import GStreamerPlaybackError, _load_gstreamer
    from vmaf_app.core.locked_presentation import _CropMeta, cropped

    try:
        gst, video = _load_gstreamer()
    except GStreamerPlaybackError:
        pytest.skip("GStreamer is not installed")
    caps = gst.Caps.from_string("video/x-raw,format=RGBA,width=8,height=6")
    sample = gst.Sample.new(gst.Buffer.new_allocate(None, 8 * 6 * 4, None), caps, None, None)
    shown = cropped(sample, (1, 2, 4, 3))
    api = video.VideoCropMeta.get_info().api
    meta = shown.get_buffer().get_meta(api)
    assert meta is not None and sample.get_buffer().get_meta(api) is None
    from vmaf_app.core.d3d11_tonemap import boxed_pointer

    crop = _CropMeta.from_address(boxed_pointer(meta))
    assert (crop.x, crop.y, crop.width, crop.height) == (1, 2, 4, 3)
    assert shown.get_caps().is_equal(caps)
