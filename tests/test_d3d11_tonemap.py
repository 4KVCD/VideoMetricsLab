from types import SimpleNamespace

import pytest

from vmaf_app.core import d3d11_tonemap
from vmaf_app.core.d3d11_tonemap import D3D11ToneMapper


@pytest.fixture(autouse=True)
def _fake_frame_pointer(monkeypatch, request):
    """The fake buffers' memory is a plain object: its pointer is made up
    (boxed_pointer has tests of its own below)."""
    if "pointer" not in request.node.name and "readings" not in request.node.name:
        monkeypatch.setattr(d3d11_tonemap, "boxed_pointer", lambda _memory: 789)


def _mapper(result=0):
    calls = []
    mapper = object.__new__(D3D11ToneMapper)
    mapper.device = SimpleNamespace(lock=lambda: calls.append("lock"), unlock=lambda: calls.append("unlock"))
    mapper.kind, mapper.handle = 1, None
    mapper.gst = SimpleNamespace(gst_is_d3d11_memory=lambda p: True,
                                 gst_d3d11_memory_get_resource_handle=lambda p: 123)
    mapper.lib = SimpleNamespace(vmaf_tonemap_create=lambda p, k: 456,
                                 vmaf_tonemap_render=lambda h, p: result,
                                 vmaf_tonemap_destroy=lambda h: calls.append("destroy"))
    return mapper, calls


def _buffer(count=1):
    return SimpleNamespace(n_memory=lambda: count, peek_memory=lambda i: object())


def test_shader_unlocks_device_on_native_failure():
    mapper, calls = _mapper(-1)
    with pytest.raises(RuntimeError, match="ffffffff"):
        mapper.render(_buffer())
    assert calls == ["lock", "unlock"]


def test_shader_rejects_cpu_memory_before_native_render():
    mapper, calls = _mapper()
    mapper.gst.gst_is_d3d11_memory = lambda p: False
    with pytest.raises(RuntimeError, match="CPU memory"):
        mapper.render(_buffer())
    assert calls == []


def test_shader_rejects_multiplane_buffers():
    mapper, calls = _mapper()
    with pytest.raises(RuntimeError, match="one private"):
        mapper.render(_buffer(2))
    assert calls == []


def test_shader_closes_idempotently_after_processing():
    mapper, calls = _mapper()
    mapper.render(_buffer())
    assert mapper.handle == 456
    mapper.close()
    mapper.close()
    assert calls == ["lock", "unlock", "destroy"]
    assert mapper.handle is None


def test_a_gstreamer_memorys_pointer_is_the_one_gstreamer_gives():
    """hash(memory) was taken as the native pointer: a PyGObject detail. It
    is now read two ways, which must agree with GStreamer's own."""
    import ctypes

    pytest.importorskip("gi")
    from vmaf_app.core.d3d11_tonemap import boxed_pointer
    from vmaf_app.core.gstreamer_playback import GStreamerPlaybackError, _load_gstreamer

    try:
        gst, _ = _load_gstreamer()
    except GStreamerPlaybackError:
        pytest.skip("GStreamer is not installed")
    buffer = gst.Buffer.new_allocate(None, 64, None)
    memory = buffer.peek_memory(0)
    library = ctypes.CDLL("gstreamer-1.0-0.dll")
    library.gst_buffer_peek_memory.restype = ctypes.c_void_p
    library.gst_buffer_peek_memory.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    assert boxed_pointer(memory) == library.gst_buffer_peek_memory(boxed_pointer(buffer), 0)


def test_a_wrapper_whose_two_readings_disagree_is_refused():
    from vmaf_app.core.d3d11_tonemap import boxed_pointer

    with pytest.raises(RuntimeError, match="native pointer"):
        boxed_pointer(object())
