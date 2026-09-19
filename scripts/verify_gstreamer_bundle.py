"""Prove a packaged GStreamer runtime can do what the app asks of it.

The pruned bundle (scripts/gstreamer_bundle.py) is only as good as the
allow-list behind it, and a mistake there does not announce itself: the app
falls back to FFmpeg playback on any GStreamer error, so a missing demuxer
looks like a slow build rather than a broken one. This makes the mistake
fail the build instead.

Every inspection runs in a child interpreter started with -S: no site
directory, so the development wheels and their .pth bootstrap are out of
reach; a fresh registry in a temporary folder; the plugin path restricted to
the runtime under test. The child reports what it found, and this process
compares the bundle with the development installation.

Checks:
  1. every plugin GStreamer found lives inside the runtime under test;
  2. exactly the allow-listed plugins loaded -- one that failed to load is a
     missing dependency, an unexpected one means the policy leaked;
  3. every element the app creates by name exists (REQUIRED_ELEMENTS);
  4. every codec the development installation could decode, the bundle can
     too (compared by the caps decoders accept), except the losses named in
     ALLOWED_LOSSES;
  5. for each --media file, the app's own video branch (filesrc -> decodebin3
     -> videocrop -> d3d11upload -> d3d11convert) delivers a GPU frame, and the
     soundtrack branch delivers audio, with the decoders that did it named.

    .venv\\Scripts\\python.exe scripts\\verify_gstreamer_bundle.py --bundle <dist>\\_internal [--media file.mkv ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

#: Audio and video caps the development installation can decode and the
#: bundle knowingly cannot, each with the reason it was not worth shipping.
ALLOWED_LOSSES: dict[str, str] = {
    # a52dec and dtsdec are GPL; libav decodes AC-3 and DTS under their
    # standard caps, only the DVD "private stream 1" spellings are lost.
    "audio/ac3": "a52dec's alternative AC-3 caps name",
    "audio/x-private1-ac3": "AC-3 inside a DVD .vob private stream (a52dec, GPL)",
    "audio/x-private1-dts": "DTS inside a DVD .vob private stream (dtsdec, GPL)",
    "audio/x-sbc": "Bluetooth audio",
    "audio/x-siren": "VoIP audio",
    "audio/x-speex": "VoIP audio",
    "video/x-cdg": "karaoke CD+G graphics",
    "video/x-fli": "Autodesk FLI/FLC animation",
    "video/x-raw": "lcevcdec enhancement layers and dvbsubenc, misfiled as decoders",
    "video/x-xvid": "a DirectShow wrapper; libav decodes MPEG-4 ASP itself",
}


# ----------------------------------------------------------------- the child
def inspect_runtime(root: Path, required: list[str], media: list[str]) -> dict:
    root = root.resolve()
    sys.path.insert(0, str(root))
    for key in list(os.environ):
        if key.startswith("GST_") or key in {"GI_TYPELIB_PATH", "PYGI_DLL_DIRS"}:
            del os.environ[key]
    import gstreamer_libs

    gstreamer_libs.setup_python_environment()
    plugin_paths = [str(p) for p in root.glob("gstreamer_*/lib/gstreamer-1.0") if p.is_dir()]
    os.environ["GST_PLUGIN_PATH_1_0"] = os.pathsep.join(plugin_paths)
    os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"] = ""
    with tempfile.TemporaryDirectory(prefix="vmc-gst-registry-") as temp:
        os.environ["GST_REGISTRY_1_0"] = str(Path(temp) / "registry.bin")
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        registry = Gst.Registry.get()
        plugins: dict[str, str] = {}
        for plugin in registry.get_plugin_list():
            filename = plugin.get_filename()
            if not filename:
                continue  # static plugins (coreelements' "staticelements")
            path = Path(filename).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError(f"a plugin outside the runtime under test was loaded: {filename}")
            plugins[plugin.get_name()] = path.relative_to(root).as_posix()
        # Rank NONE is excluded: decodebin3 never autoplugs those, so a
        # rank-0 decoder does not make a codec playable.
        decodable = sorted({
            structure.get_name()
            for factory in registry.get_feature_list(Gst.ElementFactory)
            if "Decoder" in (factory.get_metadata(Gst.ELEMENT_METADATA_KLASS) or "").split("/")
            and factory.get_rank() > Gst.Rank.NONE
            for template in factory.get_static_pad_templates()
            if template.direction == Gst.PadDirection.SINK
            for structure in _structures(template.get_caps())
        })
        missing = [name for name in required if Gst.ElementFactory.find(name) is None]
        probes = [_probe(Gst, path, kind) for path in media for kind in ("video", "audio")]
        return {
            "version": Gst.version_string(),
            "plugins": plugins,
            "decodable": decodable,
            "missing_elements": missing,
            "probes": probes,
        }


def _structures(caps):
    return [caps.get_structure(i) for i in range(caps.get_size())]


def _probe(Gst, path: str, kind: str) -> dict:  # noqa: N803 - gi namespace
    """Runs the app's own branch topology on one file and pulls one buffer."""
    if kind == "video":
        # gstreamer_playback._build_video_branch, minus the sink and the
        # tone-map probe: decode, crop, upload, GPU convert, GPU memory out.
        chain = (
            "queue ! videocrop ! d3d11upload ! video/x-raw(memory:D3D11Memory) ! "
            "d3d11convert ! video/x-raw(memory:D3D11Memory)"
        )
    else:
        # gstreamer_playback._build_audio_branch, with the sink replaced.
        chain = "queue ! audioconvert ! audioresample ! volume"
    pipeline = Gst.parse_launch(
        f"filesrc name=file ! decodebin3 name=decode ! {chain} ! appsink name=out sync=false max-buffers=1 drop=true"
    )
    pipeline.get_by_name("file").set_property("location", str(Path(path).resolve()))
    decoder = pipeline.get_by_name("decode")
    # Only the stream kind under test, the way the app selects streams; an
    # unselected stream is never decoded, so its codec cannot fail the probe.
    decoder.connect(
        "select-stream",
        lambda _dec, _collection, stream: int(_caps_name(stream.get_caps()).startswith(kind + "/")),
    )
    sink = pipeline.get_by_name("out")
    result = {"file": Path(path).name, "kind": kind}
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("the pipeline refused to start")
        sample = sink.emit("try-pull-sample", 60 * Gst.SECOND)
        if sample is None:
            message = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
            detail = message.parse_error()[0].message if message else "no buffer arrived (no such stream?)"
            raise RuntimeError(f"{kind} probe of {Path(path).name} failed: {detail}")
        result["caps"] = sample.get_caps().to_string()
        result["decoders"] = _decoders_inside(Gst, decoder)
        del sample
    finally:
        pipeline.set_state(Gst.State.NULL)
    return result


def _caps_name(caps) -> str:
    return caps.get_structure(0).get_name() if caps is not None and caps.get_size() else ""


def _decoders_inside(Gst, element) -> list[str]:  # noqa: N803 - gi namespace
    """Names of the decoder elements decodebin3 ended up plugging."""
    found = []
    iterator = element.iterate_recurse()
    while True:
        status, child = iterator.next()
        if status != Gst.IteratorResult.OK:
            if status == Gst.IteratorResult.RESYNC:
                iterator.resync()
                continue
            break
        factory = child.get_factory()
        parts = (factory.get_metadata(Gst.ELEMENT_METADATA_KLASS) or "").split("/") if factory else []
        if "Decoder" in parts and "Bin" not in parts:  # decodebin3 calls itself a decoder too
            found.append(factory.get_name())
    return sorted(set(found))


# ---------------------------------------------------------------- the parent
def snapshot(root: Path, required: list[str], media: list[str]) -> dict:
    command = [sys.executable, "-S", str(Path(__file__).resolve()), "--child", str(root),
               "--require", json.dumps(required)]
    for path in media:
        command += ["--media", path]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120 + 90 * len(media))
    if result.returncode:
        raise SystemExit(f"inspecting {root} failed:\n{result.stderr.strip()}\n{result.stdout.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a packaged GStreamer runtime against the app's needs.")
    parser.add_argument("--bundle", type=Path, help="the build's _internal directory")
    parser.add_argument("--media", action="append", default=[], metavar="FILE",
                        help="a video to decode for real, through the app's branches (repeatable)")
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--require", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child is not None:
        print(json.dumps(inspect_runtime(args.child, json.loads(args.require or "[]"), args.media)))
        return
    if args.bundle is None:
        parser.error("--bundle is required")
    if not (args.bundle / "gstreamer_libs").is_dir():
        parser.error(f"{args.bundle} does not contain gstreamer_libs")

    sys.path.insert(0, str(PROJECT))
    from gstreamer_bundle import KEEP_PLUGINS

    from vmaf_app.core.gstreamer_playback import GPU_DECODERS, REQUIRED_ELEMENTS

    required = list(REQUIRED_ELEMENTS)
    print("inspecting the development installation...")
    baseline = snapshot(Path(sysconfig.get_paths()["purelib"]), required, [])
    print(f"inspecting {args.bundle}...")
    bundled = snapshot(args.bundle, required, args.media)

    problems = []
    expected = {stem.removeprefix("gst") for stem in KEEP_PLUGINS}
    loaded = set(bundled["plugins"])
    if failed := sorted(expected - loaded):
        problems.append(f"allow-listed plugins that did not load (a dependency is missing?): {failed}")
    if leaked := sorted(loaded - expected):
        problems.append(f"plugins in the bundle that the policy does not name: {leaked}")
    if bundled["missing_elements"]:
        problems.append(f"elements the app needs are missing: {bundled['missing_elements']}")
    media = {caps for caps in baseline["decodable"] if caps.startswith(("video/", "audio/"))}
    lost = sorted(media - set(bundled["decodable"]) - set(ALLOWED_LOSSES))
    if lost:
        problems.append(f"codecs the development installation decodes and the bundle does not: {lost}")
    if problems:
        raise SystemExit("GStreamer bundle verification FAILED:\n  - " + "\n  - ".join(problems))

    print(f"OK  {bundled['version']}: {len(loaded)} plugins, all allow-listed, all loaded from the bundle")
    print(f"OK  all {len(required)} elements the app uses exist")
    waived = sorted(media & set(ALLOWED_LOSSES) - set(bundled["decodable"]))
    print(f"OK  decodes {len(media) - len(waived)} of the {len(media)} audio/video codec caps the development "
          f"installation does; waived by policy: {', '.join(waived) or 'none'}")
    gpu = [name for name in GPU_DECODERS if name in {d for p in bundled["probes"] for d in p.get("decoders", [])}]
    for probe in bundled["probes"]:
        print(f"OK  {probe['kind']:<5} {probe['file']}: {', '.join(probe['decoders']) or 'no decoder element'}"
              f"  ->  {probe['caps'][:110]}")
    if args.media and not gpu:
        print("note: no probe decoded on the GPU (D3D11); this machine may have no hardware decoder for these codecs")


if __name__ == "__main__":
    main()
