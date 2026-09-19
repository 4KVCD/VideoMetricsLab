"""Which files of the GStreamer wheels the Windows build ships.

The wheels are a complete media framework: 446 binaries, 302 MB, of which
this app reaches 95. It plays two videos through D3D11 and one
soundtrack; it never encodes, streams, captions, or talks to a network. So
the bundle is built from an allow-list of plugins, their DLL dependencies
found by reading import tables, and the handful of files that imports cannot
reveal -- not by copying the wheels and hoping.

Dependencies are computed, never listed by hand, so upgrading the wheels
needs no edit here unless a plugin is renamed. The list of plugins IS hand
maintained, on purpose: every entry says what the app uses it for.

Nothing in the development installation is modified or deleted.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sysconfig
from pathlib import Path

PACKAGES = (
    "gstreamer_libs",
    "gstreamer_plugins",
    "gstreamer_plugins_libs",
    "gstreamer_plugins_restricted",
    "gstreamer_plugins_gpl",
    "gstreamer_plugins_gpl_restricted",
    "gstreamer_python",
    "gstreamer_ext_runtime",
)

#: Plugin DLL stems and what the app uses each for. The pipelines are built
#: in vmaf_app/core/gstreamer_playback.py and locked_presentation.py;
#: decodebin3 and playbin3 pick demuxers, parsers and decoders from
#: whatever is registered, which is why the containers and codecs are here.
KEEP_PLUGINS: dict[str, str] = {
    # -- the pipeline itself ------------------------------------------------
    "gstcoreelements": "filesrc, queue, multiqueue, capsfilter, typefind, identity, tee",
    "gsttypefindfunctions": "recognises what kind of file a URI points at",
    "gstplayback": "decodebin3, parsebin, playbin3 and its uridecodebin3, playsink",
    "gstapp": "appsink (frame-locked pool) and appsrc (locked presenter)",
    "gstvideocrop": "videocrop: removes the detected black bars",
    "gstdebug": "capssetter: retags tone-mapped frames as SDR",
    "gstd3d11": "d3d11{h264,h265,av1,vp9,mpeg2}dec, d3d11upload, d3d11convert, d3d11videosink",
    "gstvideoconvertscale": "videoconvert/videoscale for anything left in system memory",
    "gstpbtypes": "dynamic caps types (multiview flags) the parsers may reference",
    # -- elementary stream parsers (decodebin3 needs them before decoders) --
    "gstvideoparsersbad": "h264parse, h265parse, h266parse, av1parse, vp9parse, mpegvideoparse",
    "gstaudioparsers": "aacparse, ac3parse, dcaparse, flacparse, mpegaudioparse",
    "gstopusparse": "opusparse: Opus in Matroska needs it to negotiate",
    # -- containers ---------------------------------------------------------
    "gstmatroska": "matroskademux: .mkv and .webm",
    "gstisomp4": "qtdemux: .mp4, .mov, .m4v",
    "gstmpegtsdemux": "tsdemux: .ts and .m2ts",
    "gstmpegpsdemux": "mpegpsdemux: .mpg and .vob",
    "gstavi": "avidemux",
    "gstmxf": "mxfdemux",
    "gstogg": "oggdemux",
    "gstflv": "flvdemux",
    "gstasf": "asfdemux: .wmv",
    "gstwavparse": "wavparse",
    # -- software decoders, for what the GPU cannot do -----------------------
    "gstlibav": "avdec_h266 (VVC), avdec_h264 10-bit, ProRes, DNxHD, and every audio codec",
    "gstdav1d": "dav1ddec: AV1 on GPUs without an AV1 decoder",
    "gsttheora": "theoradec: video in .ogv",
    # -- the soundtrack (playbin3 audio flags, gstreamer_playback audio branch)
    "gstaudioconvert": "audioconvert",
    "gstaudioresample": "audioresample",
    "gstvolume": "volume: soft volume and mute",
    "gstautodetect": "autoaudiosink",
    "gstwasapi2": "wasapi2sink: the sink autoaudiosink picks on Windows 10+",
    "gstwasapi": "wasapisink: fallback",
    "gstdirectsound": "directsoundsink: fallback",
    "gstopus": "opusdec",
    "gstvorbis": "vorbisdec",
    "gstflac": "flacdec",
    "gstmpg123": "mpg123audiodec",
    "gstwavpack": "wavpackdec",
    "gstalaw": "alawdec: telephone-law PCM in .mov and .avi",
    "gstmulaw": "mulawdec",
    "gstdvdlpcmdec": "dvdlpcmdec: LPCM in Blu-ray .m2ts and DVD .vob",
}

#: Started by GLib and GStreamer as processes, so no import table names them.
#: The scanner loads plugins out of process so that a broken plugin cannot
#: take the app down; GLib launches it through the spawn helpers.
KEEP_TOOLS = (
    "gstreamer_libs/libexec/gstreamer-1.0/gst-plugin-scanner.exe",
    "gstreamer_libs/bin/gspawn-win64-helper.exe",
    "gstreamer_libs/bin/gspawn-win64-helper-console.exe",
)

#: Loaded by the app itself through ctypes, by name (vmaf_app/core/d3d11_tonemap.py).
KEEP_LIBRARIES = ("gstd3d11-1.0-0.dll",)

#: Copied whole: the gi bindings the app imports, and every typelib (3 MB
#: in all; choosing among them would save nothing worth a missed dependency).
KEEP_TREES = (
    "gstreamer_python/Lib/site-packages/gi",
    "gstreamer_python/lib/girepository-1.0",
    "gstreamer_libs/lib/girepository-1.0",
)

#: Inside the kept trees: cairo is a foreign type gi only loads on request
#: (gi.require_foreign), which the app never makes.
DROP_WITHIN_TREES = ("*/_gi_cairo*.pyd",)

#: Each wheel's package module. gstreamer_libs.gstreamer_env() imports the
#: others by name to assemble PATH and the plugin path, so the packages must
#: exist even where nothing else of theirs is shipped.
PACKAGE_FILES = ("__init__.py",)

BINARY_SUFFIXES = {".dll", ".pyd", ".exe"}


def pe_imports(path: Path) -> set[str]:
    """DLL names a PE file imports, normal and delay-loaded, lower-cased."""
    import pefile  # PyInstaller's own dependency; the app never needs it

    with pefile.PE(str(path), fast_load=True) as pe:
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
            ]
        )
        return {
            entry.dll.decode("ascii", "replace").lower()
            for field in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT")
            for entry in getattr(pe, field, None) or ()
        }


def dependency_closure(seeds: set[Path], binaries: list[Path], imports=pe_imports) -> set[Path]:
    """Every binary reachable from `seeds` through import tables.

    Only names that exist among `binaries` are followed; system DLLs are
    not shipped. When a name exists in several packages every copy is kept:
    which one Windows loads depends on the DLL search order at run time,
    and second-guessing that is how a build works on one machine only.
    """
    by_name: dict[str, list[Path]] = {}
    for path in binaries:
        by_name.setdefault(path.name.lower(), []).append(path)
    kept: set[Path] = set()
    pending = list(seeds)
    while pending:
        path = pending.pop()
        if path in kept:
            continue
        kept.add(path)
        for name in imports(path):
            for candidate in by_name.get(name, ()):
                if candidate not in kept:
                    pending.append(candidate)
    return kept


def _tree_files(site: Path, tree: str) -> list[Path]:
    root = site / tree
    if not root.is_dir():
        return []
    return [
        p for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
        and not any(fnmatch.fnmatch(p.as_posix(), pattern) for pattern in DROP_WITHIN_TREES)
    ]


def collect(site: Path) -> tuple[list[tuple[str, str]], dict]:
    """PyInstaller `datas` entries for the pruned bundle, and a report.

    Every entry is one file, placed at the same path relative to the
    site-packages root, so each wheel still finds its own bin/, plugin and
    typelib directories from __file__.
    """
    missing = [p for p in PACKAGES if not (site / p).is_dir()]
    if missing:
        raise RuntimeError(f"GStreamer wheels not installed: {', '.join(missing)}; "
                           f"run pip install -r requirements.txt")

    binaries: list[Path] = []
    plugin_by_stem: dict[str, list[Path]] = {}
    for package in PACKAGES:
        for sub in ("bin", "lib/gstreamer-1.0", "libexec/gstreamer-1.0"):
            folder = site / package / sub
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in BINARY_SUFFIXES:
                    binaries.append(path)
                    if sub == "lib/gstreamer-1.0" and path.suffix.lower() == ".dll":
                        plugin_by_stem.setdefault(path.stem.lower(), []).append(path)
    all_shipped = sorted({p for package in PACKAGES for p in (site / package).rglob("*")
                          if p.is_file() and "__pycache__" not in p.parts})

    unknown = sorted(stem for stem in KEEP_PLUGINS if stem.lower() not in plugin_by_stem)
    if unknown:
        raise RuntimeError(f"KEEP_PLUGINS names plugins these wheels do not have: {', '.join(unknown)}")

    seeds: set[Path] = set()
    for stem in KEEP_PLUGINS:
        seeds.update(plugin_by_stem[stem.lower()])
    for rel in KEEP_TOOLS:
        path = site / rel
        if not path.is_file():
            raise RuntimeError(f"expected {rel} in the GStreamer wheels")
        seeds.add(path)
    for name in KEEP_LIBRARIES:
        found = [p for p in binaries if p.name.lower() == name.lower()]
        if not found:
            raise RuntimeError(f"expected {name} in the GStreamer wheels")
        seeds.update(found)
    trees: list[Path] = []
    for tree in KEEP_TREES:
        trees.extend(_tree_files(site, tree))
    # The gi extension modules pull in girepository, gobject, glib and ffi.
    seeds.update(p for p in trees if p.suffix.lower() == ".pyd")

    kept_binaries = dependency_closure(seeds, binaries + [p for p in trees if p.suffix.lower() == ".pyd"])
    kept: set[Path] = set(kept_binaries) | set(trees)
    for package in PACKAGES:
        for name in PACKAGE_FILES:
            path = site / package / name
            if path.is_file():
                kept.add(path)

    kept_sorted = sorted(kept)
    omitted = [p for p in all_shipped if p not in kept]
    report = {
        "policy": "allow-listed plugins, their import closure, the plugin scanner and "
                  "spawn helpers, the gi bindings and all typelibs",
        "original_bytes": sum(p.stat().st_size for p in all_shipped),
        "bundled_bytes": sum(p.stat().st_size for p in kept_sorted),
        "kept_plugins": sorted(p.stem for p in kept_binaries if p.stem.lower() in plugin_by_stem),
        "kept": [_entry(site, p) for p in kept_sorted],
        "omitted": [_entry(site, p) for p in omitted],
    }
    return [(str(p), p.relative_to(site).parent.as_posix()) for p in kept_sorted], report


def _entry(site: Path, path: Path) -> dict:
    return {"path": path.relative_to(site).as_posix(), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Show what the GStreamer bundle would contain.")
    parser.add_argument("--site", type=Path, default=Path(sysconfig.get_paths()["purelib"]))
    parser.add_argument("--json", action="store_true", help="the full report as JSON")
    args = parser.parse_args()
    _datas, report = collect(args.site)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    mib = 1024 ** 2
    print(f"GStreamer wheels: {report['original_bytes'] / mib:.1f} MiB installed, "
          f"{report['bundled_bytes'] / mib:.1f} MiB shipped ({len(report['kept'])} files)")
    print(f"plugins kept ({len(report['kept_plugins'])}): {', '.join(report['kept_plugins'])}")
    print("\nlargest files shipped:")
    for item in sorted(report["kept"], key=lambda e: -e["bytes"])[:15]:
        print(f"  {item['bytes'] / mib:6.1f} MiB  {item['path']}")
    print("\nlargest files left out:")
    for item in sorted(report["omitted"], key=lambda e: -e["bytes"])[:15]:
        print(f"  {item['bytes'] / mib:6.1f} MiB  {item['path']}")


if __name__ == "__main__":
    main()
