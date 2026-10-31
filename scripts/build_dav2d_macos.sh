#!/bin/bash
# Builds dav2d, the AV2 decoder the app decodes AV2 with (the software frame
# decoder, native/software_frames.cpp, and Video Compare's playback), into
# resources/tools/ffmpeg beside FFmpeg's libraries and dav1d, where the app
# looks for it (dav2d_library_path), as scripts/build_dav2d.ps1 builds
# Windows':
#   libdav2d.dylib                VideoLAN's dav2d, named @rpath/libdav2d.dylib
#   licenses/COPYING.dav2d.txt    its licence (BSD 2-clause)
#
# dav2d (VideoLAN's AV2 decoder, based on dav1d) has no release yet, and AV2
# is not final: its source is pinned by the same commit of VideoLAN's own
# repository as Windows' (the commit's hash fixes every byte of the source),
# and built as build_dav1d_macos.sh builds dav1d, without dav2d's tools,
# tests or examples. It is not committed (.gitignore): the release script
# runs this.
#
#     scripts/build_dav2d_macos.sh
#
# Needs meson and ninja (brew install meson) and git.
set -euo pipefail

commit="3dde00d02827fada3bb70c8fad9b2e06f598ea33"
deployment_target="14.4"

project="$(cd "$(dirname "$0")/.." && pwd)"
output="$project/resources/tools/ffmpeg"
work="$(mktemp -d "${TMPDIR:-/tmp}/dav2d-${commit:0:8}-XXXXXX")"
trap 'rm -rf "$work"' EXIT

git clone --quiet https://code.videolan.org/videolan/dav2d.git "$work/dav2d"
git -C "$work/dav2d" -c advice.detachedHead=false checkout --quiet "$commit"
head="$(git -C "$work/dav2d" rev-parse HEAD)"
if [[ "$head" != "$commit" ]]; then
    echo "dav2d checked out $head, expected $commit" >&2
    exit 1
fi

export MACOSX_DEPLOYMENT_TARGET="$deployment_target"
meson setup "$work/build" "$work/dav2d" --buildtype=release --default-library=shared -Db_ndebug=true \
    -Denable_tools=false -Denable_tests=false -Denable_examples=false \
    -Dc_args="-arch arm64" -Dc_link_args="-arch arm64" > "$work/setup.log" \
    || { cat "$work/setup.log" >&2; exit 1; }
ninja -C "$work/build" > "$work/build.log" || { tail -40 "$work/build.log" >&2; exit 1; }

built="$(find "$work/build/src" -name 'libdav2d.*.dylib' -type f | head -1)"
if [[ -z "$built" ]]; then
    echo "the build made no libdav2d dylib" >&2
    exit 1
fi
mkdir -p "$output/licenses"
cp "$built" "$output/libdav2d.dylib"
chmod 644 "$output/libdav2d.dylib"
install_name_tool -id @rpath/libdav2d.dylib "$output/libdav2d.dylib" 2>/dev/null
codesign --force --sign - "$output/libdav2d.dylib"
cp "$work/dav2d/COPYING" "$output/licenses/COPYING.dav2d.txt"

# The system's libraries, nothing else.
if otool -L "$output/libdav2d.dylib" | tail -n +2 | awk '{print $1}' \
    | grep -v -E '^(@rpath/libdav2d\.dylib|/usr/lib/|/System/Library/)'; then
    echo "libdav2d.dylib links a library that is not the system's" >&2
    exit 1
fi

echo "dav2d $commit into $output"
otool -L "$output/libdav2d.dylib" | tail -n +2
printf '  %s bytes  SHA-256 %s\n' "$(stat -f %z "$output/libdav2d.dylib")" \
    "$(shasum -a 256 "$output/libdav2d.dylib" | cut -d' ' -f1)"
