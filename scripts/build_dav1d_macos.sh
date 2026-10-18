#!/bin/bash
# Builds dav1d, the AV1 decoder Video Compare's native playback decodes AV1
# with in software where VideoToolbox cannot (Macs before the M3), into
# resources/tools/ffmpeg beside FFmpeg's libraries, where the app looks for
# it (dav1d_library_path), as scripts/build_dav1d.ps1 builds Windows':
#   libdav1d.dylib                VideoLAN's dav1d, named @rpath/libdav1d.dylib
#   licenses/COPYING.dav1d.txt    its licence (BSD 2-clause)
#
# dav1d's release source is pinned by its version and the SHA-256 of its
# archive, from VideoLAN's own server, and built without dav1d's tools,
# tests or examples. AV1's decoding is defined exactly, so any dav1d release
# decodes the same pictures. It is not committed (.gitignore): the release
# script runs this.
#
#     scripts/build_dav1d_macos.sh
#
# Needs meson and ninja (brew install meson) and curl.
set -euo pipefail

version="1.5.4"
sha256="686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5"
deployment_target="14.4"

project="$(cd "$(dirname "$0")/.." && pwd)"
output="$project/resources/tools/ffmpeg"
work="$(mktemp -d "${TMPDIR:-/tmp}/dav1d-$version-XXXXXX")"
trap 'rm -rf "$work"' EXIT

curl -sSfL -o "$work/dav1d-$version.tar.xz" "https://downloads.videolan.org/pub/videolan/dav1d/$version/dav1d-$version.tar.xz"
actual="$(shasum -a 256 "$work/dav1d-$version.tar.xz" | cut -d' ' -f1)"
if [[ "$actual" != "$sha256" ]]; then
    echo "dav1d-$version.tar.xz has SHA-256 $actual, expected $sha256" >&2
    exit 1
fi
tar -xJf "$work/dav1d-$version.tar.xz" -C "$work"

export MACOSX_DEPLOYMENT_TARGET="$deployment_target"
meson setup "$work/build" "$work/dav1d-$version" --buildtype=release --default-library=shared -Db_ndebug=true \
    -Denable_tools=false -Denable_tests=false -Denable_examples=false \
    -Dc_args="-arch arm64" -Dc_link_args="-arch arm64" > "$work/setup.log" \
    || { cat "$work/setup.log" >&2; exit 1; }
ninja -C "$work/build" > "$work/build.log" || { tail -40 "$work/build.log" >&2; exit 1; }

built="$(find "$work/build/src" -name 'libdav1d.*.dylib' -type f | head -1)"
if [[ -z "$built" ]]; then
    echo "the build made no libdav1d dylib" >&2
    exit 1
fi
mkdir -p "$output/licenses"
cp "$built" "$output/libdav1d.dylib"
chmod 644 "$output/libdav1d.dylib"
install_name_tool -id @rpath/libdav1d.dylib "$output/libdav1d.dylib" 2>/dev/null
codesign --force --sign - "$output/libdav1d.dylib"
cp "$work/dav1d-$version/COPYING" "$output/licenses/COPYING.dav1d.txt"

echo "dav1d $version into $output"
otool -L "$output/libdav1d.dylib" | tail -n +2
shasum -a 256 "$output/libdav1d.dylib"
