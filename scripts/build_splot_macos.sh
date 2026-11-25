#!/bin/bash
# Builds splot, the AV2 decoder the app decodes AV2 with (the software frame
# decoder, native/software_frames.cpp, the AV2 probe and Video Compare's
# playback), into resources/tools/ffmpeg beside FFmpeg's libraries and dav1d,
# where the app looks for it (splot_library_path), as scripts/build_splot.ps1
# builds Windows':
#   libsplot.dylib                splot, named @rpath/libsplot.dylib
#   licenses/LICENSE.splot.md     its licence
#
# splot (github.com/splotlabs/splot, an AV2 decoder in Rust) has no release
# yet, and AV2 is not final: its source is pinned by the same commit as
# Windows' (the hash fixes every byte of the source). Apple's chips all have
# NEON, which splot's arm64 code needs: one library serves every Mac. It is
# not committed (.gitignore): the release script runs this.
#
#     scripts/build_splot_macos.sh
#
# Needs git and rustup (the commit's rust-toolchain.toml names the Rust
# version, which rustup installs on first use).
set -euo pipefail

commit="19ba2a57e89b1176f9d78c3bdbbf6127e80ea004"
deployment_target="14.4"

project="$(cd "$(dirname "$0")/.." && pwd)"
output="$project/resources/tools/ffmpeg"
work="$(mktemp -d "${TMPDIR:-/tmp}/splot-${commit:0:8}-XXXXXX")"
trap 'rm -rf "$work"' EXIT

git clone --quiet https://github.com/splotlabs/splot.git "$work/splot"
git -C "$work/splot" -c advice.detachedHead=false checkout --quiet "$commit"
head="$(git -C "$work/splot" rev-parse HEAD)"
if [[ "$head" != "$commit" ]]; then
    echo "splot checked out $head, expected $commit" >&2
    exit 1
fi

(cd "$work/splot" && MACOSX_DEPLOYMENT_TARGET="$deployment_target" CARGO_TARGET_DIR="$work/target" \
    cargo build --release --locked -p splot-capi)
built="$work/target/release/libsplot_capi.dylib"
if [[ ! -f "$built" ]]; then
    echo "the build made no libsplot_capi.dylib" >&2
    exit 1
fi

mkdir -p "$output/licenses"
cp "$built" "$output/libsplot.dylib"
chmod 644 "$output/libsplot.dylib"
install_name_tool -id @rpath/libsplot.dylib "$output/libsplot.dylib" 2>/dev/null
codesign --force --sign - "$output/libsplot.dylib"
cp "$work/splot/LICENSE.md" "$output/licenses/LICENSE.splot.md"

# It links the system's libraries only.
if otool -L "$output/libsplot.dylib" | tail -n +2 | awk '{print $1}' \
    | grep -v -E '^(@rpath/libsplot\.dylib|/usr/lib/|/System/Library/)'; then
    echo "libsplot.dylib links a library that is not the system's" >&2
    exit 1
fi
echo "splot $commit into $output"
otool -L "$output/libsplot.dylib" | tail -n +2
printf '  %s bytes  SHA-256 %s\n' "$(stat -f %z "$output/libsplot.dylib")" \
    "$(shasum -a 256 "$output/libsplot.dylib" | cut -d' ' -f1)"
