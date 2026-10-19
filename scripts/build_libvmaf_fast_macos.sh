#!/bin/bash
# Builds libvmaf-fast's libvmaf.dylib for macOS (arm64) from its release tag
# and installs it into resources/tools/libvmaf, beside the Windows
# libvmaf.dll and the libvmaf-fast.json both are built from. libvmaf-fast
# publishes Windows builds only; this is the build its CI makes on macOS.
#
#     scripts/build_libvmaf_fast_macos.sh [VERSION]
#
# Needs git, meson and ninja (brew install meson). The dylib is not
# committed (.gitignore): the release script runs this.
set -euo pipefail

version="${1:-v1.1}"
commit_expected="994b1814a2197c3dda9a983fc4ae08f6a2fc01c0"  # libvmaf-fast-v1.1
deployment_target="14.4"

project="$(cd "$(dirname "$0")/.." && pwd)"
target="$project/resources/tools/libvmaf"
work="$(mktemp -d "${TMPDIR:-/tmp}/libvmaf-fast-XXXXXX")"
trap 'rm -rf "$work"' EXIT

git -c advice.detachedHead=false clone --quiet --depth 1 --branch "libvmaf-fast-$version" https://github.com/4KVCD/libvmaf-fast.git "$work/src"
commit="$(git -C "$work/src" rev-parse HEAD)"
if [[ "$version" == "v1.1" && "$commit" != "$commit_expected" ]]; then
    echo "libvmaf-fast-$version is commit $commit, expected $commit_expected" >&2
    exit 1
fi
if ! grep -q "\"commit\": \"$commit\"" "$target/libvmaf-fast.json"; then
    echo "resources/tools/libvmaf/libvmaf-fast.json is not commit $commit: install the same version on Windows" >&2
    exit 1
fi

export MACOSX_DEPLOYMENT_TARGET="$deployment_target"
# Built with Clang's own floating-point settings, as Homebrew builds FFmpeg's
# libvmaf: the app's PSNR, SSIM and XPSNR are then the user's FFmpeg's to the
# last digit, as the app's are Gyan's FFmpeg's on Windows, so that a score
# saved from either is the same. (With -ffp-contract=off, which stops Clang
# fusing a multiply and an add into one rounding on Apple's chips, SSIM is
# Windows' to the last digit instead, and differs from Homebrew's FFmpeg's
# on 3 frames in 24: LibvmafCpu.PsnrSsimAndXpsnrInTheAppAreFfmpegsOwnScores.)
meson setup "$work/src/libvmaf" "$work/build" --buildtype release -Denable_float=true \
    -Denable_tests=false -Denable_docs=false -Dbuilt_in_models=true \
    -Dc_args="-arch arm64" -Dcpp_args="-arch arm64" -Dc_link_args="-arch arm64" -Dcpp_link_args="-arch arm64" \
    > "$work/setup.log" || { cat "$work/setup.log" >&2; exit 1; }
ninja -C "$work/build" > "$work/build.log" || { tail -40 "$work/build.log" >&2; exit 1; }

built="$(find "$work/build/src" -name 'libvmaf.*.dylib' -type f | head -1)"
if [[ -z "$built" ]]; then
    echo "the build made no libvmaf dylib" >&2
    exit 1
fi
cp "$built" "$target/libvmaf.dylib"
chmod 644 "$target/libvmaf.dylib"
# The app loads it by its path (dlopen): its own name says where it is.
install_name_tool -id @loader_path/libvmaf.dylib "$target/libvmaf.dylib" 2>/dev/null
codesign --force --sign - "$target/libvmaf.dylib"

echo "libvmaf-fast $version ($commit) into $target"
otool -L "$target/libvmaf.dylib" | tail -n +2
shasum -a 256 "$target/libvmaf.dylib"
