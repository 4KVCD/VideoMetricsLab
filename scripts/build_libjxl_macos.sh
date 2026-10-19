#!/bin/bash
# Builds libjxl's ssimulacra2 and butteraugli_main for macOS (arm64) from
# libjxl's release tag and installs them into resources/tools/libjxl, beside
# the Windows .exe files of the same release. libjxl publishes no macOS
# binaries. They are linked statically with libjxl's own copies of their
# libraries (libpng, for the 16-bit PNGs the app hands them, brotli,
# highway, skcms, zlib), so that they need only the system's.
#
#     scripts/build_libjxl_macos.sh
#
# Needs git, CMake and Ninja. The tools are not committed (.gitignore): the
# release script runs this.
set -euo pipefail

version="v0.12.0"
commit_expected="a7a9c787341cf703dede03c2009fa460cae5e5df"
deployment_target="14.4"

project="$(cd "$(dirname "$0")/.." && pwd)"
target="$project/resources/tools/libjxl"
work="$(mktemp -d "${TMPDIR:-/tmp}/libjxl-XXXXXX")"
trap 'rm -rf "$work"' EXIT

git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$version" https://github.com/libjxl/libjxl.git "$work/src"
commit="$(git -C "$work/src" rev-parse HEAD)"
if [[ "$commit" != "$commit_expected" ]]; then
    echo "libjxl $version is commit $commit, expected $commit_expected" >&2
    exit 1
fi
git -C "$work/src" submodule update --quiet --init --depth 1 \
    third_party/brotli third_party/highway third_party/skcms third_party/libpng third_party/zlib

# Nothing from Homebrew or MacPorts: their libraries would be linked in, and
# the tools would need them on the user's Mac.
# -ffp-contract=off: Clang would fuse a multiply and an add into one rounding
# on Apple's chips, which the Windows build does not. Without it, Butteraugli
# differed from Windows' in the last digit on 10 frames in 96; with it, on
# one. (The app's own code is built so too: cpp/CMakeLists.txt.) SSIMULACRA2
# still differs by up to 0.015, its vector code summing in another order on
# Apple's chips (Highway's NEON against AVX2).
PKG_CONFIG_PATH="" PKG_CONFIG_LIBDIR="/nonexistent" cmake -S "$work/src" -B "$work/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
    -DCMAKE_IGNORE_PREFIX_PATH="/opt/homebrew;/usr/local;/opt/local" -DCMAKE_C_FLAGS=-ffp-contract=off -DCMAKE_CXX_FLAGS=-ffp-contract=off \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON -DJPEGXL_BUNDLE_LIBPNG=ON \
    -DJPEGXL_ENABLE_TOOLS=ON -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF \
    -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_SJPEG=OFF \
    -DJPEGXL_ENABLE_OPENEXR=OFF -DJPEGXL_ENABLE_PLUGINS=OFF \
    -DJPEGXL_ENABLE_SKCMS=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=OFF -DJPEGXL_FORCE_SYSTEM_HWY=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_GIF=ON -DCMAKE_DISABLE_FIND_PACKAGE_JPEG=ON \
    > "$work/configure.log" || { tail -40 "$work/configure.log" >&2; exit 1; }
cmake --build "$work/build" --target ssimulacra2 butteraugli_main > "$work/build.log" \
    || { tail -40 "$work/build.log" >&2; exit 1; }

for tool in ssimulacra2 butteraugli_main; do
    built="$(find "$work/build/tools" -name "$tool" -type f -perm -u+x | head -1)"
    if [[ -z "$built" ]]; then
        echo "the build made no $tool" >&2
        exit 1
    fi
    cp "$built" "$target/$tool"
    chmod 755 "$target/$tool"
    codesign --force --sign - "$target/$tool"
    # Only the system's libraries.
    if otool -L "$target/$tool" | tail -n +2 | grep -v -E '^\s+/(usr/lib|System/Library)/'; then
        echo "$tool links a library that is not the system's" >&2
        exit 1
    fi
done

echo "libjxl $version ($commit) into $target"
for tool in ssimulacra2 butteraugli_main; do
    otool -L "$target/$tool" | tail -n +2
    shasum -a 256 "$target/$tool"
done
