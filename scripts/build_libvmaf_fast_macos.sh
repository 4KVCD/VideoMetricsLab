#!/bin/bash
# Builds libvmaf-fast's libvmaf.dylib for macOS (arm64) from the commit the
# Windows libvmaf.dll is built from (resources/tools/libvmaf/libvmaf-fast.json:
# its release tag, or during development a commit of the fork's fast branch)
# and installs it into resources/tools/libvmaf. libvmaf-fast publishes Windows
# builds only; this is the build its CI makes on macOS. zimg (HDR VMAF's
# picture conversion) is linked in statically, as on Windows, so the dylib
# needs only the system's libraries.
#
#     scripts/build_libvmaf_fast_macos.sh
#
# Needs git, meson, ninja and zimg (brew install meson zimg). The dylib is not
# committed (.gitignore): the release script runs this.
set -euo pipefail

deployment_target="14.4"

project="$(cd "$(dirname "$0")/.." && pwd)"
target="$project/resources/tools/libvmaf"
work="$(mktemp -d "${TMPDIR:-/tmp}/libvmaf-fast-XXXXXX")"
trap 'rm -rf "$work"' EXIT

pinned() {  # a field of libvmaf-fast.json
    sed -n "s/.*\"$1\": *\"\{0,1\}\([^\",]*\)\"\{0,1\}.*/\1/p" "$target/libvmaf-fast.json"
}
version="$(pinned version)"
commit_expected="$(pinned commit)"
# Fetched into an empty repository: a shallow clone of an annotated tag
# (clone --depth 1 --branch) warns that the tag "is not a commit".
git init --quiet "$work/src"
if [[ "$(pinned release)" == "true" ]]; then
    ref="refs/tags/libvmaf-fast-$version"
    git -C "$work/src" fetch --quiet --depth 1 https://github.com/4KVCD/libvmaf-fast.git "$ref:$ref"
else
    git -C "$work/src" fetch --quiet --depth 1 https://github.com/4KVCD/libvmaf-fast.git "$commit_expected"
fi
git -C "$work/src" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
commit="$(git -C "$work/src" rev-parse HEAD)"
if [[ "$commit" != "$commit_expected" ]]; then
    echo "libvmaf-fast $version is commit $commit, libvmaf-fast.json says $commit_expected" >&2
    exit 1
fi

# zimg's static library: Homebrew's zimg.pc names -lstdc++ for a static link,
# which macOS does not have (its C++ library is libc++).
zimg="$(brew --prefix zimg)"
zimg_version="$(pkg-config --modversion "$zimg/lib/pkgconfig/zimg.pc")"
mkdir -p "$work/pkgconfig"
cat > "$work/pkgconfig/zimg.pc" <<EOF
Name: zimg
Description: zimg, linked statically
Version: $zimg_version
Cflags: -I$zimg/include
Libs: $zimg/lib/libzimg.a -lc++
EOF
export PKG_CONFIG_PATH="$work/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

export MACOSX_DEPLOYMENT_TARGET="$deployment_target"
# Built with Clang's own floating-point settings, as Homebrew builds FFmpeg's
# libvmaf: the app's PSNR, SSIM and XPSNR are then the user's FFmpeg's to the
# last digit, as the app's are Gyan's FFmpeg's on Windows, so that a score
# saved from either is the same. (With -ffp-contract=off, which stops Clang
# fusing a multiply and an add into one rounding on Apple's chips, SSIM is
# Windows' to the last digit instead, and differs from Homebrew's FFmpeg's
# on 3 frames in 24: LibvmafCpu.PsnrSsimAndXpsnrInTheAppAreFfmpegsOwnScores.)
meson setup "$work/src/libvmaf" "$work/build" --buildtype release -Denable_float=true \
    -Denable_tests=false -Denable_docs=false -Dbuilt_in_models=true -Denable_zimg=true \
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
