#!/bin/bash
# Builds the FFmpeg libraries Video Compare's native playback uses on macOS
# (arm64), as scripts/build_ffmpeg_decoders.ps1 builds Windows': FFmpeg's
# release source, pinned by its version and the SHA-256 of its archive,
# configured LGPL (no --enable-gpl, no --enable-nonfree), with no program,
# filter or network: the same decoders, parsers and demuxers as Windows',
# VideoToolbox's decoding for H.264, HEVC, VP9, AV1, MPEG-2 and ProRes
# instead of Direct3D 11's, NEON's assembly and POSIX threads, and the same
# one external library: Fraunhofer HHI's VVC decoder VVdeC (master pinned by
# commit with 4KVCD's pull requests, scripts/patches/vvdec), linked into
# libavcodec as the decoder "libvvdec" (scripts/patches/ffmpeg). Into
# resources/tools/ffmpeg:
#
#   libavcodec.63.dylib libavformat.63.dylib libavutil.61.dylib
#   libswresample.7.dylib libswscale.10.dylib
#   licenses/COPYING.LGPLv2.1.txt licenses/COPYING.vvdec.txt
#
# each named @rpath/<itself>, with a run path of their own folder, so that
# they find each other wherever they are loaded from (the software decoder's,
# software_frames.dylib, loads libavcodec by its path in the command line
# too, which has no run path to them), and the window finds them through its
# run paths (cpp/app/CMakeLists.txt). They are not
# committed (.gitignore): the release script runs this. The headers the app
# compiles against are native/ffmpeg's, Windows' build's, which are the same
# for this version.
#
#     scripts/build_ffmpeg_decoders_macos.sh
#
# Needs the Command Line Tools' clang and make, curl, git, CMake and
# pkg-config (Homebrew's cmake and pkgconf).
set -euo pipefail

version="9.0.2"
sha256="8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e"
deployment_target="14.4"
# As scripts/build_ffmpeg_decoders.ps1's: VVdeC's master on 2026-10-28.
vvdec_commit="17e0f2f2089c5c0553f1bfc486b1b7aa551e7819"
decoders="h264,hevc,vvc,libvvdec,vp9,av1,mpeg2video,ffv1,prores,\
aac,aac_latm,ac3,eac3,dca,truehd,mlp,flac,opus,vorbis,mp3float,mp2float,alac,wmav1,wmav2,wmapro,\
pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_bluray,pcm_dvd"
parsers="h264,hevc,vvc,vp9,av1,mpegvideo,aac,aac_latm,ac3,dca,mlp,flac,opus,vorbis,mpegaudio"
demuxers="matroska,mov,mpegts,mpegps,avi,mxf,ogg,flv,asf,ivf,obu,h264,hevc,vvc,mpegvideo"
hwaccels="h264_videotoolbox,hevc_videotoolbox,vp9_videotoolbox,av1_videotoolbox,mpeg2_videotoolbox,prores_videotoolbox"
libraries=(libavcodec.63.dylib libavformat.63.dylib libavutil.61.dylib libswresample.7.dylib libswscale.10.dylib)

project="$(cd "$(dirname "$0")/.." && pwd)"
output="$project/resources/tools/ffmpeg"
work="$(mktemp -d "${TMPDIR:-/tmp}/ffmpeg-$version-XXXXXX")"
trap 'rm -rf "$work"' EXIT

curl -sSfL -o "$work/ffmpeg-$version.tar.xz" "https://ffmpeg.org/releases/ffmpeg-$version.tar.xz"
actual="$(shasum -a 256 "$work/ffmpeg-$version.tar.xz" | cut -d' ' -f1)"
if [[ "$actual" != "$sha256" ]]; then
    echo "ffmpeg-$version.tar.xz has SHA-256 $actual, expected $sha256" >&2
    exit 1
fi
tar -xJf "$work/ffmpeg-$version.tar.xz" -C "$work"

# VVdeC, as a static library installed under the work folder, where FFmpeg's
# configure finds it through pkg-config: the library only, without link-time
# optimisation and with its warnings not taken for errors, as on Windows.
git -c advice.detachedHead=false clone --quiet https://github.com/fraunhoferhhi/vvdec.git "$work/vvdec"
git -C "$work/vvdec" -c advice.detachedHead=false checkout --quiet "$vvdec_commit"
vvdec_version="$(git -C "$work/vvdec" describe --tags --always)+4kvcd-prs"
git -C "$work/vvdec" apply "$project/scripts/patches/vvdec/4kvcd-pull-requests.patch"
cmake -S "$work/vvdec" -B "$work/vvdec/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
    -DVVDEC_LIBRARY_ONLY=ON -DVVDEC_ENABLE_LINK_TIME_OPT=OFF -DVVDEC_ENABLE_WERROR=OFF \
    -DVVDEC_TOPLEVEL_OUTPUT_DIRS=OFF -DCMAKE_INSTALL_PREFIX="$work/vvdec-install" \
    > "$work/vvdec-configure.log" 2>&1 || { tail -30 "$work/vvdec-configure.log" >&2; exit 1; }
cmake --build "$work/vvdec/build" --target install -j"$(sysctl -n hw.ncpu)" > "$work/vvdec-build.log" 2>&1 \
    || { tail -40 "$work/vvdec-build.log" >&2; exit 1; }
export PKG_CONFIG_PATH="$work/vvdec-install/lib/pkgconfig"

# The decoder libvvdec in FFmpeg's source, and the name the app logs for it.
git -C "$work/ffmpeg-$version" apply "$project/scripts/patches/ffmpeg/libvvdec.patch"
cp "$project/scripts/patches/ffmpeg/libvvdec.c" "$work/ffmpeg-$version/libavcodec/libvvdec.c"
printf '#define LIBVVDEC_BUILD "VVdeC %s"\n' "$vvdec_version" > "$work/ffmpeg-$version/libavcodec/libvvdec_build.h"

# The prefix is a neutral path, installed under a staging folder: the
# configure line is in the libraries (avcodec_configuration), and a path of
# the Mac that built them, with its user's name, would be too.
cd "$work/ffmpeg-$version"
./configure --prefix=/ffmpeg --install-name-dir=@rpath --arch=arm64 --cc=clang \
    --extra-cflags="-arch arm64 -mmacosx-version-min=$deployment_target" \
    --extra-ldflags="-arch arm64 -mmacosx-version-min=$deployment_target -Wl,-rpath,@loader_path" \
    --enable-shared --disable-static --disable-programs --disable-doc --disable-avdevice \
    --disable-avfilter --disable-network --disable-autodetect --disable-everything \
    --enable-decoder="$decoders" --enable-parser="$parsers" --enable-demuxer="$demuxers" --enable-protocol=file \
    --enable-videotoolbox --enable-hwaccel="$hwaccels" --enable-pthreads \
    --enable-libvvdec --pkg-config-flags=--static \
    > "$work/configure.log" || { tail -30 "$work/configure.log" >&2; exit 1; }
make -j"$(sysctl -n hw.ncpu)" > "$work/build.log" 2>&1 || { tail -40 "$work/build.log" >&2; exit 1; }
make install DESTDIR="$work/stage" > /dev/null

mkdir -p "$output/licenses"
for library in "${libraries[@]}"; do
    cp "$work/stage/ffmpeg/lib/$library" "$output/$library"
    chmod 644 "$output/$library"
    codesign --force --sign - "$output/$library"
    # Each other's names and the system's, nothing else.
    if otool -L "$output/$library" | tail -n +2 | awk '{print $1}' \
        | grep -v -E '^(@rpath/|/usr/lib/|/System/Library/)'; then
        echo "$library links a library that is neither FFmpeg's nor the system's" >&2
        exit 1
    fi
done
cp "$work/ffmpeg-$version/COPYING.LGPLv2.1" "$output/licenses/COPYING.LGPLv2.1.txt"
cp "$work/vvdec/LICENSE.txt" "$output/licenses/COPYING.vvdec.txt"

echo "FFmpeg $version's libraries in $output"
for library in "${libraries[@]}"; do
    printf '  %s  %s bytes  SHA-256 %s\n' "$library" "$(stat -f %z "$output/$library")" \
        "$(shasum -a 256 "$output/$library" | cut -d' ' -f1)"
done
