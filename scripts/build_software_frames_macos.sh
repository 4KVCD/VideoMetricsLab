#!/bin/bash
# Builds the app's software decoder for macOS (arm64), as
# scripts/build_gpu_frames.ps1 builds Windows' software_frames.dll: the
# decoder the GPU metrics are given the pictures of a video FFmpeg would
# decode in software by (native/software_frames.cpp, with the C API of
# native/gpu_frames.h), and the only one that adds VVC's H.274 film grain as
# the video is scored with (VVdeC, the app's FFmpeg's libvvdec). Into
# resources/native, where the app looks for it (gpu_frames.cpp's
# library_path):
#
#   software_frames.dylib    named @rpath/software_frames.dylib
#
# It links neither FFmpeg nor dav1d: the app loads the libraries of
# resources/tools/ffmpeg (scripts/build_ffmpeg_decoders_macos.sh,
# scripts/build_dav1d_macos.sh) and hands them over. It is compiled against
# native/ffmpeg's headers, Windows' build's, which are the same for this
# version. It is not committed (.gitignore): the release script runs this.
#
#     scripts/build_software_frames_macos.sh
#
# Needs the Command Line Tools' clang.
set -euo pipefail

deployment_target="14.4"

project="$(cd "$(dirname "$0")/.." && pwd)"
native="$project/native"
output="$project/resources/native"

mkdir -p "$output"
clang++ -std=c++17 -O3 -Wall -Wextra -Werror -shared -fvisibility=hidden \
    -arch arm64 -mmacosx-version-min="$deployment_target" \
    -I "$native" -I "$native/ffmpeg/include" "$native/software_frames.cpp" \
    -install_name @rpath/software_frames.dylib -o "$output/software_frames.dylib"
codesign --force --sign - "$output/software_frames.dylib"
# The system's libraries, nothing else.
if otool -L "$output/software_frames.dylib" | tail -n +2 | awk '{print $1}' \
    | grep -v -E '^(@rpath/software_frames\.dylib|/usr/lib/|/System/Library/)'; then
    echo "software_frames.dylib links a library that is not the system's" >&2
    exit 1
fi

echo "software_frames.dylib into $output"
printf '  %s bytes  SHA-256 %s\n' "$(stat -f %z "$output/software_frames.dylib")" \
    "$(shasum -a 256 "$output/software_frames.dylib" | cut -d' ' -f1)"
