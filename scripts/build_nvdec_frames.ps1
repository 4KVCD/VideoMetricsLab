# Builds vmaf_app/native/nvdec_frames.dll, the NVIDIA decoder the GPU metrics
# read their frames from (native/nvdec_frames.cpp). Needs only MinGW-w64 g++:
# the CUDA and NVDEC libraries come with the NVIDIA driver and are loaded at
# run time, and the kernels are PTX in the source. The headers in
# native/ffnvcodec are FFmpeg's nv-codec-headers (MIT).
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$outputDirectory = Join-Path $projectDirectory 'vmaf_app/native'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
& g++ -std=c++17 -O2 -Wall -Wextra -shared -static -s `
    -I (Join-Path $projectDirectory 'native') `
    (Join-Path $projectDirectory 'native/nvdec_frames.cpp') `
    -o (Join-Path $outputDirectory 'nvdec_frames.dll') '-Wl,--no-insert-timestamp'
if ($LASTEXITCODE -ne 0) { throw 'NVDEC frame decoder build failed' }
