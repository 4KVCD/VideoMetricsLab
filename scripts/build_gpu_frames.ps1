# Builds the GPU frame decoders the GPU metrics read their frames from, into
# vmaf_app/native: nvdec_frames.dll (NVIDIA, native/nvdec_frames.cpp),
# vpl_frames.dll (Intel, native/vpl_frames.cpp), amf_frames.dll (AMD,
# native/amf_frames.cpp) and mf_frames.dll (AMD, Windows' own decoders:
# native/mf_frames.cpp), with one C API (native/gpu_frames.h). Needs only
# MinGW-w64 g++: each GPU maker's decoder library comes with its driver and is
# loaded at run time, and NVIDIA's kernels are PTX in the source. The headers
# in native/ffnvcodec, native/onevpl, native/amf and native/vulkan are FFmpeg's
# nv-codec-headers, Intel's oneVPL API, AMD's AMF SDK and Khronos' Vulkan
# headers (all MIT; amf_frames hands its pictures over to Vulkan on the GPU).
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$outputDirectory = Join-Path $projectDirectory 'vmaf_app/native'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$native = Join-Path $projectDirectory 'native'
foreach ($build in @(
        @{ Name = 'nvdec_frames'; Libraries = @(); Includes = @() },
        @{ Name = 'vpl_frames'; Libraries = @('-ld3d11', '-ldxgi', '-luuid'); Includes = @() },
        @{ Name = 'amf_frames'; Libraries = @('-ld3d11', '-ldxgi', '-luuid')
            Includes = @('-I', (Join-Path $native 'vulkan')) },
        @{ Name = 'mf_frames'; Libraries = @('-ld3d11', '-ldxgi', '-luuid', '-lmfplat', '-lmfuuid', '-lole32')
            Includes = @('-I', (Join-Path $native 'vulkan')) })) {
    & g++ -std=c++17 -O3 -Wall -Wextra -shared -static -s -I $native @($build.Includes) `
        (Join-Path $native "$($build.Name).cpp") `
        -o (Join-Path $outputDirectory "$($build.Name).dll") @($build.Libraries) '-Wl,--no-insert-timestamp'
    if ($LASTEXITCODE -ne 0) { throw "$($build.Name).dll build failed" }
}
