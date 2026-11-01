# Builds resources/native/vvdecvk_frames.dll (native/vvdecvk_frames.cpp): VVC decoded on the GPU by vvdec-vulkan
# (github.com/4KVCD/vvdec-vulkan, Fraunhofer's VVdeC with its reconstruction on the GPU in Vulkan, its parse on the
# CPU), its pictures handed over on the GPU to Vulkan VMAF (vvdec_handover_*). v2.4's first step: the app takes it only
# with VML_VVDEC_HANDOVER=1.
#
# The decoder is built with MinGW-w64 (as the other native decoders, which link it statically) from a checkout of
# vvdec-vulkan with the hand-over (branch gpu-handover), its shaders compiled with Slang (slangc) and Khronos' Vulkan
# headers; vulkan-1.dll is loaded at run time.
#
#   scripts\build_vvdecvk_frames.ps1 -Source <vvdec-vulkan checkout> -Slangc <slangc.exe> -VulkanHeaders <include dir>
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Slangc,
    [Parameter(Mandatory = $true)][string]$VulkanHeaders
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$native = Join-Path $projectDirectory 'native'
$outputDirectory = Join-Path $projectDirectory 'resources/native'
$build = Join-Path $env:LOCALAPPDATA 'VideoMetricsLab-build/vvdec-vulkan'
foreach ($tool in @('g++', 'cmake', 'mingw32-make')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool (MinGW-w64) is needed on the PATH" }
}
# The tools' own warnings go to stderr, which Windows PowerShell turns into errors when it is redirected: their exit
# codes say whether they failed (checked after each).
$ErrorActionPreference = 'Continue'

# The decoder's library, with the Vulkan path
& cmake -S $Source -B $build -G 'MinGW Makefiles' -DCMAKE_BUILD_TYPE=Release -DVVDEC_ENABLE_VULKAN=ON `
    "-DVVDEC_SLANGC=$Slangc" "-DVVDEC_VULKAN_HEADERS=$VulkanHeaders" -DBUILD_SHARED_LIBS=OFF -DVVDEC_LIBRARY_ONLY=ON `
    -DVVDEC_ENABLE_LINK_TIME_OPT=OFF -DVVDEC_ENABLE_WERROR=OFF
if ($LASTEXITCODE -ne 0) { throw 'configuring vvdec-vulkan failed' }
& mingw32-make -C $build -j ([Environment]::ProcessorCount) vvdec
if ($LASTEXITCODE -ne 0) { throw 'building vvdec-vulkan failed' }
$library = Join-Path $Source 'lib/release-static/libvvdec.a'

# The decoder module, vvdec linked in (its generated vvdec.h is in the build folder)
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
& g++ -std=c++17 -O3 -Wall -Wextra -shared -static -s -I $native -I $build -I (Join-Path $Source 'include') `
    (Join-Path $native 'vvdecvk_frames.cpp') $library -o (Join-Path $outputDirectory 'vvdecvk_frames.dll') `
    '-Wl,--no-insert-timestamp'
if ($LASTEXITCODE -ne 0) { throw 'vvdecvk_frames.dll build failed' }
Write-Host "vvdecvk_frames.dll: $(Join-Path $outputDirectory 'vvdecvk_frames.dll')"
