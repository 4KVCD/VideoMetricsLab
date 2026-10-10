# Builds vulkan_presenter.dll, Video Compare's Vulkan presenter
# (native/vulkan_presenter.cpp), into vmaf_app/native. Needs MinGW-w64's g++.
# Its shaders are compiled into native/vulkan_presenter_spv.h, which is
# committed: scripts/build_vulkan_presenter_shader.py makes it again after a
# change to native/vulkan_presenter.slang.
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$native = Join-Path $projectDirectory 'native'
$outputDirectory = Join-Path $projectDirectory 'vmaf_app/native'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
& g++ -std=c++17 -O2 -Wall -Wextra -shared -static -I $native -I (Join-Path $native 'vulkan') `
    (Join-Path $native 'vulkan_presenter.cpp') `
    -o (Join-Path $outputDirectory 'vulkan_presenter.dll') -ld3d11 -ld3dcompiler -ldxgi -luser32
if ($LASTEXITCODE -ne 0) { throw 'Vulkan presenter build failed' }
