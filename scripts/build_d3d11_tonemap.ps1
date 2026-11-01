$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$outputDirectory = Join-Path $projectDirectory 'resources/native'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
# Reproducible in any folder: no link time, and the image base not made from the output's path
& g++ -std=c++17 -O2 -shared -static `
    (Join-Path $projectDirectory 'native/d3d11_tonemap.cpp') `
    -o (Join-Path $outputDirectory 'd3d11_tonemap.dll') -ld3d11 -ld3dcompiler -lole32 -ldxgi -luser32 `
    '-Wl,--no-insert-timestamp' '-Wl,--disable-auto-image-base'
if ($LASTEXITCODE -ne 0) { throw 'D3D11 tone mapper build failed' }
