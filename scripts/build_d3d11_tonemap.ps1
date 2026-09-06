$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$outputDirectory = Join-Path $projectDirectory 'vmaf_app/native'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
& g++ -std=c++17 -O2 -shared -static `
    (Join-Path $projectDirectory 'native/d3d11_tonemap.cpp') `
    -o (Join-Path $outputDirectory 'd3d11_tonemap.dll') -ld3d11 -ld3dcompiler -lole32
if ($LASTEXITCODE -ne 0) { throw 'D3D11 tone mapper build failed' }
