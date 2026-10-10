# Builds d3d11_film_grain.dll, native playback's film grain synthesis
# (native/d3d11_film_grain.cpp), into vmaf_app/native. Needs MinGW-w64's g++.
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$outputDirectory = Join-Path $projectDirectory 'vmaf_app/native'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
& g++ -std=c++17 -O2 -Wall -Wextra -shared -static -I (Join-Path $projectDirectory 'native') `
    (Join-Path $projectDirectory 'native/d3d11_film_grain.cpp') `
    -o (Join-Path $outputDirectory 'd3d11_film_grain.dll') -ld3d11 -ld3dcompiler
if ($LASTEXITCODE -ne 0) { throw 'Film grain build failed' }
