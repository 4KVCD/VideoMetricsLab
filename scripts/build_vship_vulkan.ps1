# Builds Vship's Vulkan library (vmaf_app/tools/vship/vulkan/libvship.dll)
# from a pinned commit, with the MinGW-w64 g++ that builds the D3D11 tone
# mapper (WinLibs, UCRT, POSIX threads).
#
# Why a commit and not a release: Vship 5.1.1's Vulkan build cannot be loaded
# on a PC whose only GPU is Intel's (WinError 1114: it starts Vulkan inside
# DllMain, where Intel's driver cannot initialise), and the app then crashes
# as it exits. That fix (3fa9ed6) and those for issues 18-21 came after 5.1.1:
# SSIMULACRA2 far too high on NVIDIA GPUs, whose Vulkan driver miscompiled a
# small two-dimensional array in its blur (97d0dc5); SMPTE 170M/240M primaries;
# the BT.470BG transfer; and 4:1:0 video. 97d0dc5 is to be released as 5.1.2,
# the version it reports. Built the same way, v5.1.1 scores exactly as the
# official release does.
#
# The shaders are the SPIR-V committed in Vship's libvshipSpvShaders.
#
# Needs git, g++ on PATH and a Vulkan driver (vulkan-1.dll is linked by name).
param(
    [string]$VshipCommit = '97d0dc55b273f8f370496d0e206d94413f44e62f',          # 2026-10-03
    [string]$VulkanHeadersCommit = '3c65a01745e4a1134d32b9c2c456472212dba16d',  # 2026-09-25
    [string]$WorkDirectory = (Join-Path $env:TEMP 'vship-vulkan-build')
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$output = Join-Path $projectDirectory 'vmaf_app/tools/vship/vulkan/libvship.dll'

function Invoke-Checked([string]$what, [scriptblock]$command) {
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

function Get-Source([string]$url, [string]$commit, [string]$directory) {
    if (-not (Test-Path (Join-Path $directory '.git'))) {
        Invoke-Checked "Cloning $url" { git clone --quiet --filter=blob:none --no-checkout $url $directory }
    }
    # A clone made for an earlier pin may not have this commit yet.
    git -C $directory cat-file -e "$commit^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) { Invoke-Checked "Fetching $url" { git -C $directory fetch --quiet origin } }
    Invoke-Checked "Checking out $commit" { git -C $directory -c advice.detachedHead=false checkout --quiet --force $commit }
}

New-Item -ItemType Directory -Path $WorkDirectory -Force | Out-Null
$vship = Join-Path $WorkDirectory 'Vship'
$headers = Join-Path $WorkDirectory 'Vulkan-Headers'
Get-Source 'https://codeberg.org/Line-fr/Vship.git' $VshipCommit $vship
Get-Source 'https://github.com/KhronosGroup/Vulkan-Headers.git' $VulkanHeadersCommit $headers

Push-Location $vship
try {
    # The SPIR-V shaders, embedded as a C++ header (Makefile: shaderEmbedder).
    Invoke-Checked 'Building the shader embedder' {
        g++ src/Vulkan/spvFileToCppHeader.cpp -std=c++17 -O2 -static -o shaderEmbedder.exe
    }
    Invoke-Checked 'Embedding the shaders' { .\shaderEmbedder.exe libvshipSpvShaders include/libvshipSpvShaders.hpp }

    $makefile = Get-Content Makefile
    $version = foreach ($part in 'MAJOR', 'MINOR', 'MINORMINOR') {
        ($makefile | Select-String "^VSHIP_VERSION_$part=(\d+)").Matches[0].Groups[1].Value
    }
    # The Makefile's Vulkan flags. libstdc++, libgcc and winpthreads are linked
    # in, so the DLL needs only vulkan-1.dll and Windows' own runtime. No link
    # time in its header: the same source gives the same file.
    Invoke-Checked 'Building libvship.dll' {
        g++ src/VshipLib.cpp `
            "-DVSHIP_VERSION_MAJOR=$($version[0])" "-DVSHIP_VERSION_MINOR=$($version[1])" `
            "-DVSHIP_VERSION_MINORMINOR=$($version[2])" `
            -std=c++17 -I include -I (Join-Path $headers 'include') -DNDEBUG -O3 -DVULKANBUILD -w `
            -shared -static-libgcc -static-libstdc++ '-Wl,-Bstatic' -lwinpthread '-Wl,-Bdynamic' `
            '-Wl,--no-insert-timestamp' `
            (Join-Path $env:SystemRoot 'System32/vulkan-1.dll') -o $output
    }
}
finally {
    Pop-Location
}
$hash = (Get-FileHash $output -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Vship $($version -join '.') Vulkan (commit $($VshipCommit.Substring(0, 7))): $output"
Write-Host "SHA-256 $hash, $((Get-Item $output).Length) bytes"
