# Builds Vship's Vulkan library (vmaf_app/tools/vship/vulkan/libvship.dll)
# from a pinned commit -- the v5.1.2 release's tag -- with the MinGW-w64 g++
# that builds the D3D11 tone mapper (WinLibs, UCRT, POSIX threads).
#
# Built rather than taken from the release, as the bundle has been since
# 5.1.2's fixes were only on Vship's main branch (5.1.1's Vulkan build could
# not be loaded on a PC whose only GPU is Intel's, and scored SSIMULACRA2 far
# too high on NVIDIA GPUs): the file follows from the source and this script
# alone. Built this way, v5.1.2 scores Butteraugli and CVVDP exactly as the
# release's own Vulkan library does, and SSIMULACRA2 within 0.00003 (the
# rounding of another compiler's build; measured 2026-10-06 on three videos).
#
# The shaders are the SPIR-V committed in Vship's libvshipSpvShaders.
#
# Needs git, g++ on PATH and a Vulkan driver (vulkan-1.dll is linked by name).
param(
    [string]$VshipCommit = '5a627933274196219f782b3b74e13fad47cd6370',          # v5.1.2, 2026-10-06
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
    # time in its header, and no image base of ld's choosing -- ld hashes the
    # output file's path into it, so the same source built in another folder
    # gave another file (Windows relocates the DLL wherever it loads it
    # anyway): the same source gives the same file.
    Invoke-Checked 'Building libvship.dll' {
        g++ src/VshipLib.cpp `
            "-DVSHIP_VERSION_MAJOR=$($version[0])" "-DVSHIP_VERSION_MINOR=$($version[1])" `
            "-DVSHIP_VERSION_MINORMINOR=$($version[2])" `
            -std=c++17 -I include -I (Join-Path $headers 'include') -DNDEBUG -O3 -DVULKANBUILD -w `
            -shared -static-libgcc -static-libstdc++ '-Wl,-Bstatic' -lwinpthread '-Wl,-Bdynamic' `
            '-Wl,--no-insert-timestamp' '-Wl,--disable-auto-image-base' `
            (Join-Path $env:SystemRoot 'System32/vulkan-1.dll') -o $output
    }
}
finally {
    Pop-Location
}
$hash = (Get-FileHash $output -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Vship $($version -join '.') Vulkan (commit $($VshipCommit.Substring(0, 7))): $output"
Write-Host "SHA-256 $hash, $((Get-Item $output).Length) bytes"
