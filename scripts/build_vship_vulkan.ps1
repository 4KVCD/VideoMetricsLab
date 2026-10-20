# Builds Vship-fast's Vulkan library (resources/tools/vship/vulkan/libvship.dll)
# from a pinned commit with the MinGW-w64 g++ that builds the D3D11 tone
# mapper (WinLibs, UCRT, POSIX threads).
#
# Vship-fast (github.com/4KVCD/Vship-fast) is this project's fork of Vship,
# faster, with metrics that compute as libjxl 0.12.0's and pycvvdp's do. It
# has no releases: the file follows from the source and this script alone.
#
# The shaders are the SPIR-V committed in its libvshipSpvShaders, embedded by
# its own shaderEmbedder.
#
# Needs git, g++ on PATH and a Vulkan driver (vulkan-1.dll is linked by name).
param(
    [string]$VshipCommit = '77edf23ce21bb3161b20cf2db10a997f7b5b21cc',          # Vship-fast perf-metrics, 2026-10-20
    [string]$VulkanHeadersCommit = '3c65a01745e4a1134d32b9c2c456472212dba16d',  # 2026-09-25
    [string]$WorkDirectory = (Join-Path $env:TEMP 'vship-vulkan-build')
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$output = Join-Path $projectDirectory 'resources/tools/vship/vulkan/libvship.dll'

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
$vship = Join-Path $WorkDirectory 'Vship-fast'
$headers = Join-Path $WorkDirectory 'Vulkan-Headers'
Get-Source 'https://github.com/4KVCD/Vship-fast.git' $VshipCommit $vship
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
    # The Makefile's Vulkan flags, its -ffp-contract=off among them (the host's
    # float arithmetic rounded as written). libstdc++, libgcc and winpthreads
    # are linked in, so the DLL needs only vulkan-1.dll and Windows' own
    # runtime. There is no link time in its header and no image base of ld's
    # choosing, so the same source gives the same file.
    # Don't let ld choose the image base: it hashes the output file's path
    # into it, so a build in another folder gives another file. Windows
    # relocates the DLL wherever it loads it anyway.
    Invoke-Checked 'Building libvship.dll' {
        g++ src/VshipLib.cpp `
            "-DVSHIP_VERSION_MAJOR=$($version[0])" "-DVSHIP_VERSION_MINOR=$($version[1])" `
            "-DVSHIP_VERSION_MINORMINOR=$($version[2])" `
            -std=c++17 -I include -I (Join-Path $headers 'include') -DNDEBUG -O3 -DVULKANBUILD -ffp-contract=off -w `
            -shared -static-libgcc -static-libstdc++ '-Wl,-Bstatic' -lwinpthread '-Wl,-Bdynamic' `
            '-Wl,--no-insert-timestamp' '-Wl,--disable-auto-image-base' `
            (Join-Path $env:SystemRoot 'System32/vulkan-1.dll') -o $output
    }
}
finally {
    Pop-Location
}
$hash = (Get-FileHash $output -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Vship-fast $($version -join '.') Vulkan (commit $($VshipCommit.Substring(0, 7))): $output"
Write-Host "SHA-256 $hash, $((Get-Item $output).Length) bytes"
