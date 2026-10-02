# Builds Vship's Vulkan library (vmaf_app/tools/vship/vulkan/libvship.dll)
# from a pinned commit, with the MinGW-w64 g++ that builds the D3D11 tone
# mapper (WinLibs, UCRT, POSIX threads).
#
# Why a commit and not a release: Vship 5.1.1's Vulkan build cannot be loaded
# on a PC whose only GPU is Intel's (WinError 1114: it starts Vulkan inside
# DllMain, where Intel's driver cannot initialise), and the app then crashes
# as it exits. The fix, 3fa9ed6, and the SMPTE 170M/240M, BT.470BG and 4:1:0
# fixes came after 5.1.1 with no release yet. Built the same way, v5.1.1
# scores exactly as the official release does.
#
# One shader is patched: vship_ssimulacra2_nvidia.patch (beside this script)
# works around NVIDIA's Vulkan driver miscompiling SSIMULACRA2's blur, which
# scored it far too high on NVIDIA GPUs (Vship issue 18). That shader is
# compiled here with a pinned Slang release, downloaded and checked by its
# SHA-256; the others are the SPIR-V committed in Vship's libvshipSpvShaders.
#
# Needs git, g++ on PATH and a Vulkan driver (vulkan-1.dll is linked by name).
param(
    [string]$VshipCommit = '0732ed3cb81c696a017f51d5f327a66b18cebdcd',          # 2026-09-28
    [string]$VulkanHeadersCommit = '3c65a01745e4a1134d32b9c2c456472212dba16d',  # 2026-09-25
    [string]$SlangVersion = '2026.10.2',                                         # 2026-06-02
    [string]$SlangSha256 = 'f21fca4ba78bfb366ef3b282b4926e3efca61c2ffd72a482b024c9f8413331d0',
    [string]$WorkDirectory = (Join-Path $env:TEMP 'vship-vulkan-build')
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$output = Join-Path $projectDirectory 'vmaf_app/tools/vship/vulkan/libvship.dll'
$patch = Join-Path $PSScriptRoot 'vship_ssimulacra2_nvidia.patch'

function Invoke-Checked([string]$what, [scriptblock]$command) {
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

function Get-Source([string]$url, [string]$commit, [string]$directory) {
    if (-not (Test-Path (Join-Path $directory '.git'))) {
        Invoke-Checked "Cloning $url" { git clone --quiet --filter=blob:none --no-checkout $url $directory }
    }
    Invoke-Checked "Checking out $commit" { git -C $directory -c advice.detachedHead=false checkout --quiet --force $commit }
}

function Get-Slangc([string]$version, [string]$sha256, [string]$directory) {
    $name = "slang-$version-windows-x86_64"
    $slangc = Join-Path $directory "$name/bin/slangc.exe"
    if (Test-Path $slangc) { return $slangc }
    $zip = Join-Path $directory "$name.zip"
    if (-not (Test-Path $zip)) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -OutFile $zip `
            "https://github.com/shader-slang/slang/releases/download/v$version/$name.zip"
    }
    $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $sha256) { Remove-Item $zip; throw "$name.zip has SHA-256 $actual, expected $sha256" }
    Expand-Archive $zip (Join-Path $directory $name) -Force
    return $slangc
}

New-Item -ItemType Directory -Path $WorkDirectory -Force | Out-Null
$vship = Join-Path $WorkDirectory 'Vship'
$headers = Join-Path $WorkDirectory 'Vulkan-Headers'
Get-Source 'https://codeberg.org/Line-fr/Vship.git' $VshipCommit $vship
Get-Source 'https://github.com/KhronosGroup/Vulkan-Headers.git' $VulkanHeadersCommit $headers
$slangc = Get-Slangc $SlangVersion $SlangSha256 $WorkDirectory

Push-Location $vship
try {
    # The checkout above reset the shader source, so the patch applies afresh.
    Invoke-Checked 'Applying the SSIMULACRA2 patch' { git apply $patch }
    # The Makefile's command for this shader (its "shaders" target).
    Invoke-Checked 'Compiling the patched SSIMULACRA2 shader' {
        & $slangc src/Vulkan/ssimu2/shaders/scoreSSIMU2.slang -O2 -target spirv -profile spirv_1_3 `
            -emit-spirv-directly -fvk-use-entrypoint-name -entry planescale_map -o libvshipSpvShaders/scoreSSIMU2.spv
    }
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
    # in, so the DLL needs only vulkan-1.dll and Windows' own runtime.
    Invoke-Checked 'Building libvship.dll' {
        g++ src/VshipLib.cpp `
            "-DVSHIP_VERSION_MAJOR=$($version[0])" "-DVSHIP_VERSION_MINOR=$($version[1])" `
            "-DVSHIP_VERSION_MINORMINOR=$($version[2])" `
            -std=c++17 -I include -I (Join-Path $headers 'include') -DNDEBUG -O3 -DVULKANBUILD -w `
            -shared -static-libgcc -static-libstdc++ '-Wl,-Bstatic' -lwinpthread '-Wl,-Bdynamic' `
            (Join-Path $env:SystemRoot 'System32/vulkan-1.dll') -o $output
    }
}
finally {
    Pop-Location
}
$hash = (Get-FileHash $output -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Vship $($version -join '.') Vulkan (commit $($VshipCommit.Substring(0, 7)), SSIMULACRA2 patched): $output"
Write-Host "SHA-256 $hash, $((Get-Item $output).Length) bytes"
