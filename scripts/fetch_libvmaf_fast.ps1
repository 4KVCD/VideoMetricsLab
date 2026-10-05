# Fetches libvmaf-fast (https://github.com/4KVCD/libvmaf-fast), the fork of
# Netflix's libvmaf the app's GPU VMAF comes from, into vmaf_app/tools:
#   libvmaf/libvmaf.dll          libvmaf with CUDA: VMAF and VMAF NEG on NVIDIA
#                                GPUs, and the libvmaf every GPU scorer
#                                predicts with and VMAF v1's CAMBI and SpEED
#   vmaf_vulkan/vmaf_vulkan.dll  VMAF's features with Vulkan, and VMAF v1's
#                                ADM3 and motion3
# each with its licences.
#
# The release is pinned by its version and the SHA-256 of its archive, and
# every file in it is checked against the archive's SHA256SUMS as well. The
# DLLs are committed, as Vship's are: building the app does not run this,
# only updating them does. Building them instead is the fork's
# fast/scripts (see its fast/README.md).
param(
    [string]$Version = '3.2.0-fast.1',
    [string]$Sha256 = '3ad8ab2b0fd26f0c89389d55e086f4cad1d0e059d63845ecf88e5aaed6c38f9c',
    [string]$Archive = ''  # a copy of the release archive on disk, instead of downloading it
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$tools = Join-Path $projectDirectory 'vmaf_app/tools'
$name = "libvmaf-fast-$Version-windows-x64"
$work = Join-Path $env:TEMP "fetch-$name"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null
try {
    if (-not $Archive) {
        $Archive = Join-Path $work "$name.zip"
        $url = "https://github.com/4KVCD/libvmaf-fast/releases/download/v$Version/$name.zip"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $Archive -UseBasicParsing
    }
    $hash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $Sha256) { throw "$name.zip has SHA-256 $hash, expected $Sha256" }

    $unpacked = Join-Path $work 'unpacked'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Archive, $unpacked)
    # Every file against the archive's own sums: what is installed is what
    # was released, not only what was downloaded.
    $listed = @{}
    foreach ($line in Get-Content (Join-Path $unpacked 'SHA256SUMS')) {
        if ($line -match '^([0-9a-f]{64})  (.+)$') { $listed[$matches[2]] = $matches[1] }
    }
    $libraries = 'libvmaf/libvmaf.dll', 'vmaf_vulkan/vmaf_vulkan.dll'
    foreach ($file in $libraries) {
        if (-not $listed.ContainsKey($file)) { throw "$file is not in the release" }
    }
    foreach ($entry in $listed.GetEnumerator()) {
        $actual = (Get-FileHash (Join-Path $unpacked $entry.Key) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $entry.Value) { throw "$($entry.Key) does not match the release's SHA256SUMS" }
    }

    # Replaced whole, licences included: nothing of an older release stays.
    foreach ($folder in 'libvmaf', 'vmaf_vulkan') {
        $target = Join-Path $tools $folder
        if (Test-Path $target) { Remove-Item -Recurse -Force $target }
        Copy-Item -Recurse (Join-Path $unpacked $folder) $target
    }
    Write-Host "libvmaf-fast $Version into $tools"
    Get-Content (Join-Path $unpacked 'BUILD.txt') | Select-Object -Skip 1 -First 2 | ForEach-Object { Write-Host "  $_" }
    foreach ($file in $libraries) { Write-Host "  $file SHA-256 $($listed[$file])" }
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
