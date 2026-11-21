# Builds splot, the AV2 decoder the app decodes AV2 with (the software frame
# decoder, native/software_frames.cpp, the AV2 probe and Video Compare's
# playback), into resources/tools/ffmpeg beside FFmpeg's libraries and dav1d,
# where the app looks for it (splot_library_path in gpu_frames.cpp):
#   splot.dll                     for CPUs that run x86-64-v3 code (AVX2: Intel
#                                 since 2013, AMD since 2015)
#   splot-x86-64-v2.dll           the same for older CPUs (SSE4.2), slower
#   licenses/LICENSE.splot.md     its licence
#
# splot (github.com/splotlabs/splot, an AV2 decoder in Rust) has no release
# yet, and AV2 is not final: its source is pinned by a commit (the hash fixes
# every byte of the source). Its C API has what the app needs beyond
# decoding: a setting to leave the film grain out (the app scores a video with
# or without it), and a flag saying whether a picture carries a film grain
# model (the AV2 probe reports it).
# splot picks its SIMD code when it is compiled, not when it runs, so a DLL
# built for AVX2 would crash on a CPU without it: hence the two DLLs, the app
# choosing by the CPU (platform::cpu_runs_x86_64_v3). Both are built with
# the C runtime linked in, so they need only Windows. The DLLs are
# committed: building the app does not run this, only updating splot does.
#
# Needs Git and rustup (the commit's rust-toolchain.toml names the Rust
# version, which rustup installs on first use) with the MSVC toolchain.
param(
    [string]$Commit = '19ba2a57e89b1176f9d78c3bdbbf6127e80ea004'
)
$ErrorActionPreference = 'Stop'

# Runs a program, and throws when it fails. Windows PowerShell turns each
# line a program writes to stderr into an error, which 'Stop' would make
# fatal, and Git and Cargo write progress there: only the exit code means
# anything.
function Invoke-Program([string]$What, [scriptblock]$Command) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command 2>&1 | ForEach-Object { "    $_" }
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

$projectDirectory = Split-Path $PSScriptRoot -Parent
$output = Join-Path $projectDirectory 'resources/tools/ffmpeg'
# Outside the repository: it normally lives in OneDrive (docs/BUILD.md).
$work = Join-Path $env:LOCALAPPDATA "VideoMetricsLab-build/splot-$($Commit.Substring(0, 8))"

foreach ($tool in @('git', 'cargo')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is not on PATH" }
}

if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null
$source = Join-Path $work 'splot'
Invoke-Program 'cloning splot' {
    git clone --quiet -c core.autocrlf=false https://github.com/splotlabs/splot.git $source
}
Invoke-Program "checking out splot $Commit" { git -C $source -c advice.detachedHead=false checkout --quiet $Commit }
$head = (& git -C $source rev-parse HEAD).Trim()
if ($head -ne $Commit) { throw "splot checked out $head, expected $Commit" }

New-Item -ItemType Directory -Path (Join-Path $output 'licenses') -Force | Out-Null
# RUSTFLAGS replaces the repository's own flags for x86-64 (.cargo/config.toml),
# so they are repeated here with the CPU level changed: x86-64-v3 is theirs.
$variants = @(
    @{ Name = 'splot.dll'; Cpu = 'x86-64-v3' },
    @{ Name = 'splot-x86-64-v2.dll'; Cpu = 'x86-64-v2' }
)
try {
    foreach ($variant in $variants) {
        $env:RUSTFLAGS = "-C target-cpu=$($variant.Cpu) --cfg disable_dispatch_avx512 -C target-feature=+crt-static"
        $env:CARGO_TARGET_DIR = Join-Path $work "target-$($variant.Cpu)"
        Push-Location $source
        try {
            Invoke-Program "building splot for $($variant.Cpu)" { cargo build --release --locked -p splot-capi }
        } finally {
            Pop-Location
        }
        $built = Join-Path $env:CARGO_TARGET_DIR 'release/splot_capi.dll'
        if (-not (Test-Path $built)) { throw "the build for $($variant.Cpu) made no splot_capi.dll" }
        Copy-Item $built (Join-Path $output $variant.Name) -Force
    }
} finally {
    Remove-Item Env:RUSTFLAGS, Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
}
Copy-Item (Join-Path $source 'LICENSE.md') (Join-Path $output 'licenses/LICENSE.splot.md') -Force

Write-Host "splot $Commit in $output"
foreach ($variant in $variants) {
    $file = Join-Path $output $variant.Name
    Write-Host ("  {0}  {1:N0} bytes  SHA-256 {2}" -f $variant.Name, (Get-Item $file).Length,
        (Get-FileHash $file -Algorithm SHA256).Hash.ToLowerInvariant())
}
# What they import, said for the record: Windows' own libraries only.
if (Get-Command objdump -ErrorAction SilentlyContinue) {
    foreach ($variant in $variants) {
        $file = Join-Path $output $variant.Name
        $imports = & objdump -p $file | Select-String -Pattern 'DLL Name: (.+)$' |
            ForEach-Object { $_.Matches[0].Groups[1].Value }
        Write-Host "  $($variant.Name) imports $($imports -join ', ')"
    }
}
