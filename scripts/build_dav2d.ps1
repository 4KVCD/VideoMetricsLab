# Builds dav2d, the AV2 decoder the app decodes AV2 with (the software frame
# decoder, native/software_frames.cpp, and Video Compare's playback), into
# resources/tools/ffmpeg beside FFmpeg's libraries and dav1d, where the app
# looks for it:
#   dav2d.dll                     VideoLAN's dav2d
#   licenses/COPYING.dav2d.txt    its licence (BSD 2-clause)
#
# dav2d (VideoLAN's AV2 decoder, based on dav1d) has no release yet, and AV2
# is not final: its source is pinned by a commit of VideoLAN's own repository
# (the commit's hash fixes every byte of the source), which decodes the
# streams of AOM's AVM at the av2-normative branch the app's AV2 tests were
# made with. It is built as build_dav1d.ps1 builds dav1d: MinGW-w64's gcc and
# NASM, without dav2d's tools, tests or examples, the C runtime's support
# linked in, so the DLL needs only Windows. The DLL is committed: building
# the app does not run this, only updating dav2d does.
#
# Needs MinGW-w64's gcc, NASM and Ninja on PATH (WinLibs has all three), Git,
# and Python for Meson, which is installed, pinned, in a folder of its own
# outside the repository.
param(
    [string]$Commit = '3dde00d02827fada3bb70c8fad9b2e06f598ea33',
    [string]$MesonVersion = '1.12.1',
    # The Python that Meson's own folder is made with.
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'

# Runs a program, and throws when it fails. Windows PowerShell turns each
# line a program writes to stderr into an error, which 'Stop' would make
# fatal, and Git, Meson and Ninja write progress there: only the exit code
# means anything.
function Invoke-Program([string]$What, [scriptblock]$Command) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command | ForEach-Object { "    $_" }
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

$projectDirectory = Split-Path $PSScriptRoot -Parent
$output = Join-Path $projectDirectory 'resources/tools/ffmpeg'
# Outside the repository: it normally lives in OneDrive (docs/BUILD.md).
$work = Join-Path $env:LOCALAPPDATA "VideoMetricsLab-build/dav2d-$($Commit.Substring(0, 8))"

foreach ($tool in @('gcc', 'nasm', 'ninja', 'git')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is not on PATH" }
}
if (-not $Python) { $Python = Join-Path $projectDirectory '.venv/Scripts/python.exe' }
if (-not (Test-Path $Python)) { throw "no Python at $Python for Meson: name one with -Python" }

if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null
$source = Join-Path $work 'dav2d'
Invoke-Program 'cloning dav2d' { git clone --quiet https://code.videolan.org/videolan/dav2d.git $source }
Invoke-Program "checking out dav2d $Commit" { git -C $source -c advice.detachedHead=false checkout --quiet $Commit }
$head = (& git -C $source rev-parse HEAD).Trim()
if ($head -ne $Commit) { throw "dav2d checked out $head, expected $Commit" }

# Meson, pinned, in a folder of its own.
$tools = Join-Path $work 'meson'
Invoke-Program 'making the folder for Meson' { & $Python -m venv $tools }
Invoke-Program "installing Meson $MesonVersion" {
    & (Join-Path $tools 'Scripts/python.exe') -m pip install --quiet --disable-pip-version-check "meson==$MesonVersion"
}
$meson = Join-Path $tools 'Scripts/meson.exe'

# A fixed time for the DLL's header, as for dav1d.
$env:SOURCE_DATE_EPOCH = '0'
$env:CC = 'gcc'
$env:LDFLAGS = '-static-libgcc -static -Wl,--no-insert-timestamp'
$build = Join-Path $work 'build'
Invoke-Program "configuring dav2d $Commit" {
    & $meson setup $build $source --buildtype=release --default-library=shared -Db_ndebug=true `
        -Denable_tools=false -Denable_tests=false -Denable_examples=false
}
Invoke-Program "building dav2d $Commit" { ninja -C $build }

$built = Get-ChildItem -Path $build -Recurse -Filter 'libdav2d*.dll' | Select-Object -First 1
if (-not $built) { throw 'the build made no dav2d DLL' }
New-Item -ItemType Directory -Path (Join-Path $output 'licenses') -Force | Out-Null
$file = Join-Path $output 'dav2d.dll'
Copy-Item $built.FullName $file -Force
Copy-Item (Join-Path $source 'COPYING') (Join-Path $output 'licenses/COPYING.dav2d.txt') -Force

# What it imports, said for the record: Windows' own libraries only.
$imports = & objdump -p $file | Select-String -Pattern 'DLL Name: (.+)$' | ForEach-Object { $_.Matches[0].Groups[1].Value }
Write-Host "dav2d $Commit in $output, importing $($imports -join ', ')"
Write-Host ("  dav2d.dll  {0:N0} bytes  SHA-256 {1}" -f (Get-Item $file).Length,
    (Get-FileHash $file -Algorithm SHA256).Hash.ToLowerInvariant())
