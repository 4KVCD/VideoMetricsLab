# Builds dav1d, the AV1 decoder the app decodes AV1 with in software (the
# software frame decoder, native/software_frames.cpp, and Video Compare's
# playback, cpp/app/native_dav1d.cpp), into resources/tools/ffmpeg beside
# FFmpeg's libraries, where the app looks for it:
#   dav1d.dll                     VideoLAN's dav1d
#   licenses/COPYING.dav1d.txt    its licence (BSD 2-clause)
#
# dav1d's release source is pinned by its version and the SHA-256 of its
# archive, from VideoLAN's own server. It is built with MinGW-w64's gcc and
# NASM's assembly, without dav1d's tools, tests or examples, and with the C
# runtime's support linked in, so the DLL needs only Windows. AV1's decoding
# is defined exactly, so any dav1d release decodes the same pictures. The DLL
# is committed, as FFmpeg's are: building the app does not run this, only
# updating dav1d does.
#
# Needs MinGW-w64's gcc, NASM and Ninja on PATH (WinLibs has all three), and
# Python for Meson, which is installed, pinned, in a folder of its own
# outside the repository.
param(
    [string]$Version = '1.5.4',
    [string]$Sha256 = '686616b7c69eb88d44459391ab25cac13b6647a3b288835c5784e71c1514a5c5',
    [string]$MesonVersion = '1.12.1',
    # The Python that Meson's own folder is made with.
    [string]$Python = '',
    [string]$Archive = ''  # a copy of the source archive on disk, instead of downloading it
)
$ErrorActionPreference = 'Stop'

# Runs a program, and throws when it fails. Windows PowerShell turns each
# line a program writes to stderr into an error, which 'Stop' would make
# fatal, and Python, Meson and Ninja write warnings there: only the exit
# code means anything.
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
$work = Join-Path $env:LOCALAPPDATA "VideoMetricsLab-build/dav1d-$Version"

foreach ($tool in @('gcc', 'nasm', 'ninja')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is not on PATH" }
}
if (-not $Python) { $Python = Join-Path $projectDirectory '.venv/Scripts/python.exe' }
if (-not (Test-Path $Python)) { throw "no Python at $Python for Meson: name one with -Python" }

if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null
if (-not $Archive) {
    $Archive = Join-Path $work "dav1d-$Version.tar.xz"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://downloads.videolan.org/pub/videolan/dav1d/$Version/dav1d-$Version.tar.xz" `
        -OutFile $Archive -UseBasicParsing
}
$hash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $Sha256) { throw "dav1d-$Version.tar.xz has SHA-256 $hash, expected $Sha256" }
Invoke-Program "unpacking dav1d-$Version.tar.xz" { tar -xf $Archive -C $work }

# Meson, pinned, in a folder of its own.
$tools = Join-Path $work 'meson'
Invoke-Program 'making the folder for Meson' { & $Python -m venv $tools }
Invoke-Program "installing Meson $MesonVersion" {
    & (Join-Path $tools 'Scripts/python.exe') -m pip install --quiet --disable-pip-version-check "meson==$MesonVersion"
}
$meson = Join-Path $tools 'Scripts/meson.exe'

# A fixed time for the DLL's header: the linker writes the time it links at
# otherwise, and the same source would give another DLL at each build.
$env:SOURCE_DATE_EPOCH = '0'
$env:CC = 'gcc'
$env:LDFLAGS = '-static-libgcc -static -Wl,--no-insert-timestamp'
$source = Join-Path $work "dav1d-$Version"
$build = Join-Path $work 'build'
Invoke-Program "configuring dav1d $Version" {
    & $meson setup $build $source --buildtype=release --default-library=shared -Db_ndebug=true `
        -Denable_tools=false -Denable_tests=false -Denable_examples=false
}
Invoke-Program "building dav1d $Version" { ninja -C $build }

$built = Get-ChildItem -Path $build -Recurse -Filter 'libdav1d*.dll' | Select-Object -First 1
if (-not $built) { throw 'the build made no dav1d DLL' }
New-Item -ItemType Directory -Path (Join-Path $output 'licenses') -Force | Out-Null
$file = Join-Path $output 'dav1d.dll'
Copy-Item $built.FullName $file -Force
Copy-Item (Join-Path $source 'COPYING') (Join-Path $output 'licenses/COPYING.dav1d.txt') -Force

# What it imports, said for the record: Windows' own libraries only, with
# the C runtime's support linked in.
$imports = & objdump -p $file | Select-String -Pattern 'DLL Name: (.+)$' | ForEach-Object { $_.Matches[0].Groups[1].Value }
Write-Host "dav1d $Version in $output, importing $($imports -join ', ')"
Write-Host ("  dav1d.dll  {0:N0} bytes  SHA-256 {1}" -f (Get-Item $file).Length,
    (Get-FileHash $file -Algorithm SHA256).Hash.ToLowerInvariant())
