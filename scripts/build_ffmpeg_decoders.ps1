# Builds the FFmpeg the software frame decoder (native/software_frames.cpp)
# decodes with, into vmaf_app/tools/ffmpeg:
#   avcodec-63.dll, avutil-61.dll   FFmpeg's libavcodec and libavutil, with
#                                   only the decoders the app asks for
#   licenses/COPYING.LGPLv2.1.txt   their licence
# and the headers software_frames.cpp is compiled against into native/ffmpeg.
#
# FFmpeg's release source is pinned by its version and the SHA-256 of its
# archive. It is configured LGPL (no --enable-gpl, no --enable-nonfree), with
# no external library, program, demuxer, filter or scaler: the six decoders,
# NASM's assembly, Windows' threads, and the C runtime's support linked in, so
# the DLLs need only Windows. The DLLs and headers are committed, as
# libvmaf-fast's are: building the app does not run this, only updating them
# does.
#
# Needs MinGW-w64's gcc, nasm and mingw32-make on PATH (WinLibs has all
# three) and Git for Windows' bash for FFmpeg's configure script.
param(
    [string]$Version = '9.0.2',
    [string]$Sha256 = '8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e',
    [string]$Archive = ''  # a copy of the source archive on disk, instead of downloading it
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$output = Join-Path $projectDirectory 'vmaf_app/tools/ffmpeg'
$headers = Join-Path $projectDirectory 'native/ffmpeg'
# Outside the repository: it normally lives in OneDrive (docs/BUILD.md).
$work = Join-Path $env:LOCALAPPDATA "VideoMetricsLab-build/ffmpeg-$Version"
$decoders = 'h264,hevc,vvc,vp9,mpeg2video,ffv1'

foreach ($tool in @('gcc', 'nasm', 'mingw32-make')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is not on PATH" }
}
$bash = (Get-Command bash -ErrorAction SilentlyContinue).Source
if (-not $bash) { $bash = Join-Path $env:ProgramFiles 'Git/bin/bash.exe' }
if (-not (Test-Path $bash)) { throw "Git for Windows' bash is needed for FFmpeg's configure" }

if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null
if (-not $Archive) {
    $Archive = Join-Path $work "ffmpeg-$Version.tar.xz"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://ffmpeg.org/releases/ffmpeg-$Version.tar.xz" -OutFile $Archive -UseBasicParsing
}
$hash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $Sha256) { throw "ffmpeg-$Version.tar.xz has SHA-256 $hash, expected $Sha256" }
& tar -xf $Archive -C $work
if ($LASTEXITCODE -ne 0) { throw "unpacking ffmpeg-$Version.tar.xz failed" }

# configure, make and install, in bash; TMPDIR as a POSIX path (configure
# cannot use Windows' %TEMP%).
$script = @"
set -e
# Compiler warnings on stdout: Windows PowerShell stops at a native
# command's stderr when its output is redirected.
exec 2>&1
cd "`$(cygpath -u '$work')/ffmpeg-$Version"
mkdir -p ../tmp
export TMPDIR="`$(cygpath -u '$work')/tmp"
# The prefix is a neutral path, installed under a staging folder: the
# configure line is in the libraries (avcodec_configuration), and a path of
# the PC that built them, with its user's name, would be too.
./configure --prefix=/ffmpeg --target-os=mingw32 --arch=x86_64 \
  --enable-shared --disable-static --disable-programs --disable-doc --disable-avdevice --disable-avformat \
  --disable-avfilter --disable-swscale --disable-swresample --disable-network --disable-autodetect \
  --disable-everything --enable-decoder=$decoders --enable-w32threads --x86asmexe=nasm \
  --extra-ldflags='-static-libgcc -static -Wl,--no-insert-timestamp'
mingw32-make -j`$(nproc)
mingw32-make install DESTDIR="`$(cygpath -u '$work')/stage"
"@
# As a file: Windows PowerShell 5.1 mangles quotes in a native command's arguments.
$scriptFile = Join-Path $work 'build.sh'
[IO.File]::WriteAllText($scriptFile, ($script -replace "`r", ''))
& $bash $scriptFile.Replace('\', '/')
if ($LASTEXITCODE -ne 0) { throw "building FFmpeg $Version failed" }

$install = Join-Path $work 'stage/ffmpeg'
New-Item -ItemType Directory -Path (Join-Path $output 'licenses') -Force | Out-Null
foreach ($library in @('avcodec-63.dll', 'avutil-61.dll')) {
    Copy-Item (Join-Path $install "bin/$library") $output -Force
}
$licence = Join-Path $work "ffmpeg-$Version/COPYING.LGPLv2.1"
Copy-Item $licence (Join-Path $output 'licenses/COPYING.LGPLv2.1.txt') -Force

# The headers software_frames.cpp includes, as the compiler finds them.
$include = (Resolve-Path (Join-Path $install 'include')).Path.Replace('\', '/')
$native = (Join-Path $projectDirectory 'native').Replace('\', '/')
# Forward slashes throughout: the list's lines end in " \", the separator.
$listed = & g++ -std=c++17 -MM -I $native -I $include "$native/software_frames.cpp"
if ($LASTEXITCODE -ne 0) { throw 'finding the headers software_frames.cpp includes failed' }
$used = @(($listed -join ' ') -split '\s+' | Where-Object { $_.StartsWith("$include/") } |
    ForEach-Object { $_.Substring($include.Length + 1) } | Sort-Object -Unique)
if (-not $used) { throw 'software_frames.cpp includes no header of FFmpeg''s' }
if (Test-Path $headers) { Remove-Item -Recurse -Force $headers }
foreach ($header in $used) {
    $target = Join-Path $headers "include/$header"
    New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
    Copy-Item (Join-Path $include $header) $target
}
Copy-Item $licence (Join-Path $headers 'LICENSE.ffmpeg.txt') -Force

Write-Host "FFmpeg $Version's decoders ($decoders) in $output, $($used.Count) headers in $headers"
foreach ($library in @('avcodec-63.dll', 'avutil-61.dll')) {
    $file = Join-Path $output $library
    Write-Host ("  {0}  {1:N0} bytes  SHA-256 {2}" -f $library, (Get-Item $file).Length,
        (Get-FileHash $file -Algorithm SHA256).Hash.ToLowerInvariant())
}
