# Builds the FFmpeg the app decodes with in its own processes, into
# resources/tools/ffmpeg:
#   avcodec-63.dll, avutil-61.dll   FFmpeg's libavcodec and libavutil, which
#                                   the software frame decoder (native/
#                                   software_frames.cpp) decodes with
#   avformat-63.dll, swresample-7.dll, swscale-10.dll
#                                   what Video Compare's playback in the C++
#                                   app also reads files, converts sound and
#                                   converts pictures with
#   licenses/COPYING.LGPLv2.1.txt   their licence
#   licenses/COPYING.vvdec.txt      VVdeC's licence (BSD-3-Clause-Clear)
# (licenses/LICENSE.winpthreads.txt, the licence of the MinGW threads VVdeC
# runs on, is committed there; this script does not write it.)
# and the headers software_frames.cpp is compiled against into native/ffmpeg.
#
# FFmpeg's release source is pinned by its version and the SHA-256 of its
# archive. It is configured LGPL (no --enable-gpl, no --enable-nonfree), with
# no external library, program, filter or network: only the decoders, parsers
# and demuxers listed below (the videos and soundtracks the Python app's
# GStreamer played), Direct3D 11's video decoding for H.264, HEVC, VP9, AV1 and
# MPEG-2, NASM's assembly, Windows' threads, and the C runtime's support
# linked in, so the DLLs need only Windows. libavcodec imports libswresample
# (the Opus decoder resamples).
#
# One external library: Fraunhofer HHI's VVC decoder VVdeC, its master
# pinned by commit, with the 21 pull requests of 4KVCD's that VVdeC has not
# merged yet (scripts/patches/vvdec/4kvcd-pull-requests.patch: #430-#434,
# #436-#440, #443, #445, #447, #451-#458, merged in that order; the intra
# prediction PRs' additions to IntraPredX86.h placed whole by hand). It is
# built here as a static library with the same compiler and linked into
# libavcodec (its C++ runtime and threads statically too) as the decoder
# "libvvdec" (scripts/patches/ffmpeg: libvvdec.c, and libvvdec.patch, which
# registers it). FFmpeg's own VVC decoder stays the default for VVC;
# the app asks for libvvdec by name. The DLLs and headers are committed, as
# libvmaf-fast's are: building the app does not run this, only updating them
# does.
#
# Needs MinGW-w64's gcc, nasm, mingw32-make and pkgconf on PATH (WinLibs has
# all four), CMake, and Git for Windows' git and bash (FFmpeg's configure
# script, the patch).
param(
    [string]$Version = '9.0.2',
    [string]$Sha256 = '8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e',
    [string]$Archive = '',  # a copy of the source archive on disk, instead of downloading it
    # VVdeC's master on 2026-10-28 (3.3.0-dev), as Brian chose: master, not its last
    # release, with 4KVCD's open pull requests (see above).
    [string]$VvdecCommit = '17e0f2f2089c5c0553f1bfc486b1b7aa551e7819'
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$output = Join-Path $projectDirectory 'resources/tools/ffmpeg'
$headers = Join-Path $projectDirectory 'native/ffmpeg'
# Outside the repository: it normally lives in OneDrive (docs/BUILD.md).
$work = Join-Path $env:LOCALAPPDATA "VideoMetricsLab-build/ffmpeg-$Version"
$decoders = 'h264,hevc,vvc,libvvdec,vp9,av1,mpeg2video,ffv1,prores,' +
    'aac,aac_latm,ac3,eac3,dca,truehd,mlp,flac,opus,vorbis,mp3float,mp2float,alac,wmav1,wmav2,wmapro,' +
    'pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_bluray,pcm_dvd'
$parsers = 'h264,hevc,vvc,vp9,av1,mpegvideo,aac,aac_latm,ac3,dca,mlp,flac,opus,vorbis,mpegaudio'
$demuxers = 'matroska,mov,mpegts,mpegps,avi,mxf,ogg,flv,asf,ivf,obu,h264,hevc,vvc,mpegvideo'
# Both of each: a *_d3d11va2 decoder is built from the *_d3d11va one's file.
$hwaccels = (@('h264', 'hevc', 'vp9', 'av1', 'mpeg2') | ForEach-Object { "${_}_d3d11va,${_}_d3d11va2" }) -join ','
$libraries = @('avcodec-63.dll', 'avformat-63.dll', 'avutil-61.dll', 'swresample-7.dll', 'swscale-10.dll')

# Runs a program with what it writes to stderr passed on as text: when this
# script's output is redirected, Windows PowerShell 5.1 under 'Stop' takes a
# program's first line on stderr for a failure (CMake writes its notices and
# warnings there). Each call's exit code is checked instead.
function Invoke-Program([scriptblock]$Program) {
    $ErrorActionPreference = 'Continue'
    & $Program 2>&1 | ForEach-Object { "$_" }
}

foreach ($tool in @('gcc', 'nasm', 'mingw32-make', 'pkgconf', 'cmake', 'git')) {
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

# VVdeC, as a static library installed under the work folder, where FFmpeg's
# configure finds it through pkg-config: the library only (not its player
# and tests), without link-time optimisation (its archive would hold GCC's
# intermediate code, which libavcodec's link would have to optimise), and
# with its warnings not taken for errors (a newer compiler's new warning
# would stop the build).
$vvdecSource = Join-Path $work 'vvdec'
$vvdecInstall = Join-Path $work 'vvdec-install'
Invoke-Program { git -c advice.detachedHead=false clone --quiet https://github.com/fraunhoferhhi/vvdec.git $vvdecSource }
if ($LASTEXITCODE -ne 0) { throw 'cloning VVdeC failed' }
Invoke-Program { git -C $vvdecSource -c advice.detachedHead=false checkout --quiet $VvdecCommit }
if ($LASTEXITCODE -ne 0) { throw "VVdeC has no commit $VvdecCommit" }
$vvdecVersion = (& git -C $vvdecSource describe --tags --always).Trim() + '+4kvcd-prs'
Invoke-Program { git -C $vvdecSource apply (Join-Path $PSScriptRoot 'patches/vvdec/4kvcd-pull-requests.patch') }
if ($LASTEXITCODE -ne 0) { throw "applying 4KVCD's pull requests to VVdeC failed" }
Invoke-Program {
    cmake -S $vvdecSource -B (Join-Path $vvdecSource 'build') -G 'MinGW Makefiles' -DCMAKE_BUILD_TYPE=Release `
        -DBUILD_SHARED_LIBS=OFF -DVVDEC_LIBRARY_ONLY=ON -DVVDEC_ENABLE_LINK_TIME_OPT=OFF -DVVDEC_ENABLE_WERROR=OFF `
        -DVVDEC_TOPLEVEL_OUTPUT_DIRS=OFF "-DCMAKE_INSTALL_PREFIX=$vvdecInstall"
}
if ($LASTEXITCODE -ne 0) { throw 'configuring VVdeC failed' }
Invoke-Program { cmake --build (Join-Path $vvdecSource 'build') --target install -j ([Environment]::ProcessorCount) }
if ($LASTEXITCODE -ne 0) { throw 'building VVdeC failed' }

# The decoder libvvdec in FFmpeg's source, and the name the app logs for it.
$ffmpegSource = Join-Path $work "ffmpeg-$Version"
$patches = Join-Path $PSScriptRoot 'patches/ffmpeg'
Invoke-Program { git -C $ffmpegSource apply (Join-Path $patches 'libvvdec.patch') }
if ($LASTEXITCODE -ne 0) { throw 'patching FFmpeg for libvvdec failed' }
Copy-Item (Join-Path $patches 'libvvdec.c') (Join-Path $ffmpegSource 'libavcodec/libvvdec.c')
[IO.File]::WriteAllText((Join-Path $ffmpegSource 'libavcodec/libvvdec_build.h'),
    "#define LIBVVDEC_BUILD `"VVdeC $vvdecVersion`"`n")  # its last tag, commits since, commit

# configure, make and install, in bash; TMPDIR as a POSIX path (configure
# cannot use Windows' %TEMP%). pkg-config finds VVdeC from the environment,
# not from configure's line, which is kept in the libraries.
$script = @"
set -e
# Compiler warnings on stdout: Windows PowerShell stops at a native
# command's stderr when its output is redirected.
exec 2>&1
cd "`$(cygpath -u '$work')/ffmpeg-$Version"
mkdir -p ../tmp
export TMPDIR="`$(cygpath -u '$work')/tmp"
# A fixed time for the DLLs' headers: the linker writes the time it links at
# otherwise (--no-insert-timestamp notwithstanding), and the same source
# would give other DLLs at each build.
export SOURCE_DATE_EPOCH=0
export PKG_CONFIG_PATH="`$(cygpath -u '$vvdecInstall')/lib/pkgconfig"
# The prefix is a neutral path, installed under a staging folder: the
# configure line is in the libraries (avcodec_configuration), and a path of
# the PC that built them, with its user's name, would be too.
./configure --prefix=/ffmpeg --target-os=mingw32 --arch=x86_64 \
  --enable-shared --disable-static --disable-programs --disable-doc --disable-avdevice \
  --disable-avfilter --disable-network --disable-autodetect --disable-everything \
  --enable-decoder=$decoders --enable-parser=$parsers --enable-demuxer=$demuxers --enable-protocol=file \
  --enable-d3d11va --enable-hwaccel=$hwaccels --enable-w32threads --x86asmexe=nasm \
  --enable-libvvdec --pkg-config=pkgconf --pkg-config-flags=--static \
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
foreach ($library in $libraries) {
    Copy-Item (Join-Path $install "bin/$library") $output -Force
}
$licence = Join-Path $work "ffmpeg-$Version/COPYING.LGPLv2.1"
Copy-Item $licence (Join-Path $output 'licenses/COPYING.LGPLv2.1.txt') -Force
Copy-Item (Join-Path $vvdecSource 'LICENSE.txt') (Join-Path $output 'licenses/COPYING.vvdec.txt') -Force

# Each DLL needs only Windows and the others: VVdeC's C++ runtime and threads
# linked in, not MinGW's DLLs, which this PC has on PATH but a user's does not.
foreach ($library in $libraries) {
    $imports = @(& objdump -p (Join-Path $output $library) | Select-String 'DLL Name: (.+)' |
        ForEach-Object { $_.Matches[0].Groups[1].Value.Trim() })
    $foreign = @($imports | Where-Object {
        $_ -notmatch '^(KERNEL32|ADVAPI32|USER32|SHELL32|ole32|OLEAUT32|bcrypt|ntdll)\.dll$' -and
        $_ -notmatch '^api-ms-win-' -and $_ -notin $libraries })
    if ($foreign) { throw "$library imports $($foreign -join ', '), which Windows does not have" }
}

# The headers software_frames.cpp and the C++ app's playback (cpp/app/
# ffmpeg_api.h) include, as the compiler finds them.
$include = (Resolve-Path (Join-Path $install 'include')).Path.Replace('\', '/')
$native = (Join-Path $projectDirectory 'native').Replace('\', '/')
$playback = (Join-Path $projectDirectory 'cpp/app/ffmpeg_api.h').Replace('\', '/')
# Forward slashes throughout: the list's lines end in " \", the separator.
$listed = & g++ -std=c++17 -MM -I $native -I $include "$native/software_frames.cpp" -x c++-header $playback
if ($LASTEXITCODE -ne 0) { throw 'finding the headers software_frames.cpp and ffmpeg_api.h include failed' }
$used = @(($listed -join ' ') -split '\s+' | Where-Object { $_.StartsWith("$include/") } |
    ForEach-Object { $_.Substring($include.Length + 1) } | Sort-Object -Unique)
if (-not $used) { throw 'software_frames.cpp includes no header of FFmpeg''s' }
if (Test-Path $headers) { Remove-Item -Recurse -Force $headers }
foreach ($header in $used) {
    $target = Join-Path $headers "include/$header"
    New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
    Copy-Item (Join-Path $include $header) $target
}
# The libraries' exports, from which the C++ app's build makes the import
# libraries MSVC links with (cpp/app/CMakeLists.txt).
New-Item -ItemType Directory -Path (Join-Path $headers 'lib') -Force | Out-Null
foreach ($library in $libraries) {
    $name = [IO.Path]::GetFileNameWithoutExtension($library)  # avcodec-63
    $folder = 'lib' + ($name -replace '-\d+$', '')             # libavcodec
    Copy-Item (Join-Path $work "ffmpeg-$Version/$folder/$name.def") (Join-Path $headers "lib/$name.def")
}
Copy-Item $licence (Join-Path $headers 'LICENSE.ffmpeg.txt') -Force

Write-Host "FFmpeg $Version's libraries in $output, $($used.Count) headers in $headers"
foreach ($library in $libraries) {
    $file = Join-Path $output $library
    Write-Host ("  {0}  {1:N0} bytes  SHA-256 {2}" -f $library, (Get-Item $file).Length,
        (Get-FileHash $file -Algorithm SHA256).Hash.ToLowerInvariant())
}
