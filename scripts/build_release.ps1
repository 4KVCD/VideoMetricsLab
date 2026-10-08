# Builds the self-contained Windows distributable.
#
#   .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
#   ./scripts/build_release.ps1
#
# Output goes OUTSIDE the repository by default, under
# %LOCALAPPDATA%\VideoMetricsLab-build. This repository lives in a
# OneDrive folder: building into it would upload ~450 MB on every build, and
# OneDrive's own file handles made the previous build's output impossible to
# delete ("Access is denied" on _internal\...). Pass -OutputRoot to override.
param(
    [string]$OutputRoot = (Join-Path $env:LOCALAPPDATA 'VideoMetricsLab-build'),
    # Videos to decode for real through the packaged GStreamer, in addition
    # to the structural checks that always run. Any files with video and
    # audio will do; the more codecs they cover, the more the build proves.
    [string[]]$VerifyMedia = @(),
    # Builds with a local build of libvmaf-fast, to test; see below.
    [switch]$AllowLocalLibvmaf
)
$ErrorActionPreference = 'Stop'

$projectDirectory = Split-Path $PSScriptRoot -Parent
Push-Location $projectDirectory
try {
    # 0. A release takes a release of libvmaf-fast (fetch_libvmaf_fast.ps1),
    #    published before it, not the local build of the fork the app is
    #    developed with (build_libvmaf_fast_local.ps1): its source is public.
    $libvmafFast = Get-Content -Raw (Join-Path $projectDirectory 'vmaf_app/tools/libvmaf/libvmaf-fast.json') |
        ConvertFrom-Json
    if (-not $libvmafFast.release -and -not $AllowLocalLibvmaf) {
        throw ("libvmaf-fast is a local build ($($libvmafFast.version)): publish libvmaf-fast and install " +
               'its release with scripts/fetch_libvmaf_fast.ps1 (docs/RELEASING.md), or pass ' +
               '-AllowLocalLibvmaf to build one to test')
    }

    $python = Join-Path $projectDirectory '.venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    $workPath = Join-Path $OutputRoot 'work'
    $distPath = Join-Path $OutputRoot 'dist'
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

    # 1. The native libraries, built from this commit's source every time:
    #    the GPU HDR->SDR shader and the GPU frame decoders the GPU metrics
    #    read their frames from (NVIDIA's, Intel's and AMD's). The app runs
    #    without them -- FFmpeg tone maps and decodes instead -- which is
    #    exactly why a release missing one would look fine. They are not in
    #    git, so ones left from another commit would be stale; a release
    #    needs g++ (MinGW-w64) and builds them, or does not build at all.
    if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
        throw 'g++ (MinGW-w64) is needed to build the native libraries; see docs/BUILD.md'
    }
    Write-Host '==> Building d3d11_tonemap.dll' -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'build_d3d11_tonemap.ps1')
    Write-Host '==> Building the GPU frame decoders' -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'build_gpu_frames.ps1')

    # 2. Freeze. --noconfirm so a rebuild does not stop to ask about dist/.
    Write-Host '==> Running PyInstaller' -ForegroundColor Cyan
    # Resolve Windows system DLLs before anything injected into the calling
    # process's PATH.  Development shells can prepend private tool runtimes
    # (Poppler is a common example) that ship an unrelated icuuc.dll with the
    # same name as Windows' ICU compatibility DLL.  If PyInstaller finds that
    # copy first, Qt6Core loads it and QtWidgets fails at startup with a
    # missing-procedure error.
    $system32 = Join-Path $env:SystemRoot 'System32'
    $env:PATH = "$system32;$env:PATH"
    # PyInstaller writes its progress log to stderr. Under
    # $ErrorActionPreference = 'Stop' PowerShell turns each of those lines
    # into a terminating NativeCommandError, so a perfectly successful build
    # "fails" on its first INFO line. Only the exit code means anything here.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python -m PyInstaller --noconfirm `
            --workpath $workPath --distpath $distPath `
            'VideoMetricsLab.spec'
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

    $output = Join-Path $distPath 'VideoMetricsLab'
    $exe = Join-Path $output 'VideoMetricsLab.exe'
    if (-not (Test-Path $exe)) {
        throw 'PyInstaller reported success but produced no executable'
    }
    # Qt uses the Windows ICU compatibility DLL.  It must not be copied into
    # the application root from an unrelated SDK or command-line tool.
    $foreignIcu = Join-Path $output '_internal/icuuc.dll'
    if (Test-Path $foreignIcu) {
        throw "The bundle contains a foreign icuuc.dll ($foreignIcu); refusing to package a Qt runtime that will not start"
    }

    # The GStreamer bundle is pruned to what the app uses (see the spec and
    # scripts/gstreamer_bundle.py). Prove the pruned copy still does all of
    # it, in child interpreters that cannot see the development wheels --
    # otherwise the full installation on this machine would conceal a
    # missing plugin or DLL in the packaged one.
    Write-Host '==> Verifying the packaged GStreamer' -ForegroundColor Cyan
    $verifyArguments = @('--bundle', (Join-Path $output '_internal'))
    foreach ($file in $VerifyMedia) { $verifyArguments += @('--media', $file) }
    & $python (Join-Path $PSScriptRoot 'verify_gstreamer_bundle.py') @verifyArguments
    if ($LASTEXITCODE -ne 0) { throw 'The packaged GStreamer failed verification' }

    # 3. Prove it runs and can find everything it does not contain. A build
    #    that starts and then silently falls back to software playback looks
    #    identical to a good one until someone plays a video.
    Write-Host '==> Self-test' -ForegroundColor Cyan
    # Start-Process -Wait, not `& $exe`: this is a windowed executable, and
    # PowerShell does not wait for those. Called directly, the script read
    # the previous build's report and a stale exit code while the new
    # executable was still starting.
    $report = Join-Path $env:USERPROFILE '.videometricslab/self-test.txt'
    if (Test-Path $report) { Remove-Item $report -Force }
    $selfTest = (Start-Process -FilePath $exe -ArgumentList '--self-test', '--quiet' -Wait -PassThru).ExitCode
    if (Test-Path $report) {
        Get-Content $report | ForEach-Object { "    $_" }
    } else {
        throw 'The self-test wrote no report; the executable did not get as far as running it'
    }
    if ($selfTest -ne 0) {
        throw 'The self-test reported a failure (see above)'
    }

    # The command line: the second program in the folder. It must start,
    # find FFmpeg and say what calculates each metric (the GPU libraries
    # load and are probed in a process of its own, as in a comparison).
    Write-Host '==> Command line' -ForegroundColor Cyan
    $cli = Join-Path $output 'VideoMetricsLab-cli.exe'
    if (-not (Test-Path $cli)) { throw 'The build has no VideoMetricsLab-cli.exe' }
    $devices = & $cli devices 2>&1
    $devicesExit = $LASTEXITCODE
    $devices | ForEach-Object { "    $_" }
    if ($devicesExit -ne 0) { throw "VideoMetricsLab-cli.exe devices failed with exit code $devicesExit" }

    # The native libraries and the notices of the headers built into them.
    $native = Join-Path $output '_internal/vmaf_app/native'
    foreach ($library in @('d3d11_tonemap.dll', 'nvdec_frames.dll', 'vpl_frames.dll', 'amf_frames.dll',
            'mf_frames.dll', 'software_frames.dll')) {
        if (-not (Test-Path (Join-Path $native $library))) {
            throw "Bundled native library is missing: $library"
        }
    }
    foreach ($notice in @('LICENSE.nv-codec-headers.txt', 'LICENSE.onevpl.txt', 'LICENSE.amf.txt',
            'LICENSE.vulkan-headers.txt')) {
        if (-not (Test-Path (Join-Path $native "licenses/$notice"))) {
            throw "Bundled native library license notice is missing: $notice"
        }
    }
    # FFmpeg's decoders for the software frame decoder, and their licence.
    $ffmpeg = Join-Path $output '_internal/vmaf_app/tools/ffmpeg'
    foreach ($file in @('avcodec-63.dll', 'avutil-61.dll', 'licenses/COPYING.LGPLv2.1.txt')) {
        if (-not (Test-Path (Join-Path $ffmpeg $file))) { throw "Bundled FFmpeg file is missing: $file" }
    }

    $perceptualTools = @('ssimulacra2.exe', 'butteraugli_main.exe')
    foreach ($tool in $perceptualTools) {
        if (-not (Test-Path (Join-Path $output "_internal/vmaf_app/tools/libjxl/$tool"))) {
            throw "Bundled perceptual metric tool is missing: $tool"
        }
    }

    # Vship's three builds: CUDA (nvidia/), HIP (amd/) and Vulkan (vulkan/).
    foreach ($build in @('nvidia', 'amd', 'vulkan')) {
        if (-not (Test-Path (Join-Path $output "_internal/vmaf_app/tools/vship/$build/libvship.dll"))) {
            throw "Bundled Vship $build GPU library is missing"
        }
    }

    foreach ($notice in @('LICENSE.vship.txt', 'LICENSE.ssimulacra2.txt', 'LICENSE.butteraugli.txt', 'LICENSE.winpthreads.txt')) {
        if (-not (Test-Path (Join-Path $output "_internal/vmaf_app/tools/vship/licenses/$notice"))) {
            throw "Bundled Vship metric license notice is missing: $notice"
        }
    }

    # libvmaf with CUDA (libvmaf-fast's): without it, VMAF on NVIDIA GPUs is
    # quietly calculated on the CPU; without its record, scores name no
    # version of it.
    $libvmaf = Join-Path $output '_internal/vmaf_app/tools/libvmaf'
    foreach ($file in @('libvmaf.dll', 'libvmaf-fast.json')) {
        if (-not (Test-Path (Join-Path $libvmaf $file))) {
            throw "Bundled libvmaf (CUDA) is missing $file`: run scripts/fetch_libvmaf_fast.ps1"
        }
    }
    foreach ($notice in @('LICENSE.libvmaf.txt', 'LICENSE.pthreads4w.txt', 'LICENSE.nv-codec-headers.txt',
                          'LICENSE.xpsnr.txt')) {
        if (-not (Test-Path (Join-Path $libvmaf "licenses/$notice"))) {
            throw "Bundled libvmaf license notice is missing: $notice"
        }
    }

    # VMAF's features with Vulkan (libvmaf-fast's too): without it,
    # VMAF on Intel and AMD GPUs is quietly calculated on the CPU.
    $vmafVulkan = Join-Path $output '_internal/vmaf_app/tools/vmaf_vulkan'
    if (-not (Test-Path (Join-Path $vmafVulkan 'vmaf_vulkan.dll'))) {
        throw 'Bundled vmaf_vulkan is missing: run scripts/fetch_libvmaf_fast.ps1'
    }
    if (-not (Test-Path (Join-Path $vmafVulkan 'licenses/LICENSE.libvmaf.txt'))) {
        throw 'Bundled vmaf_vulkan license notice is missing: LICENSE.libvmaf.txt'
    }

    # Distributions must carry the project's MIT terms and the notices for
    # the third-party components bundled alongside the executable.
    Copy-Item (Join-Path $projectDirectory 'LICENSE') (Join-Path $output 'LICENSE.txt') -Force
    Copy-Item (Join-Path $projectDirectory 'docs/THIRD_PARTY.md') (Join-Path $output 'THIRD_PARTY.md') -Force

    # 4. Zip it, so a release asset is one file.
    Write-Host '==> Packaging' -ForegroundColor Cyan
    $zip = Join-Path $OutputRoot 'VideoMetricsLab-windows.zip'
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path $output -DestinationPath $zip -CompressionLevel Optimal

    $folderSize = (Get-ChildItem $output -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
    $zipSize = (Get-Item $zip).Length / 1MB
    Write-Host ''
    Write-Host ('Built {0:N0} MB folder, {1:N0} MB zip' -f $folderSize, $zipSize) -ForegroundColor Green
    Write-Host "  $output"
    Write-Host "  $zip"
    Write-Host ''
    Write-Host 'FFmpeg itself (ffmpeg.exe, ffprobe.exe) is not bundled. See docs/BUILD.md.' -ForegroundColor Yellow
} finally {
    Pop-Location
}
