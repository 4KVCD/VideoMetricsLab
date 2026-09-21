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
    [string[]]$VerifyMedia = @()
)
$ErrorActionPreference = 'Stop'

$projectDirectory = Split-Path $PSScriptRoot -Parent
Push-Location $projectDirectory
try {
    $python = Join-Path $projectDirectory '.venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    $workPath = Join-Path $OutputRoot 'work'
    $distPath = Join-Path $OutputRoot 'dist'
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

    # 1. The GPU HDR->SDR shader. Optional -- the app falls back to FFmpeg
    #    tone mapping without it -- so a missing compiler is a warning, not a
    #    failure. But a release should have it, so say so loudly.
    $tonemap = Join-Path $projectDirectory 'vmaf_app/native/d3d11_tonemap.dll'
    if (Get-Command g++ -ErrorAction SilentlyContinue) {
        Write-Host '==> Building d3d11_tonemap.dll' -ForegroundColor Cyan
        & (Join-Path $PSScriptRoot 'build_d3d11_tonemap.ps1')
    } elseif (Test-Path $tonemap) {
        Write-Warning 'g++ not found; reusing the existing d3d11_tonemap.dll.'
    } else {
        Write-Warning ('g++ not found and no d3d11_tonemap.dll present. The build ' +
                       'will fall back to FFmpeg tone mapping. Install MinGW-w64 ' +
                       'to include the GPU shader.')
    }

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
    Write-Host 'FFmpeg is not bundled. See docs/BUILD.md.' -ForegroundColor Yellow
} finally {
    Pop-Location
}
