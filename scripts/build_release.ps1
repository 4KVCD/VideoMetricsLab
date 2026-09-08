# Builds the self-contained Windows distributable.
#
#   .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
#   ./scripts/build_release.ps1
#
# Output goes OUTSIDE the repository by default, under
# %LOCALAPPDATA%\VideoMetricsCalculator-build. This repository lives in a
# OneDrive folder: building into it would upload ~450 MB on every build, and
# OneDrive's own file handles made the previous build's output impossible to
# delete ("Access is denied" on _internal\...). Pass -OutputRoot to override.
param(
    [string]$OutputRoot = (Join-Path $env:LOCALAPPDATA 'VideoMetricsCalculator-build')
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
    # PyInstaller writes its progress log to stderr. Under
    # $ErrorActionPreference = 'Stop' PowerShell turns each of those lines
    # into a terminating NativeCommandError, so a perfectly successful build
    # "fails" on its first INFO line. Only the exit code means anything here.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python -m PyInstaller --noconfirm `
            --workpath $workPath --distpath $distPath `
            'VideoMetricsCalculator.spec'
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

    $output = Join-Path $distPath 'VideoMetricsCalculator'
    $exe = Join-Path $output 'VideoMetricsCalculator.exe'
    if (-not (Test-Path $exe)) {
        throw 'PyInstaller reported success but produced no executable'
    }

    # 3. Prove it runs and can find everything it does not contain. A build
    #    that starts and then silently falls back to software playback looks
    #    identical to a good one until someone plays a video.
    Write-Host '==> Self-test' -ForegroundColor Cyan
    & $exe --self-test --quiet
    $selfTest = $LASTEXITCODE
    $report = Join-Path $env:USERPROFILE '.vmaf-calculator/self-test.txt'
    if (Test-Path $report) { Get-Content $report | ForEach-Object { "    $_" } }
    if ($selfTest -ne 0) {
        Write-Warning 'Self-test reported a failure (see above). The build exists but is not usable as-is.'
    }

    # 4. Zip it, so a release asset is one file.
    Write-Host '==> Packaging' -ForegroundColor Cyan
    $zip = Join-Path $OutputRoot 'VideoMetricsCalculator-windows.zip'
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
