# Builds the app's self-contained Windows distributable: the window
# (VideoMetricsLab.exe) and the command line (VideoMetricsLab-cli.exe), with
# Qt and Microsoft's C++ runtime beside them and the app's own files in
# resources\ (the VMAF models, the translations, the bundled tools and the
# native libraries), where the programs look for them (resource_folder() in
# cpp/core/src/model_select.cpp).
#
#   ./scripts/build_release.ps1 [-QtPath C:\Qt\6.12.0\msvc2022_64]
#
# Needs what building the app needs (cpp/README.md), g++ (MinGW-w64) for the
# native libraries, and ffmpeg.exe and ffprobe.exe on the PATH for the
# self-test. Output goes outside the repository, under
# %LOCALAPPDATA%\VideoMetricsLab-build\cpp.
param(
    [string]$OutputRoot = (Join-Path $env:LOCALAPPDATA 'VideoMetricsLab-build\cpp'),
    # Qt's folder, for a build folder that has not been configured with it.
    [string]$QtPath = '',
    # Builds with a local build of libvmaf-fast (build_libvmaf_fast_local.ps1),
    # to test; a release takes a release of libvmaf-fast.
    [switch]$AllowLocalLibvmaf
)
$ErrorActionPreference = 'Stop'

$projectDirectory = Split-Path $PSScriptRoot -Parent
$cppDirectory = Join-Path $projectDirectory 'cpp'
$appFiles = Join-Path $projectDirectory 'resources'

# Runs a program, and throws when it fails. Windows PowerShell turns each
# line a program writes to stderr into an error, which 'Stop' would make
# fatal, and CMake and windeployqt write their progress there: only the
# exit code means anything.
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

# 0. A release takes a release of libvmaf-fast (docs/RELEASING.md), not a
#    local build of the fork.
$libvmafFast = Get-Content -Raw (Join-Path $appFiles 'tools/libvmaf/libvmaf-fast.json') | ConvertFrom-Json
if (-not $libvmafFast.release -and -not $AllowLocalLibvmaf) {
    throw ("libvmaf-fast is a local build ($($libvmafFast.version)): publish libvmaf-fast and install " +
           'its release with scripts/fetch_libvmaf_fast.ps1 (docs/RELEASING.md), or pass ' +
           '-AllowLocalLibvmaf to build one to test')
}
$ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
if (-not $ffmpeg -or -not (Get-Command ffprobe.exe -ErrorAction SilentlyContinue)) {
    throw 'The self-test needs ffmpeg.exe and ffprobe.exe on the PATH'
}

# Microsoft's tools: dumpbin, to read what each program and library
# imports, and the C++ runtime's redistributable DLLs, from the Visual
# Studio the app is built with.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$visualStudio = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $visualStudio) { throw 'Visual Studio 2022 or its Build Tools, with the C++ tools, is needed' }
$build = Join-Path $visualStudio 'VC/Auxiliary/Build'
$toolsVersion = (Get-Content (Join-Path $build 'Microsoft.VCToolsVersion.default.txt')).Trim()
$redistVersion = (Get-Content (Join-Path $build 'Microsoft.VCRedistVersion.default.txt')).Trim()
$dumpbin = Join-Path $visualStudio "VC/Tools/MSVC/$toolsVersion/bin/Hostx64/x64/dumpbin.exe"
$runtime = Get-ChildItem (Join-Path $visualStudio "VC/Redist/MSVC/$redistVersion/x64") -Directory -Filter 'Microsoft.VC*.CRT' |
    Select-Object -First 1
if (-not (Test-Path $dumpbin) -or -not $runtime) {
    throw "Visual Studio's dumpbin or C++ runtime redistributable is missing ($dumpbin, VC/Redist/MSVC/$redistVersion)"
}

# The DLLs a program or library imports: those it needs to start, and those
# it loads when first called (delay-loaded, as the window loads FFmpeg's).
function Get-Imports([string]$File) {
    $lines = & $dumpbin -nologo -dependents $File
    if ($LASTEXITCODE -ne 0) { throw "dumpbin could not read $File" }
    $delayed = $false
    foreach ($line in $lines) {
        if ($line -match 'delay load dependencies') {
            $delayed = $true
        } elseif ($line -match '^\s+Summary\s*$') {
            break
        } elseif ($line -match '^\s+(\S+\.dll)\s*$') {
            [pscustomobject]@{ Name = $Matches[1]; Delayed = $delayed }
        }
    }
}

# 1. The native libraries, from this commit's source: they are not in git,
#    so ones left from another commit would be stale.
if (-not (Get-Command g++ -ErrorAction SilentlyContinue)) {
    throw 'g++ (MinGW-w64) is needed to build the native libraries; see docs/BUILD.md'
}
Write-Host '==> Building d3d11_tonemap.dll' -ForegroundColor Cyan
& (Join-Path $PSScriptRoot 'build_d3d11_tonemap.ps1')
Write-Host '==> Building d3d11_film_grain.dll' -ForegroundColor Cyan
& (Join-Path $PSScriptRoot 'build_d3d11_film_grain.ps1')
Write-Host '==> Building the GPU frame decoders' -ForegroundColor Cyan
& (Join-Path $PSScriptRoot 'build_gpu_frames.ps1')

# 2. The programs, in their Release build.
Write-Host '==> Building the C++ app' -ForegroundColor Cyan
Push-Location $cppDirectory
try {
    $configure = @('--preset', 'windows')
    if ($QtPath) { $configure += "-DCMAKE_PREFIX_PATH=$QtPath" }
    Invoke-Program 'Configuring' { cmake @configure }
    Invoke-Program 'Building' { cmake --build --preset windows-release --target VideoMetricsLab vml-cli }
} finally {
    Pop-Location
}
$buildFolder = Join-Path $cppDirectory 'build/windows'
$window = Join-Path $buildFolder 'app/Release/VideoMetricsLab.exe'
$commandLine = Join-Path $buildFolder 'cli/Release/vml-cli.exe'
if (-not (Test-Path $window)) {
    throw 'CMake built no window: it did not find Qt. Name its folder with -QtPath'
}
$qtFolder = Select-String -Path (Join-Path $buildFolder 'CMakeCache.txt') -Pattern '^Qt6_DIR:PATH=(.*)$' |
    ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1
# Qt6_DIR is <Qt>/lib/cmake/Qt6.
$windeployqt = Join-Path (Split-Path (Split-Path (Split-Path $qtFolder))) 'bin/windeployqt.exe'

# 3. The folder: the programs, Qt, the runtime and the app's files.
Write-Host '==> Assembling' -ForegroundColor Cyan
$output = Join-Path $OutputRoot 'VideoMetricsLab'
if (Test-Path $output) { Remove-Item $output -Recurse -Force }
New-Item -ItemType Directory -Path $output -Force | Out-Null
$exe = Join-Path $output 'VideoMetricsLab.exe'
$cli = Join-Path $output 'VideoMetricsLab-cli.exe'
Copy-Item $window $exe
# Named as the Python app's command line is, so that scripts calling it
# keep working.
Copy-Item $commandLine $cli

# Qt's libraries and plugins, less what windeployqt would add unasked: the
# Direct3D shader compilers (d3dcompiler_47.dll, and dxcompiler.dll and
# dxil.dll: 16 MB) for Qt's own Direct3D drawing, which the app's widgets do
# not use (Video Compare's presenter compiles its shader with Windows'
# d3dcompiler_47.dll), the software OpenGL renderer, the TUIO touch plugin
# (generic), the plugin that tells
# whether the PC is online (networkinformation, never asked) and a TLS
# backend that checks no certificate (qcertonlybackend: Windows' own,
# schannel, is the one the update check uses). The runtime is copied below.
# Qt's translations are the build's (below), not windeployqt's.
Invoke-Program 'windeployqt' {
    & $windeployqt --release --no-translations --no-system-d3d-compiler --no-system-dxc-compiler --no-opengl-sw `
        --no-compiler-runtime --skip-plugin-types generic,networkinformation --exclude-plugins qcertonlybackend $exe
}

# Qt's translations of its own dialogs and buttons (Yes, No, Cancel, the file
# dialogs), in the languages the window has and Qt has translated, where Qt
# looks for them: the translations folder the build put beside the program
# (cpp/app/CMakeLists.txt).
$qtTranslations = Join-Path (Split-Path $window) 'translations'
if (-not (Test-Path (Join-Path $qtTranslations 'qtbase_*.qm'))) {
    throw "The build has no Qt translations: install Qt's translations (qttranslations) with Qt, then configure again"
}
Copy-Item $qtTranslations (Join-Path $output 'translations') -Recurse

$resources = Join-Path $output 'resources'
New-Item -ItemType Directory -Path (Join-Path $resources 'translations') -Force | Out-Null
Copy-Item (Join-Path $appFiles 'models') (Join-Path $resources 'models') -Recurse
Copy-Item (Join-Path $appFiles 'translations/*.json') (Join-Path $resources 'translations')
$tools = Join-Path $resources 'tools'
New-Item -ItemType Directory -Path $tools -Force | Out-Null
# FFmpeg's libraries come with dav1d (scripts/build_dav1d.ps1).
foreach ($tool in @('ffmpeg', 'libjxl', 'libvmaf', 'vmaf_vulkan', 'vship')) {
    Copy-Item (Join-Path $appFiles "tools/$tool") (Join-Path $tools $tool) -Recurse
}
if (-not (Test-Path (Join-Path $tools 'ffmpeg/dav1d.dll'))) {
    throw 'resources/tools/ffmpeg has no dav1d.dll: build it with scripts/build_dav1d.ps1'
}

# The native libraries the C++ app loads, with the notices of what is
# compiled into them (headers, and the film grain library's VTM tables).
# (Not the Vulkan presenter, native/vulkan_presenter.cpp, which the app does
# without on Windows.)
$native = Join-Path $resources 'native'
New-Item -ItemType Directory -Path (Join-Path $native 'licenses') -Force | Out-Null
foreach ($library in @('d3d11_tonemap.dll', 'd3d11_film_grain.dll', 'nvdec_frames.dll', 'vpl_frames.dll',
        'amf_frames.dll', 'mf_frames.dll', 'software_frames.dll')) {
    Copy-Item (Join-Path $appFiles "native/$library") $native
}
foreach ($notice in @('ffnvcodec/LICENSE.nv-codec-headers.txt', 'onevpl/LICENSE.onevpl.txt', 'amf/LICENSE.amf.txt',
        'vulkan/LICENSE.vulkan-headers.txt', 'h274/LICENSE.vtm.txt')) {
    Copy-Item (Join-Path $projectDirectory "native/$notice") (Join-Path $native 'licenses')
}

# The project's terms, the notices of what it bundles, the licence of Qt's
# libraries (the LGPL 3, with the GPL 3 it adds to), and the MIT notices of
# nlohmann/json and miniz, which are compiled into both programs.
Copy-Item (Join-Path $projectDirectory 'LICENSE') (Join-Path $output 'LICENSE.txt')
Copy-Item (Join-Path $projectDirectory 'docs/THIRD_PARTY.md') (Join-Path $output 'THIRD_PARTY.md')
Copy-Item (Join-Path $cppDirectory 'packaging/licenses') (Join-Path $output 'licenses') -Recurse
Copy-Item (Join-Path $cppDirectory 'third_party/nlohmann/LICENSE.MIT') (Join-Path $output 'licenses/LICENSE.nlohmann-json.txt')
Copy-Item (Join-Path $cppDirectory 'third_party/miniz/LICENSE') (Join-Path $output 'licenses/LICENSE.miniz.txt')

# Microsoft's C++ runtime, beside the programs (Windows looks there first,
# for the libraries in resources\ too): whichever of its DLLs anything in
# the folder imports, so that the app runs on a PC without the runtime
# installed. Then everything the programs and Qt's plugins import must be
# in the folder or be Windows' own; FFmpeg's, which the window loads when
# first called, are in resources\tools\ffmpeg.
$programs = Get-ChildItem $output -Recurse -File | Where-Object { $_.Extension -in '.exe', '.dll' }
$imports = @{}
foreach ($program in $programs) { $imports[$program.FullName] = @(Get-Imports $program.FullName) }
$runtimeNeeded = $imports.Values | ForEach-Object { $_ } | Where-Object { $_.Name -match '^(msvcp|vcruntime|concrt|vccorlib)\d' } |
    ForEach-Object { $_.Name.ToLowerInvariant() } | Sort-Object -Unique
foreach ($name in $runtimeNeeded) {
    $source = Join-Path $runtime.FullName $name
    if (-not (Test-Path $source)) { throw "$name is imported but not in Visual Studio's redistributable ($($runtime.FullName))" }
    Copy-Item $source $output
}
$system32 = Join-Path $env:SystemRoot 'System32'
foreach ($program in $programs | Where-Object { -not $_.FullName.StartsWith($resources) }) {
    foreach ($import in $imports[$program.FullName]) {
        $found = (Test-Path (Join-Path $output $import.Name)) -or ($import.Name -match '^(api|ext)-ms-') -or
            (Test-Path (Join-Path $system32 $import.Name)) -or
            ($import.Delayed -and (Test-Path (Join-Path $tools "ffmpeg/$($import.Name)")))
        if (-not $found) {
            throw "$($program.Name) imports $($import.Name), which is neither in the build nor Windows'"
        }
    }
}
"    Microsoft C++ runtime ($redistVersion): $($runtimeNeeded -join ', ')"

# 4. Prove it runs and finds everything it contains, in a profile of its own
#    (not the user's settings), with nothing outside the folder to lean on:
#    no Qt or resources folder named by the environment, and a PATH of
#    Windows' folders and FFmpeg's.
Write-Host '==> Self-test' -ForegroundColor Cyan
$profileFolder = Join-Path $OutputRoot 'self-test-profile'
if (Test-Path $profileFolder) { Remove-Item $profileFolder -Recurse -Force }
New-Item -ItemType Directory -Path $profileFolder -Force | Out-Null
$environment = @{
    USERPROFILE = $profileFolder
    PATH = "$system32;$env:SystemRoot;$(Split-Path $ffmpeg.Source)"
    VML_RESOURCES = $null; QT_PLUGIN_PATH = $null; QT_QPA_PLATFORM_PLUGIN_PATH = $null; QT_QPA_PLATFORM = $null
}
$saved = @{}
foreach ($name in $environment.Keys) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $environment[$name], 'Process')
}
try {
    # Start-Process -Wait: a windowed program, which PowerShell does not
    # wait for; called directly, the script would read the report before
    # the program had written it.
    $selfTest = (Start-Process -FilePath $exe -ArgumentList '--self-test', '--quiet' -Wait -PassThru).ExitCode
    $report = Join-Path $profileFolder '.videometricslab/self-test.txt'
    if (-not (Test-Path $report)) {
        throw 'The self-test wrote no report; the program did not get as far as running it'
    }
    Get-Content $report | ForEach-Object { "    $_" }
    if ($selfTest -ne 0) { throw 'The self-test reported a failure (see above)' }

    # The command line: it must start, find FFmpeg and say what calculates
    # each metric (the GPU libraries are probed in a process of its own).
    Write-Host '==> Command line' -ForegroundColor Cyan
    Invoke-Program 'VideoMetricsLab-cli.exe devices' { & $cli devices 2>&1 }
} finally {
    foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
Remove-Item $profileFolder -Recurse -Force

# 5. Zip it, each entry named with forward slashes, as the zip format has
#    them: Compress-Archive (Windows PowerShell 5.1) wrote backslashes,
#    which other unzip tools took for part of one long file name.
Write-Host '==> Packaging' -ForegroundColor Cyan
$zip = Join-Path $OutputRoot 'VideoMetricsLab-windows.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::Open($zip, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    $base = Split-Path $output -Parent
    foreach ($file in Get-ChildItem $output -Recurse -File) {
        $name = $file.FullName.Substring($base.Length + 1) -replace '\\', '/'
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, $file.FullName, $name, [System.IO.Compression.CompressionLevel]::Optimal)
    }
} finally {
    $archive.Dispose()
}

$folderSize = (Get-ChildItem $output -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
$zipSize = (Get-Item $zip).Length / 1MB
Write-Host ''
Write-Host ('Built {0:N0} MB folder, {1:N0} MB zip' -f $folderSize, $zipSize) -ForegroundColor Green
Write-Host "  $output"
Write-Host "  $zip"
Write-Host ''
Write-Host 'FFmpeg itself (ffmpeg.exe, ffprobe.exe) is not bundled.' -ForegroundColor Yellow
