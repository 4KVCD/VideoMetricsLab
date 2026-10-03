# GPU-native helper, linked dynamically against UNMODIFIED shared FFmpeg.
# Requires VS 2022 C++, CUDA Toolkit, and build_libvmaf_cuda.ps1's completed
# checkout. This does not rebuild FFmpeg or the app's libvmaf DLL.
param(
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$WorkDirectory = (Join-Path $env:TEMP 'vml-native-vmaf'),
    [string]$VmafBuild = (Join-Path $env:TEMP 'libvmaf-cuda-build'),
    [string]$FFmpegRoot = ''
)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$output = Join-Path $project 'vmaf_app/tools/vmaf_native'
$asset = 'ffmpeg-n9.0.2-22-g46d8f462ee-win64-lgpl-shared-9.0.zip'
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-01-13-06/$asset"
$sha256 = '2a41605c6c28455c7029e5057f77cda18838203fa09d40bed6e6cac14200e1f5'
$savedEnvironment = @{}
Get-ChildItem env: | ForEach-Object { $savedEnvironment[$_.Name] = $_.Value }
function Invoke-Checked([string]$what, [scriptblock]$command) {
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed ($LASTEXITCODE)" }
}
try {
    if (-not $CudaPath -or -not (Test-Path "$CudaPath/bin/nvcc.exe")) { throw 'Pass -CudaPath for the CUDA Toolkit' }
    if (-not (Test-Path "$VmafBuild/build/libvmaf.def")) { throw 'Build libvmaf with scripts/build_libvmaf_cuda.ps1 first' }
    New-Item -ItemType Directory -Force $WorkDirectory, $output | Out-Null
    if (-not $FFmpegRoot) {
        $zip = Join-Path $WorkDirectory $asset
        if (-not (Test-Path $zip)) { Invoke-WebRequest $url -OutFile $zip }
        if ((Get-FileHash $zip).Hash.ToLowerInvariant() -ne $sha256) { throw 'Shared FFmpeg checksum mismatch' }
        $extract = Join-Path $WorkDirectory 'pinned-ffmpeg'
        if (-not (Test-Path $extract)) { Expand-Archive -LiteralPath $zip -DestinationPath $extract }
        $FFmpegRoot = (Get-ChildItem $extract -Directory | Select-Object -First 1).FullName
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'Visual Studio C++ tools were not found' }
    cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -LiteralPath "env:$($matches[1])" $matches[2] }
    }
    $import = Join-Path $WorkDirectory 'libvmaf.lib'
    Invoke-Checked 'libvmaf import library' { lib /nologo "/def:$VmafBuild/build/libvmaf.def" "/out:$import" /machine:x64 }
    Invoke-Checked 'Luma CUDA kernel' {
        & "$CudaPath/bin/nvcc.exe" -ptx -O3 -arch=compute_75 (Join-Path $project 'native/vmaf_prepare.cu') -o "$output/vmaf_prepare.ptx"
    }
    Invoke-Checked 'Native VMAF helper' {
        cl /nologo /std:c++17 /EHsc /O2 /MT /W4 /Brepro `
            "/I$FFmpegRoot/include" "/I$CudaPath/include" "/I$VmafBuild/vmaf/libvmaf/include" `
            (Join-Path $project 'native/vmaf_native.cpp') "/Fo$WorkDirectory/vmaf_native.obj" "/Fe$output/vmaf_native.exe" `
            /link /Brepro "/LIBPATH:$FFmpegRoot/lib" "/LIBPATH:$CudaPath/lib/x64" `
            avcodec.lib avformat.lib avfilter.lib avutil.lib cuda.lib $import
    }
    # Import closure only: no avdevice, player/encoder CLI or development files.
    foreach ($name in @('avcodec-63.dll', 'avformat-63.dll', 'avfilter-12.dll', 'avutil-61.dll', 'swscale-10.dll', 'swresample-7.dll')) {
        Copy-Item -LiteralPath (Join-Path $FFmpegRoot "bin/$name") -Destination $output
    }
    Copy-Item -LiteralPath (Join-Path $project 'vmaf_app/tools/libvmaf/libvmaf.dll') -Destination $output
    $notices = Join-Path $output 'licenses'
    New-Item -ItemType Directory -Force $notices | Out-Null
    Copy-Item "$FFmpegRoot/LICENSE.txt" "$notices/LICENSE.FFmpeg.txt"
    Copy-Item "$project/vmaf_app/tools/libvmaf/licenses/*" $notices
    Copy-Item "$project/LICENSE" "$notices/LICENSE.VideoMetricsLab.txt"
    $gpl = Join-Path $WorkDirectory 'COPYING.GPLv3'
    if (-not (Test-Path $gpl)) {
        Invoke-WebRequest 'https://raw.githubusercontent.com/FFmpeg/FFmpeg/46d8f462ee/COPYING.GPLv3' -OutFile $gpl
    }
    Copy-Item $gpl "$notices/LICENSE.GPLv3.txt"
    $version = (& "$FFmpegRoot/bin/ffmpeg.exe" -version | Select-Object -First 3) -join "`n"
    Set-Content -Encoding utf8 "$notices/BUILD.txt" @"
VideoMetricsLab native helper: MIT (native/vmaf_native.cpp and vmaf_prepare.cu).
FFmpeg DLLs: LGPL-3.0-or-later, dynamically linked and replaceable.
Unmodified BtbN build: $url
Archive SHA256: $sha256
Build scripts/source and dependency recipes: https://github.com/BtbN/FFmpeg-Builds
FFmpeg source: https://github.com/FFmpeg/FFmpeg/tree/46d8f462ee
The exact FFmpeg commit is 46d8f462ee, recorded in the version below.
libvmaf build/patch recipes: scripts/build_libvmaf_cuda.ps1.
$version
"@
    Write-Host "Native helper built: $output"
}
finally {
    Get-ChildItem env: | Where-Object { -not $savedEnvironment.ContainsKey($_.Name) } | ForEach-Object { Remove-Item -LiteralPath "env:$($_.Name)" }
    foreach ($name in $savedEnvironment.Keys) { Set-Item -LiteralPath "env:$name" $savedEnvironment[$name] }
}
