# Builds libvmaf with CUDA (vmaf_app/tools/libvmaf/libvmaf.dll), used for
# VMAF and VMAF NEG on NVIDIA GPUs (vmaf_app/core/vmaf_cuda.py). Everything
# else -- VMAF on other GPUs and on the CPU, VMAF v1, PSNR, SSIM -- still
# uses the libvmaf inside the user's FFmpeg.
#
# libvmaf master plus open pull requests, each pinned to the commit tested:
#   1477  native MSVC build (with the pthread-win32 submodule)
#   1573  CUDA: build fixes, SIGSEGV in pinned pictures
#   1583  CUDA: the end-of-stream double flush with threads, and the motion
#         SAD reset racing the motion kernel (randomly low motion scores)
#   1644  CUDA motion: the CPU's reflect-101 edge mirror (motion2 was off at
#         the frame edges)
#   1612  CUDA motion: an uninitialized previous blur on the first frame
#   1614  CUDA VIF: the accumulator reset racing scale 0 (NaN scores)
#   1647-1651  CUDA ADM: five differences from the CPU (contrast masking
#         neighbours, rounding, border clamping, the one degree angle, the
#         scale 0 denominator shift)
#   1652  vmaf_read_pictures releasing both pictures on every return
# Their regression tests each add to libvmaf/test/meson.build, which then
# conflicts; tests are not built here, so that file keeps master's version.
# Any other conflict stops the build.
#
# Needs git, Visual Studio 2022 Build Tools (C++), the CUDA Toolkit (13.x),
# Python with meson and ninja (pip install meson ninja), and nasm, cmake and
# xxd on PATH (Git for Windows has xxd in usr\bin). The CUDA runtime is not
# linked: libvmaf loads the NVIDIA driver (nvcuda.dll) when it starts.
param(
    [string]$VmafCommit = 'cea2b4d832a105116a3f16f56d6f5d953421952c',            # master, 2026-10-02
    [string]$NvCodecHeadersCommit = 'eddcea9e27f6b772057c9b3f87de2cc1737faffc',  # 13.1.15.1 in development
    [string]$Python = 'python',
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$WorkDirectory = (Join-Path $env:TEMP 'libvmaf-cuda-build')
)
$ErrorActionPreference = 'Stop'
# Pull request -> the commit of it that is merged, in this order.
$pullRequests = [ordered]@{
    '1477' = 'e991a60c8fbb2ef1a8012bbd3c92215a82c65467'
    '1573' = 'bec6747dfa156b06157c344e01be28bc558eb7c3'
    '1583' = 'dcc00a908713141af57fe8d190250e43c2f7a880'
    '1644' = '72f88b55d176231f49ceaf0e974ca93d880b7171'
    '1612' = '799cf20807122669bc695de7baf58360975f6adc'
    '1614' = 'c94761fd5b9d65306eff030d89259fc91ad6be57'
    '1647' = '474af5a0753a77824e412e31303bc6d312b4fac5'
    '1648' = 'bfae4246e5d2ee9717c3ce17b1ad6178e2380eeb'
    '1649' = '4d5bebbc79faa5fdadd79d219bb9fb16d6f59c19'
    '1650' = 'bca11dd8f2ec0db64e89860288b0f83d3950b921'
    '1651' = 'edfbbd37ed7575cf67fba9ee9ee78e1f51b75722'
    '1652' = '8781fe54df4a9d2a5894d73fa4a2f3b8c8635d82'
}
# libvmaf's public API (include/libvmaf/*.h): what the DLL exports.
$exports = @(
    'vmaf_version', 'vmaf_init', 'vmaf_close', 'vmaf_use_feature', 'vmaf_use_features_from_model',
    'vmaf_use_features_from_model_collection', 'vmaf_import_feature_score', 'vmaf_read_pictures',
    'vmaf_score_at_index', 'vmaf_score_at_index_model_collection', 'vmaf_feature_score_at_index',
    'vmaf_score_pooled', 'vmaf_score_pooled_model_collection', 'vmaf_feature_score_pooled',
    'vmaf_preallocate_pictures', 'vmaf_fetch_preallocated_picture', 'vmaf_write_output',
    'vmaf_picture_alloc', 'vmaf_picture_unref', 'vmaf_model_load', 'vmaf_model_load_from_path',
    'vmaf_model_feature_overload', 'vmaf_model_destroy', 'vmaf_model_collection_load',
    'vmaf_model_collection_load_from_path', 'vmaf_model_collection_feature_overload',
    'vmaf_model_collection_destroy', 'vmaf_feature_dictionary_set', 'vmaf_feature_dictionary_free',
    'vmaf_cuda_state_init', 'vmaf_cuda_import_state', 'vmaf_cuda_preallocate_pictures',
    'vmaf_cuda_fetch_preallocated_picture'
)
$projectDirectory = Split-Path $PSScriptRoot -Parent
$outputDirectory = Join-Path $projectDirectory 'vmaf_app/tools/libvmaf'
$output = Join-Path $outputDirectory 'libvmaf.dll'

# Native tools write progress and warnings to stderr, and under 'Stop'
# Windows PowerShell makes each such line a terminating error: they run with
# 'Continue', and their exit codes decide.
function Invoke-Native([scriptblock]$command, [switch]$Quiet) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $command 2>&1 | ForEach-Object { if (-not $Quiet) { Write-Host "$_" } }
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Invoke-Checked([string]$what, [scriptblock]$command) {
    Invoke-Native $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# The script runs in the caller's PowerShell session, and what it sets there
# -- Visual Studio's environment, PATH, INCLUDE, the fixed git dates below --
# would outlive it: a commit made in that window afterwards would be dated
# 2026-10-02 too. The environment is put back as it was at the end.
$savedEnvironment = @{}
Get-ChildItem env: | ForEach-Object { $savedEnvironment[$_.Name] = $_.Value }
try {
    if (-not $CudaPath -or -not (Test-Path (Join-Path $CudaPath 'bin/nvcc.exe'))) {
        throw 'The CUDA Toolkit was not found: pass -CudaPath or set CUDA_PATH'
    }
    # The Visual Studio environment, taken into this session.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $visualStudio = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $visualStudio) { throw 'Visual Studio with the C++ tools was not found' }
    $env:PATH = "$(Split-Path $vswhere);$env:PATH"
    # Only what it changes: setting a variable that is empty deletes it.
    cmd /c "`"$visualStudio\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$' -and $matches[2] -cne $savedEnvironment[$matches[1]]) {
            Set-Item -LiteralPath "env:$($matches[1])" $matches[2]
        }
    }
    # Git's usr\bin (xxd) goes last: it has a link.exe of its own.
    # Python's own folder too: meson looks for ninja on PATH.
    $pythonDirectory = Split-Path (& $Python -c 'import sys; print(sys.executable)')
    $env:PATH = "$CudaPath\bin;$pythonDirectory;$env:PATH;$env:ProgramFiles\Git\usr\bin"
    $env:CC = 'cl'; $env:CXX = 'cl'
    $env:CUDA_PATH = $CudaPath

    New-Item -ItemType Directory -Path $WorkDirectory -Force | Out-Null
    $vmaf = Join-Path $WorkDirectory 'vmaf'
    $headers = Join-Path $WorkDirectory 'nv-codec-headers'
    $build = Join-Path $WorkDirectory 'build'

    if (-not (Test-Path (Join-Path $vmaf '.git'))) {
        Invoke-Checked 'Cloning libvmaf' { git clone --quiet https://github.com/Netflix/vmaf.git $vmaf }
    }
    if (-not (Test-Path (Join-Path $headers '.git'))) {
        Invoke-Checked 'Cloning nv-codec-headers' { git clone --quiet https://github.com/FFmpeg/nv-codec-headers.git $headers }
    }
    Invoke-Checked 'Checking out nv-codec-headers' { git -C $headers -c advice.detachedHead=false checkout --quiet --force $NvCodecHeadersCommit }

    Push-Location $vmaf
    try {
        $refs = @($VmafCommit) + @($pullRequests.Keys | ForEach-Object { "pull/$_/head" })
        Invoke-Checked 'Fetching libvmaf and the pull requests' { git fetch --quiet origin @refs }
        foreach ($number in $pullRequests.Keys) {
            Invoke-Native -Quiet { git rev-parse --verify --quiet "$($pullRequests[$number])^{commit}" }
            if ($LASTEXITCODE -ne 0) { throw "Pull request $number no longer contains commit $($pullRequests[$number])" }
        }
        Invoke-Native -Quiet { git merge --abort }  # a merge an earlier run stopped in
        Invoke-Checked 'Checking out libvmaf' { git -c advice.detachedHead=false checkout --quiet --force $VmafCommit }
        $identity = @('-c', 'user.name=VideoMetricsLab build', '-c', 'user.email=build@localhost')
        # Fixed dates make the merge commits, and so the version libvmaf reports
        # (the commit it was built from), the same on every build.
        $env:GIT_AUTHOR_DATE = '2026-10-02T00:00:00Z'
        $env:GIT_COMMITTER_DATE = '2026-10-02T00:00:00Z'
        foreach ($number in $pullRequests.Keys) {
            Invoke-Native { git @identity merge --quiet --no-edit $pullRequests[$number] }
            if ($LASTEXITCODE -ne 0) {
                $conflicts = @(git diff --name-only --diff-filter=U)
                if ($conflicts.Count -ne 1 -or $conflicts[0] -ne 'libvmaf/test/meson.build') {
                    throw "Merging pull request $number conflicts in: $($conflicts -join ', ')"
                }
                Invoke-Checked "Resolving pull request $number" { git checkout --quiet --ours libvmaf/test/meson.build }
                Invoke-Checked "Merging pull request $number" { git add libvmaf/test/meson.build; git @identity commit --quiet --no-edit }
            }
        }
        Invoke-Checked 'Fetching pthread-win32' { git submodule update --quiet --init --force libvmaf/subprojects/pthread-win32 }
    }
    finally {
        Pop-Location
    }

    # The headers reach cl through INCLUDE, not -I in c_args: meson splits c_args
    # at spaces (a profile like C:\Users\Jo Smith), and nvcc's custom targets do
    # not get c_args at all; cl, nvcc's host compiler too, reads INCLUDE.
    $nvInclude = Join-Path $headers 'include'
    $env:INCLUDE = "$nvInclude;$env:INCLUDE"
    if (Test-Path $build) { Remove-Item -Recurse -Force $build }
    Invoke-Checked 'Configuring libvmaf' {
        & $Python -m mesonbuild.mesonmain setup (Join-Path $vmaf 'libvmaf') $build --buildtype release `
            --default-library static -Denable_cuda=true -Denable_nvcc=true -Denable_tests=false `
            -Denable_tools=false -Denable_docs=false -Denable_float=false -Dbuilt_in_models=true -Db_vscrt=mt `
            -Dc_args=/Brepro -Dcpp_args=/Brepro -Dc_link_args=/Brepro -Dcpp_link_args=/Brepro
    }
    Invoke-Checked 'Building libvmaf' { & $Python -m mesonbuild.mesonmain compile -C $build }

    # The static library linked into a DLL exporting the public API (libvmaf
    # declares no exports of its own).
    $definitions = Join-Path $build 'libvmaf.def'
    Set-Content -Path $definitions -Encoding ascii -Value (@('LIBRARY libvmaf', 'EXPORTS') + $exports)
    # The C runtime is linked in (b_vscrt=mt above), so the DLL needs nothing
    # beyond Windows. pthread-win32's own CMake build ignores b_vscrt and uses the
    # runtime DLL, so its one translation unit is compiled again here, with the
    # definitions its CMakeLists gives the static pthreadVC3 library.
    $pthreadSource = Join-Path $vmaf 'libvmaf/subprojects/pthread-win32'
    $pthreads = Join-Path $build 'pthreadVC3-mt.obj'
    Invoke-Checked 'Compiling pthread-win32' {
        cl /nologo /c /O2 /MT /Brepro /DPTW32_ARCHAMD64 /DPTW32_BUILD_INLINED /DHAVE_CONFIG_H /DPTW32_RC_MSC `
            /DPTW32_CLEANUP_C /DPTW32_STATIC_LIB "/I$pthreadSource" (Join-Path $pthreadSource 'pthread-JMP.c') "/Fo$pthreads"
    }
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    Invoke-Checked 'Linking libvmaf.dll' {
        link /nologo /DLL /MACHINE:X64 /Brepro /OUT:$output /DEF:$definitions (Join-Path $build 'src/vmaf.lib') $pthreads
    }
    Remove-Item (Join-Path $outputDirectory 'libvmaf.lib'), (Join-Path $outputDirectory 'libvmaf.exp') -ErrorAction SilentlyContinue

    # The notices that ship with it.
    $licenses = Join-Path $outputDirectory 'licenses'
    New-Item -ItemType Directory -Path $licenses -Force | Out-Null
    Copy-Item (Join-Path $vmaf 'LICENSE') (Join-Path $licenses 'LICENSE.libvmaf.txt')
    Copy-Item (Join-Path $vmaf 'libvmaf/subprojects/pthread-win32/docs/LICENSE.md') (Join-Path $licenses 'LICENSE.pthreads4w.txt')
    # nv-codec-headers' CUDA loader is compiled in; its MIT notice is the
    # comment its headers open with.
    $loader = Get-Content (Join-Path $nvInclude 'ffnvcodec/dynlink_loader.h')
    $end = [Array]::FindIndex($loader, [Predicate[string]] { param($line) $line.Trim() -eq '*/' })
    Set-Content -Path (Join-Path $licenses 'LICENSE.nv-codec-headers.txt') -Encoding ascii -Value (
        @('nv-codec-headers (https://github.com/FFmpeg/nv-codec-headers), commit ' + $NvCodecHeadersCommit, '') +
        ($loader[1..($end - 1)] | ForEach-Object { $_ -replace '^ \* ?', '' }))

    $hash = (Get-FileHash $output -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "libvmaf (commit $($VmafCommit.Substring(0, 8)) + $($pullRequests.Count) pull requests) with CUDA: $output"
    Write-Host "SHA-256 $hash, $((Get-Item $output).Length) bytes"
}
finally {
    Get-ChildItem env: | Where-Object { -not $savedEnvironment.ContainsKey($_.Name) } |
        ForEach-Object { Remove-Item -LiteralPath "env:$($_.Name)" }
    foreach ($name in $savedEnvironment.Keys) {
        $current = Get-Item -LiteralPath "env:$name" -ErrorAction SilentlyContinue
        if (-not $current -or $current.Value -cne $savedEnvironment[$name]) {
            Set-Item -LiteralPath "env:$name" $savedEnvironment[$name]
        }
    }
}
