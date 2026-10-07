# Builds libvmaf-fast (https://github.com/4KVCD/libvmaf-fast) from the latest
# commit of a branch of its local clone -- the fast branch of
# Documents\libvmaf-fast by default -- into vmaf_app/tools, where
# fetch_libvmaf_fast.ps1 puts a release:
#   libvmaf/libvmaf.dll          libvmaf with CUDA
#   vmaf_vulkan/vmaf_vulkan.dll  VMAF's features with Vulkan
# each with its licences, and libvmaf/libvmaf-fast.json saying what they were
# built from (a version such as 3.2.0-fast.1-146-ga1af96ff: 146 commits after
# that release, at a1af96ff).
#
# This is how the app is developed: the fork's work is used here before it is
# released. A release of the app takes a release of libvmaf-fast instead:
# libvmaf-fast is published first, then fetched, and build_release.ps1
# refuses a local build (docs/RELEASING.md).
#
# The commit is built in a temporary clone, removed after, by the fork's own
# build scripts of that commit: the fork's checkout and any changes in it are
# neither used nor touched, and each DLL reports the commit it was built
# from. The clone is always in the same folder: the source's path is
# compiled in, so two builds differ only where their source does (the same
# commit gives the same files; a commit that changes no library code, only
# the commit the DLLs report). Needs what those scripts need
# (fast/scripts/build_libvmaf_cuda.ps1 and build_vmaf_vulkan.ps1: Visual
# Studio 2022's C++ tools, the CUDA Toolkit, Python with meson and ninja,
# nasm, cmake and xxd).
param(
    [string]$Fork = (Join-Path $env:USERPROFILE 'Documents\libvmaf-fast'),
    [string]$Branch = 'fast',
    [string]$Commit = '',        # another commit instead of the branch's latest, e.g. to compare two builds
    [string]$Python = 'python',  # with meson and ninja (pip install meson ninja)
    [string]$CudaPath = $env:CUDA_PATH
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$tools = Join-Path $projectDirectory 'vmaf_app/tools'

function Invoke-Checked([string]$what, [scriptblock]$command) {
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

& $Python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('mesonbuild') is None)"
if ($LASTEXITCODE -ne 0) { throw "$Python has no meson: pass -Python with one that has (pip install meson ninja)" }
$revision = if ($Commit) { $Commit } else { "refs/heads/$Branch" }
$commit = git -C $Fork rev-parse --verify --quiet "$revision^{commit}"
if ($LASTEXITCODE -ne 0 -or -not $commit) { throw "$Fork has no commit $revision" }
$commit = "$commit".Trim()
# The release it follows and how far; --long, so that a build of the
# release's own commit is not taken for the release.
$described = git -C $Fork describe --tags --long --match 'v*-fast.*' $commit
if ($LASTEXITCODE -ne 0) { throw "$($commit.Substring(0, 8)) follows no libvmaf-fast release tag" }
$version = "$described".Trim() -replace '^v', ''

$work = Join-Path $env:TEMP 'libvmaf-fast-local'
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
$source = Join-Path $work 'source'
$built = Join-Path $work 'dist'
try {
    Invoke-Checked 'Cloning the fork' { git clone --quiet --shared --no-checkout $Fork $source }
    Invoke-Checked "Checking out $commit" { git -C $source -c advice.detachedHead=false checkout --quiet --detach $commit }
    # Submodules (pthread-win32) from the fork's own checkouts of them, where
    # it has them, rather than downloaded: their sources set in the clone's
    # config, which goes with it, and fetched before the build script asks.
    $local = 0
    foreach ($line in @(git -C $source config -f .gitmodules --get-regexp '^submodule\..*\.path$')) {
        if ("$line" -notmatch '^submodule\.(.+)\.path (.+)$') { continue }
        $name, $checkout = $matches[1], (Join-Path $Fork $matches[2])
        if (-not (Test-Path (Join-Path $checkout '.git'))) { continue }
        Invoke-Checked "Taking submodule $name from $checkout" { git -C $source config "submodule.$name.url" $checkout }
        $local++
    }
    if ($local) {
        # Git clones a submodule from a folder only when told to.
        Invoke-Checked 'Fetching the submodules' {
            git -C $source -c protocol.file.allow=always submodule update --quiet --init
        }
    }

    $scripts = Join-Path $source 'fast/scripts'
    & (Join-Path $scripts 'build_libvmaf_cuda.ps1') -Python $Python -CudaPath $CudaPath `
        -OutputDirectory (Join-Path $built 'libvmaf')
    & (Join-Path $scripts 'build_vmaf_vulkan.ps1') -OutputDirectory (Join-Path $built 'vmaf_vulkan')

    # Each reports the commit it was built from, as fast/scripts/package.ps1
    # checks before a release.
    $reported = & $Python -c ("import ctypes, sys; a, b = ctypes.CDLL(sys.argv[1]), ctypes.CDLL(sys.argv[2]); " +
        "a.vmaf_version.restype = b.vv_version.restype = ctypes.c_char_p; " +
        "print(a.vmaf_version().decode(), b.vv_version().decode())") `
        (Join-Path $built 'libvmaf/libvmaf.dll') (Join-Path $built 'vmaf_vulkan/vmaf_vulkan.dll')
    if ($LASTEXITCODE -ne 0) { throw 'The built libraries could not be loaded' }
    $each = "$reported".Trim() -split ' '
    if ($each.Count -ne 2 -or @($each | Where-Object { -not $_ -or -not $commit.StartsWith($_) }).Count) {
        throw "The libraries report '$reported', not $($commit.Substring(0, 8))"
    }

    # Replaced whole, licences included: nothing of an older build stays.
    foreach ($folder in 'libvmaf', 'vmaf_vulkan') {
        $target = Join-Path $tools $folder
        if (Test-Path $target) { Remove-Item -Recurse -Force $target }
        Copy-Item -Recurse (Join-Path $built $folder) $target
    }
    [System.IO.File]::WriteAllText((Join-Path $tools 'libvmaf/libvmaf-fast.json'),
        "{`n  `"version`": `"$version`",`n  `"commit`": `"$commit`",`n  `"release`": false`n}`n")
    Write-Host "libvmaf-fast $version (local, $(if ($Commit) { $Commit } else { $Branch }) of $Fork) into $tools"
    foreach ($file in 'libvmaf/libvmaf.dll', 'vmaf_vulkan/vmaf_vulkan.dll') {
        $hash = (Get-FileHash (Join-Path $tools $file) -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Host "  $file SHA-256 $hash"
    }
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
