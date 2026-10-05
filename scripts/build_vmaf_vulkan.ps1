# Builds VMAF's features for any GPU with Vulkan
# (vmaf_app/tools/vmaf_vulkan/vmaf_vulkan.dll, used by
# vmaf_app/core/vmaf_vulkan.py) from native/vmaf_vulkan: the compute shaders
# are compiled to SPIR-V with Slang and embedded in the DLL.
#
# MSVC with the static C runtime, as scripts/build_libvmaf_cuda.ps1 builds
# libvmaf: the DLL turns the GPU's sums into feature scores with the same
# powf(), pow() and log2() as libvmaf does, and needs only Windows and a
# Vulkan driver (vulkan-1.dll is loaded when first used).
#
# Needs git and Visual Studio 2022 Build Tools (C++). Slang is downloaded.
param(
    [string]$SlangVersion = '2026.10.2',
    [string]$SlangSha256 = 'f21fca4ba78bfb366ef3b282b4926e3efca61c2ffd72a482b024c9f8413331d0',
    [string]$VulkanHeadersCommit = '3c65a01745e4a1134d32b9c2c456472212dba16d',  # 2026-09-25
    [string]$WorkDirectory = (Join-Path $env:TEMP 'vmaf-vulkan-build')
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$source = Join-Path $projectDirectory 'native/vmaf_vulkan'
$outputDirectory = Join-Path $projectDirectory 'vmaf_app/tools/vmaf_vulkan'
$output = Join-Path $outputDirectory 'vmaf_vulkan.dll'

# name = file, then the preprocessor definitions. The order is the order of
# the kShader_ enumeration vmaf_vulkan.cpp indexes: the scales of a shader
# follow each other.
$shaders = [ordered]@{
    'motion_8'          = 'motion', 'BPC16=0'
    'motion_16'         = 'motion', 'BPC16=1'
    'vif_vert_0_8'      = 'vif_vert', 'SRC_PICTURE=1', 'BPC16=0', 'SCALE=0', 'FW=17', 'FW_RD=9'
    'vif_vert_0_16'     = 'vif_vert', 'SRC_PICTURE=1', 'BPC16=1', 'SCALE=0', 'FW=17', 'FW_RD=9'
    'vif_vert_1'        = 'vif_vert', 'SRC_PICTURE=0', 'BPC16=0', 'SCALE=1', 'FW=9', 'FW_RD=5'
    'vif_vert_2'        = 'vif_vert', 'SRC_PICTURE=0', 'BPC16=0', 'SCALE=2', 'FW=5', 'FW_RD=3'
    'vif_vert_3'        = 'vif_vert', 'SRC_PICTURE=0', 'BPC16=0', 'SCALE=3', 'FW=3', 'FW_RD=0'
    'vif_hori_0'        = 'vif_hori', 'NATIVE_F64=0', 'BPC16=0', 'SCALE=0', 'FW=17', 'FW_RD=9'
    'vif_hori_1'        = 'vif_hori', 'NATIVE_F64=0', 'BPC16=0', 'SCALE=1', 'FW=9', 'FW_RD=5'
    'vif_hori_2'        = 'vif_hori', 'NATIVE_F64=0', 'BPC16=0', 'SCALE=2', 'FW=5', 'FW_RD=3'
    'vif_hori_3'        = 'vif_hori', 'NATIVE_F64=0', 'BPC16=0', 'SCALE=3', 'FW=3', 'FW_RD=0'
    'vif_hori_native_0' = 'vif_hori', 'NATIVE_F64=1', 'BPC16=0', 'SCALE=0', 'FW=17', 'FW_RD=9'
    'vif_hori_native_1' = 'vif_hori', 'NATIVE_F64=1', 'BPC16=0', 'SCALE=1', 'FW=9', 'FW_RD=5'
    'vif_hori_native_2' = 'vif_hori', 'NATIVE_F64=1', 'BPC16=0', 'SCALE=2', 'FW=5', 'FW_RD=3'
    'vif_hori_native_3' = 'vif_hori', 'NATIVE_F64=1', 'BPC16=0', 'SCALE=3', 'FW=3', 'FW_RD=0'
    'adm_dwt_0_8'       = 'adm_dwt', 'SCALE0=1', 'BPC16=0'
    'adm_dwt_0_16'      = 'adm_dwt', 'SCALE0=1', 'BPC16=1'
    'adm_dwt'           = 'adm_dwt', 'SCALE0=0', 'BPC16=0'
    'adm_decouple_0'    = 'adm_decouple', 'SCALE0=1', 'BPC16=0', 'VARIANT=0'
    'adm_decouple'      = 'adm_decouple', 'SCALE0=0', 'BPC16=0', 'VARIANT=0'
    'adm_csf_den_0'     = 'adm_csf_den', 'SCALE0=1', 'BPC16=0'
    'adm_csf_den'       = 'adm_csf_den', 'SCALE0=0', 'BPC16=0'
    'adm_cm_0'          = 'adm_cm', 'SCALE0=1', 'BPC16=0'
    'adm_cm'            = 'adm_cm', 'SCALE0=0', 'BPC16=0'
    # adm_decouple_0 as the CUDA kernel writes it, in 64 bits (1), and steps
    # of that shown (2-7), for scripts/diagnose_vmaf_vulkan.py: see VARIANT
    # in the shader.
    'adm_decouple_0_v1' = 'adm_decouple', 'SCALE0=1', 'BPC16=0', 'VARIANT=1'
    'adm_decouple_0_v2' = 'adm_decouple', 'SCALE0=1', 'BPC16=0', 'VARIANT=2'
    'adm_decouple_0_v3' = 'adm_decouple', 'SCALE0=1', 'BPC16=0', 'VARIANT=3'
    'adm_decouple_0_v4' = 'adm_decouple', 'SCALE0=1', 'BPC16=0', 'VARIANT=4'
    'adm_decouple_0_v5' = 'adm_decouple', 'SCALE0=1', 'BPC16=0', 'VARIANT=5'
    'adm_decouple_0_v6' = 'adm_decouple', 'SCALE0=1', 'BPC16=0', 'VARIANT=6'
    'adm_decouple_0_v7' = 'adm_decouple', 'SCALE0=1', 'BPC16=0', 'VARIANT=7'
}

function Invoke-Checked([string]$what, [scriptblock]$command) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $command 2>&1 | ForEach-Object { Write-Host "$_" } }
    finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

New-Item -ItemType Directory -Path $WorkDirectory -Force | Out-Null

# Slang, the shader compiler.
$slang = Join-Path $WorkDirectory "slang-$SlangVersion"
$slangc = Join-Path $slang 'bin/slangc.exe'
if (-not (Test-Path $slangc)) {
    $archive = Join-Path $WorkDirectory "slang-$SlangVersion.zip"
    $url = "https://github.com/shader-slang/slang/releases/download/v$SlangVersion/slang-$SlangVersion-windows-x86_64.zip"
    Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
    $hash = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($SlangSha256 -and $hash -ne $SlangSha256) { throw "Slang $SlangVersion has SHA-256 $hash, expected $SlangSha256" }
    Expand-Archive -Path $archive -DestinationPath $slang -Force
}

# The Vulkan headers.
$headers = Join-Path $WorkDirectory 'Vulkan-Headers'
if (-not (Test-Path (Join-Path $headers '.git'))) {
    Invoke-Checked 'Cloning Vulkan-Headers' {
        git clone --quiet --filter=blob:none --no-checkout https://github.com/KhronosGroup/Vulkan-Headers.git $headers
    }
}
git -C $headers cat-file -e "$VulkanHeadersCommit^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) { Invoke-Checked 'Fetching Vulkan-Headers' { git -C $headers fetch --quiet origin } }
Invoke-Checked 'Checking out Vulkan-Headers' {
    git -C $headers -c advice.detachedHead=false checkout --quiet --force $VulkanHeadersCommit
}

# The shaders, as one header of SPIR-V words.
$build = Join-Path $WorkDirectory 'build'
New-Item -ItemType Directory -Path $build -Force | Out-Null
$header = [System.Text.StringBuilder]::new()
[void]$header.AppendLine('// Generated by scripts/build_vmaf_vulkan.ps1 from native/vmaf_vulkan/shaders.')
[void]$header.AppendLine('#include <stdint.h>')
[void]$header.AppendLine("#include <stddef.h>`n")
foreach ($name in $shaders.Keys) {
    $file, $definitions = $shaders[$name]
    $spirv = Join-Path $build "$name.spv"
    $arguments = @((Join-Path $source "shaders/$file.slang"), '-target', 'spirv', '-profile', 'spirv_1_3', '-entry', 'main',
                   '-stage', 'compute', '-O2', '-o', $spirv) + @($definitions | ForEach-Object { "-D$_" })
    Invoke-Checked "Compiling the shader $name" { & $slangc @arguments }
    $bytes = [System.IO.File]::ReadAllBytes($spirv)
    $words = for ($i = 0; $i -lt $bytes.Length; $i += 4) { '0x{0:x8}' -f [System.BitConverter]::ToUInt32($bytes, $i) }
    [void]$header.AppendLine("static const uint32_t kSpirv_$name[] = {")
    for ($i = 0; $i -lt $words.Count; $i += 10) {
        [void]$header.AppendLine('    ' + (($words[$i..([Math]::Min($i + 9, $words.Count - 1))]) -join ', ') + ',')
    }
    [void]$header.AppendLine("};`n")
}
[void]$header.AppendLine('enum {')
foreach ($name in $shaders.Keys) { [void]$header.AppendLine("    kShader_$name,") }
[void]$header.AppendLine("    kShaderCount`n};`n")
[void]$header.AppendLine('static const struct { const char *name; const uint32_t *code; size_t bytes; } kShaders[kShaderCount] = {')
foreach ($name in $shaders.Keys) { [void]$header.AppendLine("    { `"$name`", kSpirv_$name, sizeof kSpirv_$name },") }
[void]$header.AppendLine('};')
Set-Content -Path (Join-Path $build 'shaders_spv.h') -Encoding ascii -Value $header.ToString()

# The DLL. The Visual Studio environment is taken in a child cmd, so nothing
# of it stays in the caller's session.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$visualStudio = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $visualStudio) { throw 'Visual Studio with the C++ tools was not found' }
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$compile = "cl /nologo /LD /O2 /MT /EHsc /std:c++17 /W3 /wd4244 /wd4305 /wd4996 /Brepro /I`"$build`" /I`"$(Join-Path $headers 'include')`" " +
    "`"$(Join-Path $source 'vmaf_vulkan.cpp')`" /Fo`"$build\\`" /Fe`"$output`" /link /Brepro /IMPLIB:`"$build\vmaf_vulkan.lib`""
$batch = Join-Path $build 'compile.bat'
Set-Content -Path $batch -Encoding ascii -Value @(
    '@echo off',
    "set PATH=$(Split-Path $vswhere);%PATH%",
    "call `"$visualStudio\VC\Auxiliary\Build\vcvars64.bat`" >nul",
    $compile)
Invoke-Checked 'Compiling vmaf_vulkan.dll' { cmd /c $batch }

$hash = (Get-FileHash $output -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "VMAF for Vulkan: $output"
Write-Host "SHA-256 $hash, $((Get-Item $output).Length) bytes"
