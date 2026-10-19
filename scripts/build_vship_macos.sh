#!/bin/bash
# Builds what the GPU metrics run on, on macOS (arm64): Vulkan on Metal and
# Vship's Vulkan build for it, as scripts/build_vship_vulkan.ps1 builds
# Windows' Vship. Into resources/tools:
#
#   vulkan/libvulkan.1.dylib     Khronos' Vulkan loader, built from its SDK tag
#   vulkan/libMoltenVK.dylib     MoltenVK, Khronos' Vulkan on Metal (its release)
#   vulkan/MoltenVK_icd.json     the loader's manifest for it
#   vulkan/licenses/             their licences (Apache 2.0)
#   vship/vulkan/libvship.dylib  Vship v5.1.2, linked to the loader
#
# The app points the loader at MoltenVK (VK_DRIVER_FILES) before Vship loads
# it (cpp/core/src/vship.cpp). macOS has no Vulkan of its own, and MoltenVK
# is a portability driver, which Vship asks for (its instance enables
# VK_KHR_portability_enumeration on Apple), so the loader is needed: MoltenVK
# alone does not have that extension.
#
# Vship's SPIR-V is the committed libvshipSpvShaders, as on Windows, but for
# two shaders: SPIRV-Cross, which MoltenVK turns SPIR-V into Metal with,
# mistranslates a byte pointer plus an offset taken as another type's
# pointer, so no Vship pipeline could be made ("reinterpret_cast from 'uchar'
# to 'ulong' is not allowed"). scripts/patches/vship-moltenvk-addresses.patch
# does that arithmetic on the address as an integer instead (the same
# memory), and those two are compiled again with Slang's release compiler.
# Built so, the M1 Pro's scores match Windows' Vulkan ones on an RTX 5090:
# SSIMULACRA2 within 0.0006 a frame, Butteraugli 1e-5, CVVDP 7e-7.
#
# Not committed (.gitignore): the release script runs this.
#
#     scripts/build_vship_macos.sh
#
# Needs git, curl, CMake, Ninja and the Command Line Tools' clang.
set -euo pipefail

vship_commit="5a627933274196219f782b3b74e13fad47cd6370"              # v5.1.2, 2026-10-06 (as Windows')
vulkan_headers_commit="3c65a01745e4a1134d32b9c2c456472212dba16d"     # 2026-09-25 (as Windows')
vulkan_loader_commit="fa218055ff4948b159f31d43f800c808a13d51df"      # vulkan-sdk-1.4.363.0
moltenvk_version="v1.4.2"
moltenvk_sha256="f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e"
slang_version="2026.18.3"
slang_sha256="6c8bf066a4254e8d053f864a52ea18388702d409ed9f20b3d6da66e8d772c906"
deployment_target="14.4"

project="$(cd "$(dirname "$0")/.." && pwd)"
vulkan_out="$project/resources/tools/vulkan"
vship_out="$project/resources/tools/vship/vulkan"
work="$(mktemp -d "${TMPDIR:-/tmp}/vship-macos-XXXXXX")"
trap 'rm -rf "$work"' EXIT

checkout() {  # url commit folder
    git -c advice.detachedHead=false clone --quiet --filter=blob:none --no-checkout "$1" "$3"
    git -C "$3" -c advice.detachedHead=false checkout --quiet --force "$2"
}
fetch() {  # url sha256 file
    curl -sSfL -o "$3" "$1"
    local actual
    actual="$(shasum -a 256 "$3" | cut -d' ' -f1)"
    if [[ "$actual" != "$2" ]]; then
        echo "$(basename "$3") has SHA-256 $actual, expected $2" >&2
        exit 1
    fi
}

checkout https://codeberg.org/Line-fr/Vship.git "$vship_commit" "$work/Vship"
checkout https://github.com/KhronosGroup/Vulkan-Headers.git "$vulkan_headers_commit" "$work/Vulkan-Headers"
checkout https://github.com/KhronosGroup/Vulkan-Loader.git "$vulkan_loader_commit" "$work/Vulkan-Loader"
fetch "https://github.com/KhronosGroup/MoltenVK/releases/download/$moltenvk_version/MoltenVK-macos.tar" \
    "$moltenvk_sha256" "$work/MoltenVK-macos.tar"
fetch "https://github.com/shader-slang/slang/releases/download/v$slang_version/slang-$slang_version-macos-aarch64.tar.gz" \
    "$slang_sha256" "$work/slang.tar.gz"
mkdir -p "$work/slang" && tar -xzf "$work/slang.tar.gz" -C "$work/slang"
tar -xf "$work/MoltenVK-macos.tar" -C "$work"

# The loader.
cmake -S "$work/Vulkan-Headers" -B "$work/headers" -G Ninja -DCMAKE_INSTALL_PREFIX="$work/prefix" > /dev/null
cmake --install "$work/headers" > /dev/null
cmake -S "$work/Vulkan-Loader" -B "$work/loader" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$work/prefix" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" -DBUILD_TESTS=OFF \
    > "$work/loader.log" || { tail -30 "$work/loader.log" >&2; exit 1; }
cmake --build "$work/loader" > "$work/loader-build.log" || { tail -30 "$work/loader-build.log" >&2; exit 1; }

rm -rf "$vulkan_out"
mkdir -p "$vulkan_out/licenses" "$vship_out"
cp "$(find "$work/loader/loader" -name 'libvulkan.1.*.dylib' -type f | head -1)" "$vulkan_out/libvulkan.1.dylib"
install_name_tool -id @rpath/libvulkan.1.dylib "$vulkan_out/libvulkan.1.dylib" 2>/dev/null
# MoltenVK's universal build, its Apple silicon half.
lipo -thin arm64 "$work/MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib" -output "$vulkan_out/libMoltenVK.dylib"
cat > "$vulkan_out/MoltenVK_icd.json" <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "./libMoltenVK.dylib",
        "api_version": "1.4.0",
        "is_portability_driver": true
    }
}
EOF
cp "$work/MoltenVK/LICENSE" "$vulkan_out/licenses/LICENSE.moltenvk.txt"
cp "$work/Vulkan-Loader/LICENSE.txt" "$vulkan_out/licenses/LICENSE.vulkan-loader.txt"

# Vship, its two shaders compiled again from the patched source.
cd "$work/Vship"
git apply "$project/scripts/patches/vship-moltenvk-addresses.patch"
slangc="$work/slang/bin/slangc"
"$slangc" src/Vulkan/util/shaders/Planed.slang -O2 -target spirv -profile spirv_1_3 -emit-spirv-directly \
    -fvk-use-entrypoint-name -entry StrideEliminator -entry StrideAdder -o libvshipSpvShaders/Planed.spv
"$slangc" src/Vulkan/gpuColorToLinear/shaders/anyDepthToFloat.slang -O2 -target spirv -profile spirv_1_3 \
    -emit-spirv-directly -fvk-use-entrypoint-name -entry convertToFloatPlane -o libvshipSpvShaders/anyDepthToFloat.spv
clang++ src/Vulkan/spvFileToCppHeader.cpp -std=c++17 -O2 -o shaderEmbedder
./shaderEmbedder libvshipSpvShaders include/libvshipSpvShaders.hpp
version=()
for part in MAJOR MINOR MINORMINOR; do
    version+=("$(sed -n "s/^VSHIP_VERSION_$part=\([0-9]*\)/\1/p" Makefile)")
done
# The Makefile's Vulkan flags; no contraction of multiplies and adds, as the
# app's own code.
clang++ src/VshipLib.cpp -DVSHIP_VERSION_MAJOR="${version[0]}" -DVSHIP_VERSION_MINOR="${version[1]}" \
    -DVSHIP_VERSION_MINORMINOR="${version[2]}" -std=c++17 -I include -I "$work/Vulkan-Headers/include" -DNDEBUG -O3 \
    -DVULKANBUILD -w -arch arm64 -mmacosx-version-min="$deployment_target" -ffp-contract=off -fPIC -shared \
    "$vulkan_out/libvulkan.1.dylib" -install_name @rpath/libvship.dylib -Wl,-rpath,@loader_path/../../vulkan \
    -o "$vship_out/libvship.dylib"

for file in "$vulkan_out/libvulkan.1.dylib" "$vulkan_out/libMoltenVK.dylib" "$vship_out/libvship.dylib"; do
    codesign --force --sign - "$file" 2>/dev/null
    if otool -L "$file" | tail -n +2 | awk '{print $1}' | grep -v -E '^(@rpath/|/usr/lib/|/System/Library/)'; then
        echo "$(basename "$file") links a library that is neither bundled nor the system's" >&2
        exit 1
    fi
done
echo "Vulkan (loader $vulkan_loader_commit, MoltenVK $moltenvk_version) in $vulkan_out"
echo "Vship ${version[0]}.${version[1]}.${version[2]} (commit ${vship_commit:0:7}, patched for MoltenVK) in $vship_out"
shasum -a 256 "$vulkan_out"/*.dylib "$vship_out/libvship.dylib"
