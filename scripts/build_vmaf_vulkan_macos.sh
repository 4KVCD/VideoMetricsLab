#!/bin/bash
# Builds libvmaf-fast's Vulkan VMAF engine for macOS (arm64),
# vmaf_vulkan.dylib, into resources/tools/vmaf_vulkan beside Windows'
# vmaf_vulkan.dll: VMAF v0.6.1, VMAF NEG and VMAF v1's features on the GPU,
# on MoltenVK through the loader build_vship_macos.sh bundles in tools/vulkan
# (the library's rpath, @loader_path/../vulkan). libvmaf-fast's own script,
# from the commit pinned here (its macos-port branch), builds it.
#
#     scripts/build_vmaf_vulkan_macos.sh
#
# Needs git, curl, python3 and the Command Line Tools' clang. The dylib is not
# committed (.gitignore): the release script runs this.
set -euo pipefail

commit="0efaa99d6a531403b855488dc884fe9f2d4b6c2c"  # libvmaf-fast macos-port, 2026-10-18

project="$(cd "$(dirname "$0")/.." && pwd)"
target="$project/resources/tools/vmaf_vulkan"
work="$(mktemp -d "${TMPDIR:-/tmp}/vmaf-vulkan-XXXXXX")"
trap 'rm -rf "$work"' EXIT

git -c advice.detachedHead=false clone --quiet --filter=blob:none --no-checkout \
    https://github.com/4KVCD/libvmaf-fast.git "$work/src"
git -C "$work/src" -c advice.detachedHead=false checkout --quiet --force "$commit"
"$work/src/fast/scripts/build_vmaf_vulkan_macos.sh" "$work/out" > "$work/build.log" 2>&1 \
    || { tail -30 "$work/build.log" >&2; exit 1; }

mkdir -p "$target"
cp "$work/out/vmaf_vulkan.dylib" "$target/vmaf_vulkan.dylib"
# The licence is the same file as Windows' (libvmaf's LICENSE).
cmp -s "$work/out/licenses/LICENSE.libvmaf.txt" "$target/licenses/LICENSE.libvmaf.txt" \
    || { echo "libvmaf-fast's LICENSE differs from resources/tools/vmaf_vulkan/licenses'" >&2; exit 1; }
echo "VMAF for Vulkan (libvmaf-fast ${commit:0:8}) in $target"
shasum -a 256 "$target/vmaf_vulkan.dylib"
