#!/bin/bash
# Builds the app's self-contained macOS distributable, VideoMetricsLab.app
# (arm64, macOS 14.4 or later), as scripts/build_release.ps1 builds
# Windows': the window (Contents/MacOS/VideoMetricsLab) and the command line
# (Contents/MacOS/VideoMetricsLab-cli), Qt in Contents/Frameworks and
# Contents/PlugIns, and the app's own files in Contents/Resources/resources
# (the VMAF models, the translations, the bundled tools), where the programs
# look for them (packaged_resource_folder() in cpp/core/src/model_select.cpp).
#
#     scripts/build_release_macos.sh [--qt QT_FOLDER] [--allow-local-libvmaf]
#
# QT_FOLDER is Qt's macos folder (such as ~/Qt/6.12.0/macos), for a build
# folder not yet configured with it. Needs what building the app needs
# (cpp/README.md), git, meson and Ninja for libvmaf-fast, and ffmpeg and
# ffprobe on the PATH for the self-test. Output goes outside the
# repository, under ~/Library/Caches/VideoMetricsLab-build (or OUTPUT_ROOT).
set -euo pipefail

qt_folder=""
allow_local_libvmaf=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --qt) qt_folder="$2"; shift 2 ;;
        --allow-local-libvmaf) allow_local_libvmaf=true; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

project="$(cd "$(dirname "$0")/.." && pwd)"
cpp="$project/cpp"
app_files="$project/resources"
output_root="${OUTPUT_ROOT:-$HOME/Library/Caches/VideoMetricsLab-build}"
step() { printf '\033[36m==> %s\033[0m\n' "$1"; }

# 0. A release takes a release of libvmaf-fast (docs/RELEASING.md), not a
#    local build of the fork; and the self-test needs FFmpeg.
if ! grep -q '"release": true' "$app_files/tools/libvmaf/libvmaf-fast.json" && ! $allow_local_libvmaf; then
    echo "libvmaf-fast is a local build: install its release (docs/RELEASING.md), or pass --allow-local-libvmaf to test" >&2
    exit 1
fi
ffmpeg="$(command -v ffmpeg || true)"
if [[ -z "$ffmpeg" ]] || ! command -v ffprobe > /dev/null; then
    echo "The self-test needs ffmpeg and ffprobe on the PATH" >&2
    exit 1
fi

# 1. The bundled tools, from their pinned releases' source: libvmaf-fast's
#    libvmaf.dylib, libjxl's two tools, and Video Compare's FFmpeg libraries
#    and dav1d (not in git).
step "Building libvmaf-fast"
"$project/scripts/build_libvmaf_fast_macos.sh" | sed 's/^/    /'
step "Building libjxl's ssimulacra2 and butteraugli_main"
"$project/scripts/build_libjxl_macos.sh" | sed 's/^/    /'
step "Building FFmpeg's libraries for Video Compare"
"$project/scripts/build_ffmpeg_decoders_macos.sh" | sed 's/^/    /'
step "Building dav1d"
"$project/scripts/build_dav1d_macos.sh" | sed 's/^/    /'
step "Building Vulkan (the loader and MoltenVK) and Vship for the GPU metrics"
"$project/scripts/build_vship_macos.sh" | sed 's/^/    /'
step "Building libvmaf-fast's Vulkan engine for VMAF on the GPU"
"$project/scripts/build_vmaf_vulkan_macos.sh" | sed 's/^/    /'

# 2. The programs, in their Release build.
step "Building the C++ app"
configure=(--preset macos)
if [[ -n "$qt_folder" ]]; then
    configure+=("-DCMAKE_PREFIX_PATH=$qt_folder")
fi
(cd "$cpp" && cmake "${configure[@]}" > /dev/null && cmake --build --preset macos-release --target VideoMetricsLab vml-cli) \
    | sed 's/^/    /'
build="$cpp/build/macos"
built_app="$build/app/Release/VideoMetricsLab.app"
if [[ ! -d "$built_app" ]]; then
    echo "CMake built no window: it did not find Qt. Name its folder with --qt" >&2
    exit 1
fi
# Qt6_DIR is <Qt>/lib/cmake/Qt6.
qt_dir="$(sed -n 's/^Qt6_DIR:[A-Z]*=//p' "$build/CMakeCache.txt" | head -1)"
if [[ -z "$qt_dir" ]]; then
    echo "The build folder's CMakeCache.txt does not say where Qt is (Qt6_DIR): name its folder with --qt" >&2
    exit 1
fi
qt="$(cd "$qt_dir/../../.." && pwd)"

# 3. The bundle: the programs, Qt and the app's files.
step "Assembling"
rm -rf "$output_root/VideoMetricsLab.app"
mkdir -p "$output_root"
app="$output_root/VideoMetricsLab.app"
ditto "$built_app" "$app"
contents="$app/Contents"
# Named as the Windows package's command line is.
cp "$build/cli/Release/vml-cli" "$contents/MacOS/VideoMetricsLab-cli"

# Qt's frameworks and plugins (macdeployqt), less the plugins the app does
# not use, as on Windows: the TLS backends other than Apple's own (Secure
# Transport, which the update check uses; OpenSSL's needs an OpenSSL not
# bundled), the plugin that tells whether the Mac is online, and Qt's
# tools' own.
"$qt/bin/macdeployqt" "$app" -verbose=1 -executable="$contents/MacOS/VideoMetricsLab-cli" 2>&1 | sed 's/^/    /'
rm -rf "$contents/PlugIns/networkinformation" "$contents/PlugIns/tls/libqcertonlybackend.dylib" \
    "$contents/PlugIns/tls/libqopensslbackend.dylib"

# The bundle's Mach-O files: the programs, Qt's frameworks and plugins, and
# the bundled tools.
mach_o_files() {
    find "$contents" -type f \( -perm -u+x -o -name '*.dylib' \) | while IFS= read -r f; do
        if file -b "$f" | grep -q 'Mach-O'; then
            echo "$f"
        fi
    done
}
# Qt's are universal: only their Apple silicon half, as the app is.
while IFS= read -r file; do
    if lipo -archs "$file" | grep -qw x86_64; then
        lipo -thin arm64 "$file" -output "$file.arm64" && mv "$file.arm64" "$file"
    fi
done < <(mach_o_files)

# Qt's translations of its own dialogs and buttons, in the languages the
# window has and Qt has translated (cpp/app/CMakeLists.txt chooses them so
# on Windows), where qt.conf says they are.
mkdir -p "$contents/Resources/translations"
found_qm=false
for catalog in "$app_files"/translations/*.json; do
    code="$(basename "$catalog" .json)"
    if [[ -f "$qt/translations/qtbase_$code.qm" ]]; then
        cp "$qt/translations/qtbase_$code.qm" "$contents/Resources/translations/"
        found_qm=true
    fi
done
if ! $found_qm; then
    echo "Qt has no translations: install Qt's qttranslations with Qt" >&2
    exit 1
fi
cat > "$contents/Resources/qt.conf" <<'EOF'
[Paths]
Plugins = PlugIns
Translations = Resources/translations
EOF

# The app's files: the models, the window's catalogs, and macOS's tools
# (not Windows' .exe and .dll files, which share the folders).
resources="$contents/Resources/resources"
mkdir -p "$resources/translations" "$resources/tools/libjxl" "$resources/tools/libvmaf" "$resources/tools/ffmpeg" \
    "$resources/tools/vship/vulkan" "$resources/tools/vship/vship51" "$resources/tools/vmaf_vulkan"
ditto "$app_files/models" "$resources/models"
cp "$app_files"/translations/*.json "$resources/translations/"
cp "$app_files/tools/libjxl/ssimulacra2" "$app_files/tools/libjxl/butteraugli_main" \
    "$app_files/tools/libjxl/LIBJXL-VERSION.txt" "$app_files"/tools/libjxl/LICENSE.*.txt "$resources/tools/libjxl/"
cp "$app_files/tools/libvmaf/libvmaf.dylib" "$app_files/tools/libvmaf/libvmaf-fast.json" "$resources/tools/libvmaf/"
ditto "$app_files/tools/libvmaf/licenses" "$resources/tools/libvmaf/licenses"
cp "$app_files"/tools/ffmpeg/*.dylib "$resources/tools/ffmpeg/"
ditto "$app_files/tools/ffmpeg/licenses" "$resources/tools/ffmpeg/licenses"
# The GPU metrics: Vulkan (Khronos' loader with MoltenVK), Vship and VMAF's engine.
ditto "$app_files/tools/vulkan" "$resources/tools/vulkan"
cp "$app_files/tools/vship/vulkan/libvship.dylib" "$resources/tools/vship/vulkan/"
cp "$app_files/tools/vship/vship51/libvship.dylib" "$resources/tools/vship/vship51/"  # stock 5.1.2, the fallback
ditto "$app_files/tools/vship/licenses" "$resources/tools/vship/licenses"
cp "$app_files/tools/vmaf_vulkan/vmaf_vulkan.dylib" "$resources/tools/vmaf_vulkan/"
ditto "$app_files/tools/vmaf_vulkan/licenses" "$resources/tools/vmaf_vulkan/licenses"
# The window finds them by @rpath in the bundle's tools/ffmpeg, where the app
# looks for them (native_device_mac.mm), as a build run from its folder finds
# the repository's. macdeployqt copied them into Frameworks and linked the
# window to its copies: the links go back to @rpath and the copies go. It
# also dropped the repository's run path, which would name the Mac that
# built it.
for library in "$app_files"/tools/ffmpeg/lib{avcodec,avformat,avutil,swresample,swscale}.*.dylib; do
    name="$(basename "$library")"
    install_name_tool -change "@loader_path/../Frameworks/$name" "@rpath/$name" "$contents/MacOS/VideoMetricsLab"
    rm -f "$contents/Frameworks/$name"
done
install_name_tool -delete_rpath "$cpp/../resources/tools/ffmpeg" "$contents/MacOS/VideoMetricsLab" 2>/dev/null || true
if otool -l "$contents/MacOS/VideoMetricsLab" | grep -A2 LC_RPATH | grep -q "$HOME"; then
    echo "The window still has a run path of this Mac's" >&2
    exit 1
fi

# The project's terms, the notices of what it bundles, Qt's licence, and the
# MIT notices of nlohmann/json and miniz, compiled into both programs.
cp "$project/LICENSE" "$contents/Resources/LICENSE.txt"
cp "$project/docs/THIRD_PARTY.md" "$contents/Resources/THIRD_PARTY.md"
ditto "$cpp/packaging/licenses" "$contents/Resources/licenses"
cp "$cpp/third_party/nlohmann/LICENSE.MIT" "$contents/Resources/licenses/LICENSE.nlohmann-json.txt"
cp "$cpp/third_party/miniz/LICENSE" "$contents/Resources/licenses/LICENSE.miniz.txt"

# Everything each Mach-O file links must be in the bundle or be the
# system's. (A library's list starts with its own name, otool -D's.)
while IFS= read -r file; do
    own_name="$(otool -D "$file" | tail -n +2)"
    while IFS= read -r library; do
        [[ "$library" == "$own_name" ]] && continue
        case "$library" in
            /usr/lib/* | /System/Library/*) ;;
            @rpath/* | @loader_path/* | @executable_path/*) ;;
            *) echo "$(basename "$file") links $library, which is neither in the bundle nor the system's" >&2; exit 1 ;;
        esac
    done < <(otool -L "$file" | tail -n +2 | awk '{print $1}')
done < <(mach_o_files)

# Signed ad hoc: every Mach-O file inside first (after macdeployqt's
# install_name_tool and lipo, which void a signature), then the bundle.
step "Signing (ad hoc)"
while IFS= read -r file; do
    codesign --force --sign - --timestamp=none "$file"
done < <(mach_o_files)
find "$contents/Frameworks" -maxdepth 1 -name '*.framework' -exec codesign --force --sign - --timestamp=none {} \;
codesign --force --sign - --timestamp=none "$app"
codesign --verify --deep --strict "$app"
echo "    codesign --verify --deep --strict: valid"

# 4. Prove it runs and finds everything it contains, in a profile of its own
#    (not the user's settings), with nothing outside the bundle to lean on:
#    no Qt or resources folder named by the environment, and a PATH of the
#    system's folders and FFmpeg's.
step "Self-test"
profile="$(mktemp -d "${TMPDIR:-/tmp}/vml-self-test-XXXXXX")"
trap 'rm -rf "$profile"' EXIT
clean_env=(env -i HOME="$profile" PATH="/usr/bin:/bin:/usr/sbin:/sbin:$(dirname "$ffmpeg")" TMPDIR="${TMPDIR:-/tmp}")
status=0
"${clean_env[@]}" "$contents/MacOS/VideoMetricsLab" --self-test --quiet > /dev/null 2>&1 || status=$?
report="$profile/.videometricslab/self-test.txt"
if [[ ! -f "$report" ]]; then
    echo "The self-test wrote no report; the program did not get as far as running it" >&2
    exit 1
fi
sed 's/^/    /' "$report"
if [[ $status -ne 0 ]]; then
    echo "The self-test reported a failure (see above)" >&2
    exit 1
fi

# The command line: it must start, find FFmpeg and say what calculates each
# metric.
step "Command line"
"${clean_env[@]}" "$contents/MacOS/VideoMetricsLab-cli" devices 2>&1 | sed 's/^/    /'

# 5. Zip it, as the Finder would (ditto keeps the bundle's links and
#    signatures).
step "Packaging"
zip="$output_root/VideoMetricsLab-macos-arm64.zip"
rm -f "$zip"
ditto -c -k --keepParent "$app" "$zip"

echo
printf '\033[32mBuilt %s bundle, %s zip\033[0m\n' "$(du -sh "$app" | cut -f1)" "$(du -h "$zip" | cut -f1)"
echo "  $app"
echo "  $zip"
echo
printf '\033[33mFFmpeg itself (ffmpeg, ffprobe) is not bundled: brew install ffmpeg.\033[0m\n'
