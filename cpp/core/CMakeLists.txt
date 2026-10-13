# vml_core: the engine. No Qt and no operating-system headers, so the same
# code runs under the command line and the window on every system.
add_library(vml_core
    src/crop_detect.cpp
    src/export_csv.cpp
    src/ffmpeg_run.cpp
    src/ffmpeg_tools.cpp
    src/ffprobe.cpp
    src/frame_sync.cpp
    src/geometry.cpp
    src/gpu_frames.cpp
    src/gpu_vmaf.cpp
    src/hardware_decoding.cpp
    src/isolated.cpp
    src/job_runner.cpp
    src/libvmaf.cpp
    src/libvmaf_cpu.cpp
    src/metrics.cpp
    src/model_select.cpp
    src/number_format.cpp
    src/options.cpp
    src/python_json.cpp
    src/python_text.cpp
    src/results.cpp
    src/run_line.cpp
    src/settings.cpp
    src/sha256.cpp
    src/stats.cpp
    src/time_format.cpp
    src/video.cpp
    src/vmaf_vulkan.cpp
)
target_include_directories(vml_core PUBLIC include)
# libvmaf-fast's public headers, at the commit of the library the app ships
# (third_party/README.md). SYSTEM: their warnings are not ours to fix.
target_include_directories(vml_core SYSTEM PUBLIC ../third_party/libvmaf-fast/include)
target_link_libraries(vml_core PUBLIC vml_platform nlohmann_json PRIVATE vml_compile_options)
# The app's own files (VMAF models, bundled tools) in a development build:
# the repository's vmaf_app folder. An installed app sets VML_RESOURCES.
cmake_path(SET resources NORMALIZE "${PROJECT_SOURCE_DIR}/../vmaf_app")
target_compile_definitions(vml_core PRIVATE VML_DEFAULT_RESOURCES="${resources}")
