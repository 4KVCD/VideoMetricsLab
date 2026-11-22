# vml_core: the engine. No Qt and no operating-system headers, so the same
# code runs under the command line and the window on every system.
add_library(vml_core
    src/analysis_request.cpp
    src/app_log.cpp
    src/av2_probe.cpp
    src/av2_stream.cpp
    src/bd_rate.cpp
    src/bd_rate_report.cpp
    src/bitrate.cpp
    src/bitrate_store.cpp
    src/black_bars_rule.cpp
    src/build_info.cpp
    src/cancel.cpp
    src/colour.cpp
    src/cpu_pair_scorer.cpp
    src/crop_detect.cpp
    src/csv_rows.cpp
    src/cvvdp.cpp
    src/difference.cpp
    src/export_csv.cpp
    src/ffmpeg_commands.cpp
    src/ffmpeg_download.cpp
    src/ffmpeg_run.cpp
    src/ffmpeg_tools.cpp
    src/ffprobe.cpp
    src/file_fingerprint.cpp
    src/film_grain_detect.cpp
    src/film_grain_scoring.cpp
    src/frame_extract.cpp
    src/frame_pairing.cpp
    src/frame_range.cpp
    src/frame_sync.cpp
    src/geometry.cpp
    src/gpu_decoders.cpp
    src/gpu_frames.cpp
    src/gpu_checks.cpp
    src/gpu_choice.cpp
    src/gpu_vmaf.cpp
    src/grain_feed.cpp
    src/hardware_decoding.cpp
    src/host_frames.cpp
    src/isolated.cpp
    src/job_runner.cpp
    src/libvmaf.cpp
    src/libvmaf_cpu.cpp
    src/metric_cache.cpp
    src/metric_cache_load.cpp
    src/metrics.cpp
    src/notifications.cpp
    src/model_select.cpp
    src/native_playback.cpp
    src/npz.cpp
    src/number_format.cpp
    src/options.cpp
    src/perceptual.cpp
    src/perceptual_cpu.cpp
    src/picture_placement.cpp
    src/python_json.cpp
    src/python_text.cpp
    src/range_seek.cpp
    src/raw_keyframes.cpp
    src/result_json.cpp
    src/results.cpp
    src/results_summary.cpp
    src/row_pool.cpp
    src/run_file.cpp
    src/run_json.cpp
    src/run_line.cpp
    src/run_record.cpp
    src/saved_results.cpp
    src/score_logs.cpp
    src/segment_presets.cpp
    src/segment_profiles.cpp
    src/settings.cpp
    src/sha256.cpp
    src/size_changes.cpp
    src/soundtrack_clock.cpp
    src/splot_library.cpp
    src/stats.cpp
    src/temp_files.cpp
    src/time_format.cpp
    src/update_check.cpp
    src/video.cpp
    src/video_playback.cpp
    src/vship.cpp
    src/vship_run.cpp
    src/weak_frames.cpp
    src/vmaf_cuda.cpp
    src/vmaf_hdr.cpp
    src/vmaf_v1_gpu.cpp
    src/vmaf_vulkan.cpp
    src/zip_archive.cpp
)
target_include_directories(vml_core PUBLIC include)
# The engines' vector code includes itself once per instruction set through
# Highway's foreach_target.h, by its name alone (HWY_TARGET_INCLUDE).
target_include_directories(vml_core PRIVATE src)

# The SSIMULACRA2 and Butteraugli engines. MSVC has no per-function target
# attributes, so the AVX2 code Highway compiles in a file built without
# /arch:AVX2 comes out slow; clang-cl compiles it as AVX2 code (4K frame, one
# thread, Core Ultra 9 285K: SSIMULACRA2 376.8 -> 266.9 ms, Butteraugli
# 919.9 -> 707.2 ms). Its objects link with MSVC's, and its scores are
# bit-identical to MSVC's: no fused multiply-adds the source does not ask
# for (-ffp-contract=off), and the same instruction sets as MSVC's build
# (AVX2 and SSE2: MSVC's Highway compiles no AVX-512, SSE4 or SSSE3 code).
# Without clang-cl (LLVM's installer, or Visual Studio's "C++ Clang tools"),
# MSVC compiles them: same scores, slower.
set(vml_engine_sources
    src/butteraugli.cpp src/butteraugli_kernels.cpp src/jxl_linear.cpp src/ssimulacra2.cpp
    src/ssimulacra2_kernels.cpp)
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    find_program(VML_CLANG_CL clang-cl
        HINTS "$ENV{ProgramFiles}/LLVM/bin" "$ENV{LOCALAPPDATA}/Programs/LLVM/bin"
        DOC "clang-cl, for the SSIMULACRA2 and Butteraugli engines' vector code (core/CMakeLists.txt)")
endif()
if(VML_CLANG_CL)
    # Files compiled one at a time by custom commands otherwise (CMake 3.27+).
    if(POLICY CMP0147)
        cmake_policy(SET CMP0147 NEW)
    endif()
    cmake_path(SET vml_hwy_dir NORMALIZE "${PROJECT_SOURCE_DIR}/third_party/highway")
    foreach(source IN LISTS vml_engine_sources)
        cmake_path(GET source STEM name)
        set(object ${CMAKE_CURRENT_BINARY_DIR}/clang/$<CONFIG>/${name}.obj)
        set(depfile ${CMAKE_CURRENT_BINARY_DIR}/clang/$<CONFIG>/${name}.d)
        add_custom_command(OUTPUT ${object}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/clang/$<CONFIG>
            COMMAND ${VML_CLANG_CL} /nologo /c /std:c++20 /EHsc /permissive- /utf-8 /W4 /WX
                /clang:-ffp-contract=off
                "/DHWY_DISABLED_TARGETS=(HWY_AVX3|(HWY_AVX3-1)|HWY_SSE4|HWY_SSSE3)"
                /I ${CMAKE_CURRENT_SOURCE_DIR}/include /I ${CMAKE_CURRENT_SOURCE_DIR}/src /imsvc ${vml_hwy_dir}
                "$<IF:$<CONFIG:Debug>,/MDd;/Od;/Z7,/MD;/O2;/DNDEBUG>"
                /clang:-MD /clang:-MF${depfile} /Fo${object} ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            DEPFILE ${depfile}
            COMMENT "clang-cl ${source}"
            COMMAND_EXPAND_LISTS VERBATIM)
        target_sources(vml_core PRIVATE ${object})
    endforeach()
else()
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(WARNING "clang-cl not found: MSVC compiles the SSIMULACRA2 and Butteraugli engines, with the "
            "same scores but slower (core/CMakeLists.txt). Install LLVM (winget install LLVM.LLVM) and "
            "delete CMakeCache.txt, or set VML_CLANG_CL.")
    endif()
    target_sources(vml_core PRIVATE ${vml_engine_sources})
    # No SVE code on Arm: the kernels keep filter state in arrays and structs
    # of vectors, which SVE's vectors (no size known when compiling) cannot
    # be, so clang refuses them. NEON compiles them on every Arm CPU, and no
    # Apple chip has SVE.
    set_property(SOURCE ${vml_engine_sources} APPEND PROPERTY COMPILE_DEFINITIONS
        "HWY_DISABLED_TARGETS=HWY_ALL_SVE")
endif()
# The app's version, for every part of it: shown, compared by the update
# check with GitHub's latest release, and written into what the app saves
# (vml/build_info.h).
target_compile_definitions(vml_core PUBLIC VML_VERSION="${PROJECT_VERSION}")
# The commit the app is built from (build_info.h's build_commit): asked of
# git at every build, its file rewritten only when the commit changed.
set(vml_build_commit_file ${CMAKE_CURRENT_BINARY_DIR}/generated/build_commit.cpp)
add_custom_target(vml_build_commit
    COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR} -DOUTPUT=${vml_build_commit_file}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/build_commit.cmake
    BYPRODUCTS ${vml_build_commit_file}
    VERBATIM)
target_sources(vml_core PRIVATE ${vml_build_commit_file})
add_dependencies(vml_core vml_build_commit)
# The decoders' scaler (native/scale/scale_filter.h), which gpu_frames.cpp
# checks FFmpeg's scale filter against.
target_include_directories(vml_core PRIVATE ../../native/scale)
# The film grain rule the scoring decoder and Video Compare's decode by
# (native/grain/grain_decoding.h, plain C++).
target_include_directories(vml_core PUBLIC ../../native/grain)
# splot's structures as the app mirrors them (native/av2/splot_api.h), for
# decoding AV2 videos (splot_library.h); public with that header.
target_include_directories(vml_core PUBLIC ../../native)
# libvmaf-fast's public headers, at the commit of the library the app ships
# (third_party/README.md). SYSTEM: their warnings are not ours to fix.
target_include_directories(vml_core SYSTEM PUBLIC ../third_party/libvmaf-fast/include)
target_link_libraries(vml_core PUBLIC vml_platform nlohmann_json PRIVATE vml_compile_options vml_miniz vml_hwy)
if(APPLE)
    # Which codecs the Mac's GPU decodes (hardware_decoding.cpp).
    target_link_libraries(vml_core PUBLIC "-framework VideoToolbox" "-framework CoreMedia")
endif()
# The app's own files (VMAF models, bundled tools) in a development build:
# the repository's resources folder. An installed app sets VML_RESOURCES.
cmake_path(SET resources NORMALIZE "${PROJECT_SOURCE_DIR}/../resources")
target_compile_definitions(vml_core PRIVATE VML_DEFAULT_RESOURCES="${resources}")
