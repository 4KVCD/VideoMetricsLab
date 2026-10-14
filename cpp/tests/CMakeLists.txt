# The engine's and the command line's tests. Their expected values come from
# the Python app itself (reference/make_reference.py), so a test that passes
# says the C++ code gives the Python app's answers.
add_executable(vml_tests
    unit/cancel_test.cpp
    unit/cli_test.cpp
    unit/cvvdp_test.cpp
    unit/decisions_test.cpp
    unit/ffmpeg_run_test.cpp
    unit/ffmpeg_tools_test.cpp
    unit/ffprobe_test.cpp
    unit/frame_sync_by_time_test.cpp
    unit/gpu_frames_test.cpp
    unit/gpu_vmaf_test.cpp
    unit/isolated_test.cpp
    unit/job_runner_test.cpp
    unit/library_test.cpp
    unit/libvmaf_cpu_test.cpp
    unit/libvmaf_test.cpp
    unit/metric_cache_test.cpp
    unit/metrics_test.cpp
    unit/number_format_test.cpp
    unit/perceptual_cpu_test.cpp
    unit/process_test.cpp
    unit/python_json_test.cpp
    unit/run_file_test.cpp
    unit/run_line_test.cpp
    unit/settings_test.cpp
    unit/sha256_test.cpp
    unit/stats_test.cpp
    unit/temp_files_test.cpp
    unit/vmaf_cuda_test.cpp
    unit/vmaf_v1_gpu_test.cpp
    unit/vmaf_vulkan_test.cpp
    unit/vship_test.cpp
    unit/reference.cpp
    unit/test_videos.cpp
)
target_link_libraries(vml_tests PRIVATE vml_cli vml_core vml_compile_options nlohmann_json gtest gtest_main)
target_compile_definitions(vml_tests PRIVATE
    VML_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference"
    VML_TEST_CHILD="$<TARGET_FILE:vml_test_child>"
    VML_TEST_LIBRARY="$<TARGET_FILE:vml_test_library>"
    VML_CLI="$<TARGET_FILE:vml-cli>")  # the program isolated work runs in

# A program for the process tests to start: it echoes, fails, or sleeps.
add_executable(vml_test_child unit/test_child.cpp)
target_link_libraries(vml_test_child PRIVATE vml_compile_options)
if(MINGW)
    target_link_options(vml_test_child PRIVATE -municode)  # its wmain, for Unicode arguments
endif()

# A library for the library test to load.
add_library(vml_test_library SHARED unit/test_library.cpp)
target_link_libraries(vml_test_library PRIVATE vml_compile_options)
add_dependencies(vml_tests vml_test_child vml_test_library vml-cli)

include(GoogleTest)
gtest_discover_tests(vml_tests)

# The window's tests, where the window is built (app/CMakeLists.txt). They
# run with no screen, each in a profile of its own (app/main.cpp).
if(TARGET vml_app)
    find_package(Qt6 REQUIRED COMPONENTS Test)
    add_executable(vml_app_tests
        app/i18n_test.cpp
        app/main.cpp
        app/main_window_test.cpp
        unit/reference.cpp
        unit/test_videos.cpp
    )
    target_include_directories(vml_app_tests PRIVATE unit)
    target_link_libraries(vml_app_tests PRIVATE vml_app Qt6::Test vml_compile_options nlohmann_json gtest)
    target_compile_definitions(vml_app_tests PRIVATE
        VML_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference"
        VML_CLI="$<TARGET_FILE:vml-cli>")  # the program isolated work runs in
    add_dependencies(vml_app_tests vml-cli)
    if(WIN32)
        # Qt beside the tests, with the platform they run on; before the
        # tests are listed, which starts them.
        add_custom_command(TARGET vml_app_tests POST_BUILD
            COMMAND Qt6::windeployqt --no-translations --no-system-d3d-compiler --no-opengl-sw
                    --no-compiler-runtime --include-plugins qoffscreen $<TARGET_FILE:vml_app_tests>
            VERBATIM)
    endif()
    gtest_discover_tests(vml_app_tests)
endif()
