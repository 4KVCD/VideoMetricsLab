# VideoMetricsLab in C++: the engine (vml_core), the command line (vml-cli),
# the window (VideoMetricsLab, being ported) and their tests. README.md
# describes the layout, how to build, and the rules the code follows.
cmake_minimum_required(VERSION 3.25)
# C only for miniz (third_party/miniz): the app itself is C++.
project(VideoMetricsLab VERSION 3.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VML_BUILD_TESTS "Build the tests" ON)

# The options every target of ours is compiled with.
add_library(vml_compile_options INTERFACE)
if(MSVC)
    # /utf-8: the sources are UTF-8 (the "∞" and "—" the app shows).
    target_compile_options(vml_compile_options INTERFACE /W4 /WX /permissive- /utf-8)
else()
    target_compile_options(vml_compile_options INTERFACE -Wall -Wextra -Wpedantic -Werror
        # A multiply and an add fused into one instruction round once instead
        # of twice. Some compilers fuse them on their own (Clang on Apple's
        # chips), which would change the last digits of the statistics. MSVC
        # does not, and on Windows the tests ask for the Python app's numbers to
        # the last digit.
        -ffp-contract=off)
endif()

# nlohmann/json, vendored (third_party/README.md). SYSTEM: its warnings are
# not ours to fix.
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE third_party/nlohmann/include)

# miniz, vendored (third_party/README.md): the zip files and the deflate of
# the saved scores (.npz files, npz.h). Compiled without our warnings, which
# are not ours to fix, and without its file functions: the app reads and
# writes the files itself, so that every path works on Windows.
add_library(vml_miniz STATIC third_party/miniz/miniz.c)
target_include_directories(vml_miniz SYSTEM PUBLIC third_party/miniz)
target_compile_definitions(vml_miniz PUBLIC MINIZ_NO_STDIO MINIZ_NO_ZLIB_COMPATIBLE_NAMES)
set_target_properties(vml_miniz PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(platform)
add_subdirectory(core)
add_subdirectory(cli)
add_subdirectory(app)

if(VML_BUILD_TESTS)
    enable_testing()
    # GoogleTest, vendored (third_party/README.md), built with the same C
    # run-time library as our code (MSVC's /MD).
    set(GOOGLETEST_VERSION 1.18.0)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL SYSTEM)
    add_subdirectory(tests)
endif()
