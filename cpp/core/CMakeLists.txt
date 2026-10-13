# vml_core: the engine. No Qt and no operating-system headers, so the same
# code runs under the command line and the window on every system.
add_library(vml_core
    src/ffmpeg_tools.cpp
    src/ffprobe.cpp
    src/metrics.cpp
    src/number_format.cpp
    src/python_text.cpp
    src/settings.cpp
    src/stats.cpp
    src/video.cpp
)
target_include_directories(vml_core PUBLIC include)
target_link_libraries(vml_core PUBLIC vml_platform PRIVATE vml_compile_options nlohmann_json)
