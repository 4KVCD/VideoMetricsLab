# vml_core: the engine. No Qt and no operating-system headers, so the same
# code runs under the command line and the window on every system.
add_library(vml_core
    src/metrics.cpp
    src/number_format.cpp
    src/stats.cpp
)
target_include_directories(vml_core PUBLIC include)
target_link_libraries(vml_core PRIVATE vml_compile_options)
