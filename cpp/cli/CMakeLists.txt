# The command line, ported from vmaf_app/cli.py as far as compare.h says: its
# code (vml_cli, which the tests use too) and the program, vml-cli.
add_library(vml_cli STATIC
    arguments.cpp
    compare.cpp
    report.cpp
)
target_include_directories(vml_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vml_cli PUBLIC vml_core PRIVATE vml_compile_options)
target_compile_definitions(vml_cli PUBLIC VML_VERSION="${PROJECT_VERSION}-dev")

add_executable(vml-cli main.cpp)
target_link_libraries(vml-cli PRIVATE vml_cli vml_compile_options)
