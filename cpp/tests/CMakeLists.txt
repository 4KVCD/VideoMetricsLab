# The engine's tests. Their expected values come from the Python app itself
# (reference/make_reference.py), so a test that passes says the C++ code
# gives the Python app's answers.
add_executable(vml_tests
    unit/metrics_test.cpp
    unit/number_format_test.cpp
    unit/stats_test.cpp
    unit/reference.cpp
)
target_link_libraries(vml_tests PRIVATE vml_core vml_compile_options nlohmann_json gtest gtest_main)
target_compile_definitions(vml_tests PRIVATE VML_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference")

include(GoogleTest)
gtest_discover_tests(vml_tests)
