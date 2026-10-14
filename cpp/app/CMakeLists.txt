# The window, ported from vmaf_app/ui, vmaf_app/main.py and vmaf_app/i18n.py:
# its code (vml_app, which the tests use too) and the program,
# VideoMetricsLab. Built where Qt 6.5 or newer is found (CMAKE_PREFIX_PATH
# names Qt's folder: README.md); without Qt, the rest is built and the window
# is left out.
find_package(Qt6 6.5 QUIET COMPONENTS Concurrent Widgets)
if(NOT Qt6_FOUND)
    message(STATUS "Qt 6 not found: the window is not built (CMAKE_PREFIX_PATH names Qt's folder)")
    return()
endif()

add_library(vml_app STATIC
    chart.cpp
    cvvdp_display_dialog.cpp
    dialogs.cpp
    file_writes.cpp
    formatting.cpp
    graph_panel.cpp
    i18n.cpp
    main_window.cpp
    main_window_graph.cpp
    main_window_options.cpp
    main_window_run.cpp
    main_window_run_files.cpp
    main_window_saved_scores.cpp
    options_panel.cpp
    probe_worker.cpp
    row_state.cpp
    run_worker.cpp
    series_colors.cpp
    settings_panel.cpp
    theme.cpp
    widgets.cpp
)
set_target_properties(vml_app PROPERTIES AUTOMOC ON)
target_include_directories(vml_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vml_app PUBLIC vml_core Qt6::Concurrent Qt6::Widgets PRIVATE vml_compile_options)
target_compile_definitions(vml_app PUBLIC VML_APP_VERSION="${PROJECT_VERSION}-dev")

add_executable(VideoMetricsLab WIN32 main.cpp)
target_link_libraries(VideoMetricsLab PRIVATE vml_app vml_compile_options)
if(MSVC)
    # Long paths, as vml-cli (cli/CMakeLists.txt).
    target_sources(VideoMetricsLab PRIVATE ../cli/vml-cli.manifest)
endif()

if(WIN32)
    # Qt's libraries and plugins beside the program, so that it runs from the
    # build folder. Qt's own translations (its dialogs' buttons) come with the
    # packaged app: a Qt built from qtbase alone has none to deploy.
    add_custom_command(TARGET VideoMetricsLab POST_BUILD
        COMMAND Qt6::windeployqt --no-translations --no-system-d3d-compiler --no-opengl-sw
                --no-compiler-runtime $<TARGET_FILE:VideoMetricsLab>
        VERBATIM)
endif()
