# The window, ported from vmaf_app/ui, vmaf_app/main.py and vmaf_app/i18n.py:
# its code (vml_app, which the tests use too) and the program,
# VideoMetricsLab. Built where Qt 6.5 or newer is found (CMAKE_PREFIX_PATH
# names Qt's folder: README.md); without Qt, the rest is built and the window
# is left out.
find_package(Qt6 6.5 QUIET COMPONENTS Concurrent Network Widgets)
if(NOT Qt6_FOUND)
    message(STATUS "Qt 6 not found: the window is not built (CMAKE_PREFIX_PATH names Qt's folder)")
    return()
endif()

add_library(vml_app STATIC
    bitrate_panel.cpp
    chart.cpp
    cvvdp_display_dialog.cpp
    dialogs.cpp
    file_writes.cpp
    formatting.cpp
    frame_compare_panel.cpp
    graph_panel.cpp
    i18n.cpp
    main_window.cpp
    main_window_compare.cpp
    main_window_graph.cpp
    main_window_options.cpp
    main_window_run.cpp
    main_window_run_files.cpp
    main_window_saved_scores.cpp
    options_panel.cpp
    playback_worker.cpp
    probe_worker.cpp
    row_state.cpp
    run_worker.cpp
    series_colors.cpp
    settings_panel.cpp
    theme.cpp
    video_compare_view.cpp
    widgets.cpp
    zoom.cpp
)
set_target_properties(vml_app PROPERTIES AUTOMOC ON)
target_include_directories(vml_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vml_app PUBLIC vml_core Qt6::Concurrent Qt6::Network Qt6::Widgets PRIVATE vml_compile_options)
# The version shown, and the release it will be, which the update check
# compares with GitHub's latest.
target_compile_definitions(vml_app PUBLIC VML_APP_VERSION="${PROJECT_VERSION}-dev"
                                          VML_APP_RELEASE="${PROJECT_VERSION}")

if(WIN32 AND MSVC)
    # Video Compare's native playback: FFmpeg's libraries (vmaf_app/tools/
    # ffmpeg, built by scripts/build_ffmpeg_decoders.ps1, which vendors their
    # headers and exports into native/ffmpeg) and Direct3D 11. They are
    # linked through import libraries made from their exports, and loaded
    # only when first called (native_device.cpp loads them by their paths
    # first), so that the window starts without them.
    set(vml_ffmpeg ${PROJECT_SOURCE_DIR}/../native/ffmpeg)
    set(vml_ffmpeg_imports)
    foreach(library avcodec-63 avformat-63 avutil-61 swresample-7 swscale-10)
        set(import ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg/${library}.lib)
        add_custom_command(OUTPUT ${import}
            COMMAND ${CMAKE_AR} /nologo /machine:x64 /def:${vml_ffmpeg}/lib/${library}.def /name:${library}.dll
                    /out:${import}
            DEPENDS ${vml_ffmpeg}/lib/${library}.def
            VERBATIM)
        list(APPEND vml_ffmpeg_imports ${import})
        target_link_options(vml_app INTERFACE /DELAYLOAD:${library}.dll)
    endforeach()
    add_custom_target(vml_ffmpeg_imports DEPENDS ${vml_ffmpeg_imports})
    add_dependencies(vml_app vml_ffmpeg_imports)
    target_sources(vml_app PRIVATE native_decoder.cpp native_device.cpp native_pool.cpp native_presentation.cpp
                                   native_soundtrack.cpp)
    target_include_directories(vml_app SYSTEM PRIVATE ${vml_ffmpeg}/include)
    target_link_libraries(vml_app PUBLIC ${vml_ffmpeg_imports} delayimp d3d11 dxgi)
    # VML_NATIVE_PLAYBACK public: the view's members depend on it, and the
    # tests must see the same view. NOMINMAX: no min and max macros from
    # Windows' headers.
    target_compile_definitions(vml_app PUBLIC VML_NATIVE_PLAYBACK PRIVATE NOMINMAX)
endif()

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
