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
    key_guide.cpp
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
    self_test.cpp
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
# The version: shown, and compared by the update check with GitHub's latest
# release.
target_compile_definitions(vml_app PUBLIC VML_APP_VERSION="${PROJECT_VERSION}")

if(WIN32 AND MSVC)
    # Video Compare's native playback: FFmpeg's libraries (resources/tools/
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
    target_sources(vml_app PRIVATE native_dav1d.cpp native_decoder.cpp native_device.cpp native_grain.cpp
                                   native_pool.cpp native_presentation.cpp native_soundtrack.cpp)
    # dav1d's structures as the app mirrors them (native_dav1d.h).
    target_include_directories(vml_app SYSTEM PRIVATE ${vml_ffmpeg}/include ${PROJECT_SOURCE_DIR}/../native/dav1d)
    target_link_libraries(vml_app PUBLIC ${vml_ffmpeg_imports} delayimp d3d11 dxgi dxguid)
    # VML_NATIVE_PLAYBACK public: the view's members depend on it, and the
    # tests must see the same view. NOMINMAX: no min and max macros from
    # Windows' headers.
    target_compile_definitions(vml_app PUBLIC VML_NATIVE_PLAYBACK PRIVATE NOMINMAX)
endif()

# Video Compare's native playback on macOS: FFmpeg's libraries with
# VideoToolbox's decoding and dav1d (resources/tools/ffmpeg, built by
# scripts/build_ffmpeg_decoders_macos.sh and build_dav1d_macos.sh, not
# committed), Metal to draw and CoreAudio for the soundtrack. Built where
# FFmpeg's libraries have been built; else the window plays with FFmpeg's
# programs. They are linked weakly, by @rpath: the window starts without
# them (native_device_mac.mm then says they are missing), and finds them in
# the bundle (Contents/Resources/resources/tools/ffmpeg) or, in a build run
# from its folder, in the repository's.
set(vml_ffmpeg_dylibs libavcodec.63 libavformat.63 libavutil.61 libswresample.7 libswscale.10)
set(vml_ffmpeg_built ${PROJECT_SOURCE_DIR}/../resources/tools/ffmpeg)
set(vml_ffmpeg_found TRUE)
foreach(library ${vml_ffmpeg_dylibs})
    if(NOT EXISTS ${vml_ffmpeg_built}/${library}.dylib)
        set(vml_ffmpeg_found FALSE)
    endif()
endforeach()
if(APPLE AND NOT vml_ffmpeg_found)
    message(STATUS "FFmpeg's libraries are not built: Video Compare plays with FFmpeg's programs "
                   "(scripts/build_ffmpeg_decoders_macos.sh builds them)")
elseif(APPLE)
    enable_language(OBJCXX)
    set(vml_ffmpeg ${PROJECT_SOURCE_DIR}/../native/ffmpeg)
    target_sources(vml_app PRIVATE native_dav1d.cpp native_decoder.cpp native_device_mac.mm native_grain_mac.cpp
                                   native_pool.cpp native_presentation_mac.mm native_soundtrack.cpp)
    set_source_files_properties(native_device_mac.mm native_presentation_mac.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)
    # The headers are Windows' build's (native/ffmpeg), the same for this
    # version: avconfig.h's endianness and alignment are the same on arm64.
    target_include_directories(vml_app SYSTEM PRIVATE ${vml_ffmpeg}/include ${PROJECT_SOURCE_DIR}/../native/dav1d)
    foreach(library ${vml_ffmpeg_dylibs})
        target_link_options(vml_app INTERFACE "LINKER:-weak_library,${vml_ffmpeg_built}/${library}.dylib")
    endforeach()
    # The repository's copy, for a build run from its folder (the release
    # script deletes this run path from the bundle's program).
    target_link_options(vml_app INTERFACE "LINKER:-rpath,${vml_ffmpeg_built}")
    target_link_libraries(vml_app PUBLIC "-framework AppKit" "-framework AudioToolbox" "-framework CoreAudio"
                                         "-framework CoreMedia" "-framework CoreVideo" "-framework Metal"
                                         "-framework QuartzCore" "-framework VideoToolbox")
    target_compile_definitions(vml_app PUBLIC VML_NATIVE_PLAYBACK)
    # For the tests (tests/CMakeLists.txt).
    set_property(GLOBAL PROPERTY vml_native_playback_macos TRUE)
endif()

add_executable(VideoMetricsLab WIN32 MACOSX_BUNDLE main.cpp)
target_link_libraries(VideoMetricsLab PRIVATE vml_app vml_compile_options)
if(APPLE)
    # An app bundle, VideoMetricsLab.app; built for the macOS the presets set
    # (CMAKE_OSX_DEPLOYMENT_TARGET: 14.4, Qt 6.12's earliest).
    set_target_properties(VideoMetricsLab PROPERTIES
        MACOSX_BUNDLE_INFO_PLIST ${PROJECT_SOURCE_DIR}/packaging/macos/Info.plist.in)
    # The app's icon, in Contents/Resources (Info.plist's CFBundleIconFile):
    # Finder's and the Dock's.
    target_sources(VideoMetricsLab PRIVATE icons/VideoMetricsLab.icns)
    set_source_files_properties(icons/VideoMetricsLab.icns PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
    # The bundle's FFmpeg libraries, for native playback.
    target_link_options(VideoMetricsLab PRIVATE "LINKER:-rpath,@executable_path/../Resources/resources/tools/ffmpeg")
endif()
if(MSVC)
    # Long paths, as vml-cli (cli/CMakeLists.txt); and the app's icon.
    target_sources(VideoMetricsLab PRIVATE ../cli/vml-cli.manifest icons/VideoMetricsLab.rc)
endif()

if(WIN32)
    # Qt's translations of its own dialogs and buttons (qtbase_<code>.qm), for
    # each language the window has a catalog for (resources/translations) and
    # Qt has translated, put in the translations folder beside `target`: Qt
    # looks there for a program with its libraries beside it (QLibraryInfo;
    # install_qt_translations in i18n.h). The package takes the program's.
    # windeployqt's own would join every module's catalogs with lconvert,
    # which a Qt of qtbase alone does not have. Nothing where Qt's
    # translations (qttranslations) are not installed.
    function(vml_deploy_qt_translations target)
        set(qt_translations "${QT6_INSTALL_PREFIX}/${QT6_INSTALL_TRANSLATIONS}")
        file(GLOB catalogs "${PROJECT_SOURCE_DIR}/../resources/translations/*.json")
        set(files "")
        foreach(catalog IN LISTS catalogs)
            get_filename_component(code "${catalog}" NAME_WE)
            if(EXISTS "${qt_translations}/qtbase_${code}.qm")
                list(APPEND files "${qt_translations}/qtbase_${code}.qm")
            endif()
        endforeach()
        if(files)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/translations"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${files} "$<TARGET_FILE_DIR:${target}>/translations"
                VERBATIM)
        endif()
    endfunction()

    # Qt's libraries and plugins beside the program, so that it runs from the
    # build folder, the same ones as the packaged app's
    # (scripts/build_release.ps1 says why the others are left out), and
    # Qt's translations.
    add_custom_command(TARGET VideoMetricsLab POST_BUILD
        COMMAND Qt6::windeployqt --no-translations --no-system-d3d-compiler --no-system-dxc-compiler
                --no-opengl-sw --no-compiler-runtime --skip-plugin-types generic,networkinformation
                --exclude-plugins qcertonlybackend $<TARGET_FILE:VideoMetricsLab>
        VERBATIM)
    vml_deploy_qt_translations(VideoMetricsLab)
endif()
