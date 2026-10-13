# vml_platform: what differs between the systems (processes, loading
# libraries, the PATH, the GPUs), behind headers that are the same on all of
# them.
add_library(vml_platform src/process.cpp)
if(WIN32)
    target_sources(vml_platform PRIVATE
        src/console_windows.cpp src/library_windows.cpp src/process_windows.cpp src/system_windows.cpp)
    target_link_libraries(vml_platform PRIVATE dxgi shell32)  # the GPUs' vendor IDs; the wide command line
else()
    target_sources(vml_platform PRIVATE
        src/console_posix.cpp src/library_posix.cpp src/process_posix.cpp src/system_posix.cpp)
    target_link_libraries(vml_platform PRIVATE ${CMAKE_DL_LIBS})  # dlopen, where it is not in libc
endif()
target_include_directories(vml_platform PUBLIC include)
target_link_libraries(vml_platform PRIVATE vml_compile_options)
find_package(Threads REQUIRED)
target_link_libraries(vml_platform PUBLIC Threads::Threads)
