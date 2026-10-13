# vml_platform: what differs between the systems (processes, the PATH, the
# GPUs), behind headers that are the same on all of them.
add_library(vml_platform src/process.cpp)
if(WIN32)
    target_sources(vml_platform PRIVATE src/process_windows.cpp src/system_windows.cpp)
    target_link_libraries(vml_platform PRIVATE dxgi)  # the GPUs' vendor IDs
else()
    target_sources(vml_platform PRIVATE src/process_posix.cpp src/system_posix.cpp)
endif()
target_include_directories(vml_platform PUBLIC include)
target_link_libraries(vml_platform PRIVATE vml_compile_options)
find_package(Threads REQUIRED)
target_link_libraries(vml_platform PUBLIC Threads::Threads)
