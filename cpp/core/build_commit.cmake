# Writes OUTPUT, a C++ file whose build_commit() gives the commit SOURCE_DIR
# is checked out at ("unknown" where git cannot say), with "-dirty" after it
# when tracked files differ from that commit: a build with uncommitted
# changes is not that commit's. Run at every build (core/CMakeLists.txt's
# vml_build_commit; the content check takes some 50 ms); the file is
# rewritten only when its text changes, so only a new commit or the first
# change after one recompiles it.
find_program(GIT_PROGRAM git)
set(commit "unknown")
if(GIT_PROGRAM)
    execute_process(COMMAND "${GIT_PROGRAM}" -C "${SOURCE_DIR}" rev-parse --short=12 HEAD
                    OUTPUT_VARIABLE head RESULT_VARIABLE failed OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT failed AND head MATCHES "^[0-9a-f]+$")
        set(commit "${head}")
        # Exit code 1: tracked files differ (by content, not time stamps)
        execute_process(COMMAND "${GIT_PROGRAM}" -C "${SOURCE_DIR}" diff --quiet HEAD --
                        RESULT_VARIABLE changed OUTPUT_QUIET ERROR_QUIET)
        if(changed EQUAL 1)
            set(commit "${commit}-dirty")
        endif()
    endif()
endif()
set(text "// Written by core/build_commit.cmake at every build.
#include \"vml/build_info.h\"

namespace vml {

const char* build_commit() {
    return \"${commit}\";
}

}  // namespace vml
")
set(old "")
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" old)
endif()
if(NOT old STREQUAL text)
    file(WRITE "${OUTPUT}" "${text}")
endif()
