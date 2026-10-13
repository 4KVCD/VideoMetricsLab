#include "vml/platform/library.h"

#include <gtest/gtest.h>

namespace vml::platform {
namespace {

// A library is loaded from its file, its functions found by name and
// called; a missing function or file is an error that says which.
TEST(Library, LoadsALibraryAndCallsItsFunctions) {
    const DynamicLibrary library(VML_TEST_LIBRARY);
    const auto add = library.function<int (*)(int, int)>("vml_test_add");
    EXPECT_EQ(add(2, 3), 5);
    EXPECT_EQ(library.find("vml_no_such_function"), nullptr);
    EXPECT_THROW(library.function<int (*)()>("vml_no_such_function"), LibraryError);
    EXPECT_THROW(DynamicLibrary("vml-no-such-library-anywhere.dll"), LibraryError);
}

// A library of the system's is found by its name alone, where the system
// keeps it, as NVIDIA's CUDA driver is.
TEST(Library, LoadsASystemLibraryByName) {
#ifdef _WIN32
    const DynamicLibrary library = DynamicLibrary::system("kernel32.dll");
    EXPECT_NE(library.find("GetCurrentProcessId"), nullptr);
#elif defined(__APPLE__)
    const DynamicLibrary library = DynamicLibrary::system("libSystem.B.dylib");
    EXPECT_NE(library.find("getpid"), nullptr);
#else
    const DynamicLibrary library = DynamicLibrary::system("libc.so.6");
    EXPECT_NE(library.find("getpid"), nullptr);
#endif
    EXPECT_THROW(DynamicLibrary::system("vml-no-such-library-anywhere.dll"), LibraryError);
}

}  // namespace
}  // namespace vml::platform
