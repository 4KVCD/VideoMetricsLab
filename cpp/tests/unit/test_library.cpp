// The library the DynamicLibrary test loads: one exported function.
#ifdef _WIN32
#define VML_TEST_EXPORT extern "C" __declspec(dllexport)
#else
#define VML_TEST_EXPORT extern "C" __attribute__((visibility("default")))
#endif

VML_TEST_EXPORT int vml_test_add(int a, int b) { return a + b; }
