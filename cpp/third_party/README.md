# Third-party code in the C++ port

Vendored, so that a build needs no download. GoogleTest and nlohmann/json
were downloaded once, with Brian's OK, and checked against the SHA-256 their
projects publish. libvmaf-fast's headers are copied from that project's own
repository at the commit of the release the app ships.

| Library | Version | Used for | Licence | Source | SHA-256 of the download |
| --- | --- | --- | --- | --- | --- |
| [GoogleTest](https://github.com/google/googletest) | 1.18.0 | The tests (not shipped) | BSD 3-Clause (`googletest/LICENSE`) | `googletest-1.18.0.tar.gz` from the v1.18.0 release | `6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5` |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.12.0 | Reading and writing JSON | MIT (`nlohmann/LICENSE.MIT`) | `json.hpp` from the v3.12.0 release | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` |
| [libvmaf-fast](https://github.com/4KVCD/libvmaf-fast) | v1 | Calling the libvmaf the app ships (`vmaf_app/tools/libvmaf`): its functions and structures | BSD-2-Clause-Patent (`libvmaf-fast/LICENSE`) | `libvmaf/include/libvmaf/*.h` and `LICENSE` at commit `867efb83` (tag `libvmaf-fast-v1`) | Not downloaded; each file's is below |

Only what a build needs is kept: of GoogleTest, the `googletest` folder's
`CMakeLists.txt`, `cmake`, `include` and `src` (no Google Mock, samples,
tests or docs); of nlohmann/json, the single header and its licence; of
libvmaf-fast, the five headers that declare its functions and its licence
(not `version.h.in` or `meson.build`, which only its own build uses).

libvmaf-fast's files, each identical to the file at `867efb83`:

| File | SHA-256 |
| --- | --- |
| `include/libvmaf/feature.h` | `a259aed55f548cd29976b546da215ea8cc98a200df904b76573e434ef8f626b6` |
| `include/libvmaf/libvmaf.h` | `dd9f640a97b03331244b8b26e91b2b2e494dbce4d31c6b483769a10b7ff9eb64` |
| `include/libvmaf/libvmaf_cuda.h` | `411c80b098e64c788f9ea703fb5d5a6eff4575dfaed186e3e47e640fafac60c2` |
| `include/libvmaf/model.h` | `4c25a226e4caefa7eca56c782167597fff1729c8caa1c145183c9479eca94180` |
| `include/libvmaf/picture.h` | `61c47904c8d241ebc1ed60c170b7c0a7715e04b47798c25298f969cb54a95ecc` |
| `LICENSE` | `a6fdb51d02b9dfe198e55534bb8047fe6d756eda117f4ff83de98f92c1952710` |

When the app ships a newer libvmaf-fast, copy the same files from that
release's commit and update this list.

nlohmann/json is compiled into the app, so it belongs in the app's
`docs/THIRD_PARTY.md` once the C++ app ships. GoogleTest is only in the tests.
libvmaf itself already ships with the app and is listed there.
