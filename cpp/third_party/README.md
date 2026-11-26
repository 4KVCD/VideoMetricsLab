# Third-party code in the C++ port

Vendored, so that a build needs no download. GoogleTest, nlohmann/json and
miniz were downloaded once, with Brian's OK, and checked against the SHA-256
their projects publish. SQLite was downloaded once from sqlite.org and
checked against the SHA3-256 its download page publishes. Highway, whose project publishes a PGP signature
instead, was checked against that: a good signature by its maintainer's key
(Jan Wassenberg, 14D8 27B2 2124 EB2F E748 C3B6 EE0F 9065 BCB8 B678, since
expired). libvmaf-fast's headers are copied from that
project's own repository at the commit of the release the app ships.

| Library | Version | Used for | Licence | Source | SHA-256 of the download |
| --- | --- | --- | --- | --- | --- |
| [GoogleTest](https://github.com/google/googletest) | 1.18.0 | The tests (not shipped) | BSD 3-Clause (`googletest/LICENSE`) | `googletest-1.18.0.tar.gz` from the v1.18.0 release | `6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5` |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.12.0 | Reading and writing JSON | MIT (`nlohmann/LICENSE.MIT`) | `json.hpp` from the v3.12.0 release | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` |
| [libvmaf-fast](https://github.com/4KVCD/libvmaf-fast) | v1.3 | Calling the libvmaf the app ships (`resources/tools/libvmaf`): its functions and structures | BSD-2-Clause-Patent (`libvmaf-fast/LICENSE`) | `libvmaf/include/libvmaf/*.h` and `LICENSE` at commit `28776652` (tag `libvmaf-fast-v1.3`), the same files as v1's (`867efb83`), v1.1's (`994b1814`) and v1.2's (`71b7b9be`) | Not downloaded; each file's is below |
| [Highway](https://github.com/google/highway) | 1.2.0 | The SIMD code of the app's SSIMULACRA2 and Butteraugli engines (`core/src/ssimulacra2_kernels.cpp`, `core/src/butteraugli_kernels.cpp`), one build per CPU type, chosen when the app runs | Apache 2.0 or BSD 3-Clause (`highway/LICENSE`, `highway/LICENSE-BSD3`) | `highway-1.2.0.tar.gz` (2,233,645 bytes) from the 1.2.0 release, the version libjxl 0.12.0 builds with | `58e9d5d41d6573ad15245ad76aec53a69499ca7480c092d899c4424812ed906f` |
| [miniz](https://github.com/richgel999/miniz) | 3.1.2 | The saved scores' .npz files: zip archives of deflated numpy arrays (`core/npz.h`) | MIT (`miniz/LICENSE`) | `miniz-3.1.2.zip` (101,958 bytes) from the 3.1.2 release | `f0446d863f9c19926ad9483c523fdc42e42b8d4a6a431d27e09d49c79a140d9a` |
| [SQLite](https://sqlite.org) | 3.54.1 | The saved results' index (`core/saved_index.h`): one database the window and `vml-cli` share | Public domain (the notice at the top of `sqlite3.c`; no licence file) | `sqlite-amalgamation-3540100.zip` (2,990,966 bytes) from sqlite.org's download page | SHA3-256 (what sqlite.org publishes): `92281c86e2957f9b9305890f9ac04204caa7b677cc868dc326d942689df28e30` |

Only what a build needs is kept: of GoogleTest, the `googletest` folder's
`CMakeLists.txt`, `cmake`, `include` and `src` (no Google Mock, samples,
tests or docs); of nlohmann/json, the single header and its licence; of
libvmaf-fast, the five headers that declare its functions and its licence
(not `version.h.in` or `meson.build`, which only its own build uses); of
miniz, the release's single source file, its header and its licence (not
its examples, change log or readme); of Highway, its two licences and the
41 files of `hwy`: its headers and the four sources the build compiles
(`abort.cc`, `per_target.cc`, `print.cc`, `targets.cc`), each identical to the file in `highway-1.2.0.tar.gz` (no contrib, tests,
docs or build files). Of SQLite, the
amalgamation's `sqlite3.c` and `sqlite3.h` (not its shell or extension
header), built with the options SQLite recommends for an app that loads no
extensions (CMakeLists.txt). miniz is built without its file
functions (`MINIZ_NO_STDIO`): the app reads and writes the files itself, so
that a path with any characters works on Windows. In a Release build with
MSVC, what the app uses of it (reading and writing zip files, deflate both
ways) takes about 41 KB.

miniz's files, each identical to the file in `miniz-3.1.2.zip`:

| File | SHA-256 |
| --- | --- |
| `miniz.c` | `e2c1aeb66eef9191d8c3feb164db2def2335a61d039bf04ed849f6b042433b30` |
| `miniz.h` | `b53b62ed122e559b8f679e3cb787a0b0035fe87a58f909da0e44931678f4e85f` |
| `LICENSE` | `0115478d567121238cf6cc1c0c361926cf07a49d9e4c9e66da97fac6a01646b3` |

SQLite's files, each identical to the file in `sqlite-amalgamation-3540100.zip`:

| File | SHA-256 |
| --- | --- |
| `sqlite3.c` | `0c657666d819f261cc4007526ad57e1669de9949df7a8ea847aa284a2bf0ee52` |
| `sqlite3.h` | `d6c84557261893439b341d2e6c35187ae3ea24fbfd843ad94616f72d89242b8c` |

libvmaf-fast's files, each identical to the file at `28776652` (and at `71b7b9be`):

| File | SHA-256 |
| --- | --- |
| `include/libvmaf/feature.h` | `a259aed55f548cd29976b546da215ea8cc98a200df904b76573e434ef8f626b6` |
| `include/libvmaf/libvmaf.h` | `dd9f640a97b03331244b8b26e91b2b2e494dbce4d31c6b483769a10b7ff9eb64` |
| `include/libvmaf/libvmaf_cuda.h` | `411c80b098e64c788f9ea703fb5d5a6eff4575dfaed186e3e47e640fafac60c2` |
| `include/libvmaf/model.h` | `4c25a226e4caefa7eca56c782167597fff1729c8caa1c145183c9479eca94180` |
| `include/libvmaf/picture.h` | `61c47904c8d241ebc1ed60c170b7c0a7715e04b47798c25298f969cb54a95ecc` |
| `LICENSE` | `a6fdb51d02b9dfe198e55534bb8047fe6d756eda117f4ff83de98f92c1952710` |

When the app ships a newer libvmaf-fast, copy the same files from that
release's commit and update this list. A local build of the fork after the
release (scripts/build_libvmaf_fast_local.ps1, while the app is developed)
has the same headers as long as the fork has not changed them since:
`git diff 28776652 HEAD -- libvmaf/include` in the fork shows whether it
has.

nlohmann/json, miniz, Highway and SQLite are compiled into the app, so they belong in the
app's `docs/THIRD_PARTY.md` once the C++ app ships. GoogleTest is only in the
tests. libvmaf itself already ships with the app and is listed there.
