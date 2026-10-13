# Third-party code in the C++ port

Vendored, so that a build needs no download. Each was downloaded once, with
Brian's OK, and checked against the SHA-256 its project publishes.

| Library | Version | Used for | Licence | Source | SHA-256 of the download |
| --- | --- | --- | --- | --- | --- |
| [GoogleTest](https://github.com/google/googletest) | 1.18.0 | The tests (not shipped) | BSD 3-Clause (`googletest/LICENSE`) | `googletest-1.18.0.tar.gz` from the v1.18.0 release | `6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5` |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.12.0 | Reading and writing JSON | MIT (`nlohmann/LICENSE.MIT`) | `json.hpp` from the v3.12.0 release | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` |

Only what a build needs is kept: of GoogleTest, the `googletest` folder's
`CMakeLists.txt`, `cmake`, `include` and `src` (no Google Mock, samples,
tests or docs); of nlohmann/json, the single header and its licence.

nlohmann/json is compiled into the app, so it belongs in the app's
`docs/THIRD_PARTY.md` once the C++ app ships. GoogleTest is only in the tests.
