# Third-party components

The project license applies to original project code, not to dependencies.
This inventory is a release-review aid, **not** a complete license manifest or
legal clearance for the generated Windows bundle.

| Component | Role | Upstream information |
| --- | --- | --- |
| Python | Runtime, bundled for Windows releases | [Python license](https://docs.python.org/3/license.html) |
| PySide6 / Qt | Desktop UI and native integration | [Qt licensing](https://doc.qt.io/qt-6/licensing.html) |
| NumPy | Packed scores and numerical operations | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| psutil | Process control | [psutil license](https://github.com/giampaolo/psutil/blob/master/LICENSE) |
| GStreamer and plugin dependencies | Native decoding, audio and presentation | [GStreamer licensing](https://gstreamer.freedesktop.org/documentation/frequently-asked-questions/licensing.html) |
| FFmpeg and libvmaf | External metric/probing tools, not bundled by default; FFmpeg's libvmaf calculates every VMAF metric except VMAF and VMAF NEG on an NVIDIA GPU | [FFmpeg legal information](https://ffmpeg.org/legal.html), [VMAF license](https://github.com/Netflix/vmaf/blob/master/LICENSE) |
| Netflix VMAF v1 model files | Optional built-in model data passed to external libvmaf | [VMAF source models](https://github.com/Netflix/vmaf/tree/master/model/vmaf_v1.0.16), [BSD-2-Clause-Patent license](https://github.com/Netflix/vmaf/blob/master/LICENSE) |
| libjxl SSIMULACRA2 and Butteraugli tools (v0.12.0) | Bundled CPU perceptual metrics | [libjxl releases](https://github.com/libjxl/libjxl/releases), [libjxl licenses](https://github.com/libjxl/libjxl/tree/v0.12.0/LICENSE), [SSIMULACRA2 license](https://github.com/cloudinary/ssimulacra2/blob/main/LICENSE), [Butteraugli license](https://github.com/google/butteraugli/blob/master/LICENSE) |
| Vship `libvship.dll`, three builds | GPU SSIMULACRA2, Butteraugli and CVVDP: `vulkan/` (any GPU with a Vulkan driver; built from commit [97d0dc5](https://codeberg.org/Line-fr/Vship/commit/97d0dc55b273f8f370496d0e206d94413f44e62f), reporting 5.1.2, by `scripts/build_vship_vulkan.ps1` with MinGW-w64 g++ and Khronos Vulkan-Headers 3c65a01, unmodified; reproducible, SHA-256 `3b41119adbaecaac7d3ab228ca591a22ace36866235f421e009808c2137e3f27`), `nvidia/` (CUDA) and `amd/` (HIP), both the v5.1.1 release; only the libraries are bundled | [Vship source, releases and MIT license](https://codeberg.org/Line-fr/Vship), shipped notices in `vmaf_app/tools/vship/licenses/` |
| libvmaf `libvmaf.dll` with CUDA | VMAF and VMAF NEG on NVIDIA GPUs (`vmaf_app/tools/libvmaf/`): libvmaf master [cea2b4d8](https://github.com/Netflix/vmaf/commit/cea2b4d832a105116a3f16f56d6f5d953421952c) with 12 open pull requests merged (the MSVC build and CUDA fixes, each pinned in `scripts/build_libvmaf_cuda.ps1`), built by that script with MSVC and CUDA 13.4, C runtime linked in; reproducible, SHA-256 `28c762d0fe93dc83599b756ec3536ce41e3fbf7782c30c9256a7cff7fb5a6d88`. It contains pthreads4w (pthread-win32, GerHobbelt's fork, via libvmaf's submodule) and nv-codec-headers' CUDA loader; the NVIDIA driver is loaded at run time, and no CUDA runtime is shipped | [libvmaf, BSD-2-Clause-Patent](https://github.com/Netflix/vmaf/blob/master/LICENSE), [pthreads4w](https://github.com/GerHobbelt/pthread-win32) (Apache-2.0 from version 3, except four autoconf files not used here; its source headers still carry the old LGPL text), [nv-codec-headers, MIT](https://github.com/FFmpeg/nv-codec-headers); shipped notices in `vmaf_app/tools/libvmaf/licenses/` |
| NVIDIA frame decoder `nvdec_frames.dll` (this project's, `native/nvdec_frames.cpp`) | Decodes the GPU metrics' videos on NVIDIA GPUs (`vmaf_app/native/`), built by `scripts/build_nvdec_frames.ps1`; reproducible. It contains nv-codec-headers' NVDEC and CUDA definitions (commit eddcea9, as libvmaf's) and loads the NVIDIA driver's `nvcuda.dll` and `nvcuvid.dll` at run time; nothing NVIDIA's is shipped | [nv-codec-headers, MIT](https://github.com/FFmpeg/nv-codec-headers); shipped notice in `vmaf_app/native/licenses/` |
| MinGW-w64 winpthreads and the GCC runtime, linked into the Vulkan `libvship.dll` and `nvdec_frames.dll` | Threads for the self-built Vship Vulkan library and the NVIDIA frame decoder | [winpthreads license](https://github.com/mingw-w64/mingw-w64/blob/master/mingw-w64-libraries/winpthreads/COPYING) (shipped as `vmaf_app/tools/vship/licenses/LICENSE.winpthreads.txt`); libgcc and libstdc++ under the [GCC Runtime Library Exception](https://www.gnu.org/licenses/gcc-exception-3.1.html) |

The spec includes GStreamer GPL and restricted plugin packages. Inspect the
actual libraries in those packages; a package name or top-level LGPL label
does not describe every linked codec dependency. Qt's LGPL distribution
requirements also need attention when producing a frozen application.

The Vship command-line executable and FFMS2 DLL are intentionally not bundled;
the app feeds FFmpeg-decoded frames to Vship's library API (the 5.1 C API)
instead. The Vulkan build needs only the GPU driver's `vulkan-1.dll`. When no
build can use a GPU -- no driver or runtime, or initialization fails -- GPU
scoring is off for that run: SSIMULACRA2 and Butteraugli use the bundled libjxl
CPU tools, and CVVDP, which has no CPU implementation, is not calculated.

For each release, record exact versions/build configurations, retain license
and copyright notices, identify applicable source availability and library
replacement requirements, and provide the required materials with the release.
Do not assume that keeping FFmpeg external resolves licensing for bundled
GStreamer plugins. Seek qualified advice if the distribution obligations are
unclear. No project license choice waives third-party obligations.

Also review screenshots and fixture media for permission to redistribute.
Do not replace third-party copyright notices with the project's 4KVCD notice.
