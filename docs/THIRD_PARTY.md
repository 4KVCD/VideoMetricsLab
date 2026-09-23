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
| FFmpeg and libvmaf | External metric/probing tools, not bundled by default | [FFmpeg legal information](https://ffmpeg.org/legal.html), [VMAF license](https://github.com/Netflix/vmaf/blob/master/LICENSE) |
| Netflix VMAF v1 model files | Optional built-in model data passed to external libvmaf | [VMAF source models](https://github.com/Netflix/vmaf/tree/master/model/vmaf_v1.0.16), [BSD-2-Clause-Patent license](https://github.com/Netflix/vmaf/blob/master/LICENSE) |
| libjxl SSIMULACRA2 and Butteraugli tools (v0.12.0) | Bundled CPU perceptual metrics | [libjxl releases](https://github.com/libjxl/libjxl/releases), [libjxl licenses](https://github.com/libjxl/libjxl/tree/v0.12.0/LICENSE), [SSIMULACRA2 license](https://github.com/cloudinary/ssimulacra2/blob/main/LICENSE), [Butteraugli license](https://github.com/google/butteraugli/blob/master/LICENSE) |
| Vship 4.0.2 `libvship.dll` | Optional CUDA/HIP GPU backend for SSIMULACRA2 and Butteraugli; only the library is bundled | [Vship source and MIT license](https://github.com/Line-fr/Vship), shipped notices in `vmaf_app/tools/vship/licenses/` |

The spec includes GStreamer GPL and restricted plugin packages. Inspect the
actual libraries in those packages; a package name or top-level LGPL label
does not describe every linked codec dependency. Qt's LGPL distribution
requirements also need attention when producing a frozen application.

The Vship command-line executable and FFMS2 DLL are intentionally not bundled;
the app feeds FFmpeg-decoded frames to Vship's library API instead. NVIDIA
CUDA-driver or AMD HIP-runtime initialization failures disable GPU scoring for
that run and select the bundled libjxl CPU tools.

For each release, record exact versions/build configurations, retain license
and copyright notices, identify applicable source availability and library
replacement requirements, and provide the required materials with the release.
Do not assume that keeping FFmpeg external resolves licensing for bundled
GStreamer plugins. Seek qualified advice if the distribution obligations are
unclear. No project license choice waives third-party obligations.

Also review screenshots and fixture media for permission to redistribute.
Do not replace third-party copyright notices with the project's 4KVCD notice.
