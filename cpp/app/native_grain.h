// A VVC video's film grain in Video Compare's native playback, drawn on the
// GPU by the Python app's film grain library (d3d11_film_grain.dll,
// native/d3d11_film_grain.cpp), as the Python app draws it. FFmpeg's VVC
// decoder draws H.274 grain on 8-bit 4:2:0 video only; native playback has
// it give each frame's grain message instead (film_grain.h turns it into the
// library's parameters), uploads the frame's planes, has the library draw
// the grain on them and pack them into the NV12 or P010 texture shown. On
// macOS it is not drawn yet (native_grain_mac.cpp): FFmpeg draws 8-bit
// video's.
#pragma once

#include <QString>

#include <memory>
#include <optional>
#include <utility>

struct GrainParams;
struct ID3D11Device;
struct ID3D11Texture2D;

namespace vml::app {

class NativeGrain {
public:
    // Why film grain cannot be drawn here (the library absent, or one too old
    // to take the C++ app's frames), or nothing when it can. Loads the
    // library, and readies its shaders, the first time.
    static std::optional<QString> unavailable();

#ifdef _WIN32
    // A drawer on `device`. Throws std::runtime_error when it cannot be made.
    explicit NativeGrain(ID3D11Device* device);
    ~NativeGrain();
    NativeGrain(const NativeGrain&) = delete;
    NativeGrain& operator=(const NativeGrain&) = delete;

    // Draws the grain of `params` on the frame's `count` planes (Y, and Cb
    // and Cr where `count` is 3: R8 textures, or R16 with the codes in their
    // low bits; the chroma planes subsampled by `chroma_shift`, as powers of
    // two), in place, and copies them into `target`, an NV12 texture for
    // 8-bit video or a P010 one above, made to be drawn to: chroma
    // subsampled less than 4:2:0 averaged down to it, and neutral chroma for
    // a frame without (4:0:0). With the device's lock held. Why it failed,
    // or nothing.
    std::optional<QString> draw(ID3D11Texture2D* const planes[3], int count, const GrainParams& params,
                                std::pair<int, int> chroma_shift, ID3D11Texture2D* target);

private:
    void* drawer_ = nullptr;
#endif
};

}  // namespace vml::app
