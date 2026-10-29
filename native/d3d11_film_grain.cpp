// Film grain synthesis for Video Compare's VVC frames, on the GPU.
//
// A VVC encode can describe its film grain in H.274 SEI messages instead of
// coding it; the player is meant to synthesize the grain again on each
// decoded frame. FFmpeg's VVC decoder gives each frame's grain message
// instead of drawing it, so this module draws it: on a decoded frame's planes
// in place, on the app's D3D11 device, right after the frame is uploaded and
// before it is cropped, scaled or shown.
//
// The synthesis is H.274's frequency filtering model, as SMPTE RDD 5 defines
// it, and it reproduces FFmpeg's implementation (libavcodec/h274.c) exactly,
// so 8-bit video gets the same pixels as FFmpeg gives, with two exceptions:
//   - FFmpeg keeps the grain in 8 bits, so an extremely strong grain, beyond
//     +-127 codes, wraps around there; here it does not.
//   - Where a plane's width or height is not a multiple of 8 (1080p video's
//     540 chroma rows, say), its last column or row of 8x8 blocks is cut off
//     by the edge. FFmpeg, and VTM too, still average whole blocks there,
//     reading whatever their frame buffer holds past the picture, which is
//     not part of the video; here the samples past the edge repeat the edge.
//     The grain of those blocks can differ.
// FFmpeg does not synthesize grain above 8 bits; there this follows JVET's
// reference software (VTM, SEIFilmGrainSynthesizer.cpp): block averages are
// taken back to 8-bit codes, and the grain is shifted up to the video's bit
// depth.
//
// How the grain of one plane is made:
//   1. The plane is cut into 16x16 blocks. Each block gets a random position
//      in a grain pattern and a random sign, from a pseudo-random sequence
//      seeded by the frame's picture order count. Each block's numbers
//      depend on the block before it, so they are worked out on the CPU
//      (compute_block_offsets) and uploaded.
//   2. The "classify" shader works out, for each 8x8 block, which of the
//      message's intensity intervals its average brightness falls in (if
//      any). The interval says how strong the block's grain is and which of
//      13 x 13 grain patterns it uses (built once, build_pattern_database).
//   3. The "draw" shader works out each sample's grain from its block's
//      interval, pattern and position; smooths each block's left edge with
//      the block before it (RDD 5's deblocking); adds the grain to the
//      picture and clips the result. It writes to a texture of this module's,
//      which is then copied over the plane: a decoded frame's textures
//      cannot be written by a compute shader directly.
//
// The app uploads a decoded frame's planes, one texture per plane, R8 or R16
// holding the codes in their low bits; draws the grain on them
// (vmaf_grain_apply_planes); and copies them into the NV12 or P010 texture it
// shows (vmaf_grain_pack).
//
// The caller holds the device's lock during every call, because this draws on
// the device's immediate context, as the decoders and the app's presenter do.
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#include "h274/grain_params.h"
#include "h274/h274_tables.h"

// The constant buffer both shaders read, describing one plane.
struct alignas(16) ShaderConstants {
    uint32_t width, height;    // the plane's picture size
    uint32_t depth;            // bit depth
    uint32_t shift;            // log2_scale_factor + 6: the grain's scaling shift
    uint32_t blocks16_across;  // 16x16 blocks per row
    uint32_t interval_count;
    uint32_t blocks8_across;   // 8x8 blocks per row
    float sample_to_code;      // converts a sampled value to a code: 255 for R8, 65535 for R16
    // Per interval: x = lower | upper << 8, y = scale, z = h | v << 8.
    uint32_t interval[256][4];
};

// The constant buffer, as the shaders declare it.
#define SHADER_CONSTANTS R"(
cbuffer Plane : register(b0) {
    uint width; uint height; uint depth; uint shift;
    uint blocks16_across; uint interval_count; uint blocks8_across; float sample_to_code;
    uint4 interval[256];
};
)"

static const char classify_shader_source[] = SHADER_CONSTANTS R"(
Texture2D<float> picture : register(t0);
RWTexture2D<uint> block_interval : register(u0);  // per 8x8 block: its interval + 1, or 0 for no grain
groupshared uint row_sums[8][8];  // [row][block]

// Eight 8x8 blocks side by side per thread group; each thread adds up one
// row of one block, then one thread per block adds up its rows.
[numthreads(8, 8, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    uint block_x = group.x * 8 + thread.x, block_y = group.y, row = thread.y;
    uint sum = 0;
    for (uint x = 0; x < 8; x++) {
        // Samples past the picture's edge repeat the edge (where FFmpeg
        // reads its frame buffer past the picture: see the top of the file).
        int2 inside = min(int2(block_x * 8 + x, block_y * 8 + row), int2(width - 1, height - 1));
        sum += (uint)(picture.Load(int3(inside, 0)) * sample_to_code + 0.5);
    }
    row_sums[row][thread.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    if (row != 0 || block_x >= blocks8_across)
        return;

    // The block's average as an 8-bit code, and the first interval it falls in.
    sum = 0;
    for (uint y = 0; y < 8; y++)
        sum += row_sums[y][thread.x];
    uint average = sum >> (depth - 2);  // divide by 64 samples, and scale to 8 bits
    uint chosen = 0;
    for (uint i = 0; i < interval_count; i++) {
        uint bounds = interval[i].x;
        if (average >= (bounds & 0xFF) && average <= (bounds >> 8)) {
            chosen = i + 1;
            break;
        }
    }
    block_interval[uint2(block_x, block_y)] = chosen;
}
)";

static const char draw_shader_source[] = SHADER_CONSTANTS R"(
Texture2D<float> picture : register(t0);
Texture2D<int> patterns : register(t1);              // the 13 x 13 grain patterns, 64x64 each
StructuredBuffer<uint> block_offsets : register(t2);  // per 16x16 block: x | y << 8 | sign << 16
Texture2D<uint> block_interval : register(t3);
RWTexture2D<float> output : register(u0);

// How a block's grain is drawn: (has grain, signed scale, where its 8x8
// window starts in the pattern texture, x and y). A block past either end of
// the row has no grain.
int4 block_grain_settings(int block_x, uint block_y) {
    if (block_x < 0 || block_x >= (int)blocks8_across)
        return int4(0, 0, 0, 0);
    uint chosen = block_interval.Load(int3(block_x, block_y, 0));
    if (chosen == 0)
        return int4(0, 0, 0, 0);
    uint4 values = interval[chosen - 1];
    // The 16x16 block's place in the pattern, plus this 8x8 block's place within it.
    uint offset = block_offsets[(block_y >> 1) * blocks16_across + ((uint)block_x >> 1)];
    int scale = asint(values.y);
    if (offset & 0x10000)
        scale = -scale;
    uint cutoffs = values.z;
    int pattern_x = (cutoffs & 0xFF) * 64 + (offset & 0xFF) + 8 * (block_x & 1);
    int pattern_y = (cutoffs >> 8) * 64 + ((offset >> 8) & 0xFF) + 8 * (block_y & 1);
    return int4(1, scale, pattern_x, pattern_y);
}

// The grain of the sample at (x, y) in a block with these settings.
int grain_at(int4 settings, uint x, uint y) {
    int pattern = patterns.Load(int3(settings.z + x, settings.w + y, 0));
    return settings.x != 0 ? (settings.y * pattern) >> shift : 0;
}

groupshared int4 neighbourhood[3];   // the settings of the left block, this block and the right block
groupshared int block_grain[8][8];  // this block's grain, before deblocking
groupshared int edge_grain[8][2];   // per row: the left block's last column, the right block's first

// One thread group per 8x8 block, one thread per sample.
[numthreads(8, 8, 1)]
void main(uint3 block : SV_GroupID, uint3 sample : SV_GroupThreadID, uint thread_index : SV_GroupIndex) {
    uint x = sample.x, y = sample.y;
    if (thread_index < 3)
        neighbourhood[thread_index] = block_grain_settings((int)block.x + (int)thread_index - 1, block.y);
    GroupMemoryBarrierWithGroupSync();

    int own = grain_at(neighbourhood[1], x, y);
    block_grain[y][x] = own;
    // The threads on the block's left and right edges also work out the
    // neighbouring block's grain beside them.
    if (x == 0 || x == 7)
        edge_grain[y][x == 0 ? 0 : 1] = grain_at(neighbourhood[x == 0 ? 0 : 2], x == 0 ? 7 : 0, y);
    GroupMemoryBarrierWithGroupSync();

    uint2 position = block.xy * 8 + sample.xy;
    if (position.x >= width || position.y >= height)
        return;

    // Deblocking: a block that has grain smooths its left edge with the
    // block before it, changing its own first column and the last column of
    // the block before.
    int g = own;
    if (x == 0 && block.x > 0 && neighbourhood[1].x != 0)
        g = (edge_grain[y][0] + 2 * own + block_grain[y][1]) >> 2;
    else if (x == 7 && neighbourhood[2].x != 0)
        g = (edge_grain[y][1] + 2 * own + block_grain[y][6]) >> 2;

    // The grain is made in 8-bit units, so it is scaled up to the video's depth.
    int code = (int)(picture.Load(int3(position, 0)) * sample_to_code + 0.5);
    output[position] = clamp(code + g * (1 << (depth - 8)), 0, (1 << depth) - 1) / sample_to_code;
}
)";

// The shaders are compiled once per process.
static std::mutex compile_lock;
static ID3DBlob* compiled_shaders[2] = {};

static HRESULT compile_shaders(ID3DBlob** classify, ID3DBlob** draw) {
    std::lock_guard<std::mutex> guard(compile_lock);
    const char* sources[2] = {classify_shader_source, draw_shader_source};
    const size_t lengths[2] = {sizeof(classify_shader_source) - 1, sizeof(draw_shader_source) - 1};
    for (int i = 0; i < 2; i++) {
        if (compiled_shaders[i])
            continue;
        ID3DBlob* errors = nullptr;
        HRESULT hr = D3DCompile(sources[i], lengths[i], nullptr, nullptr, nullptr, "main", "cs_5_0",
                                D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &compiled_shaders[i], &errors);
        if (errors)
            errors->Release();
        if (FAILED(hr))
            return hr;
    }
    if (classify)
        *classify = compiled_shaders[0];
    if (draw)
        *draw = compiled_shaders[1];
    return S_OK;
}

// RDD 5's pseudo-random number generator: a 31-bit linear feedback shift
// register (primitive polynomial x^31 + x^3 + 1).
static void advance_random(uint32_t* state) {
    const uint32_t x = *state;
    *state = (x << 1) | ((1u ^ (x >> 2) ^ (x >> 30)) & 1u);
}

// The size of the pattern database texture: 13 x 13 patterns of 64x64 samples.
static const int PATTERN_SIZE = 64;
static const int CUTOFF_COUNT = 13;
static const int DATABASE_SIZE = CUTOFF_COUNT * PATTERN_SIZE;

// Builds RDD 5's 13 x 13 grain patterns, one for each pair of horizontal and
// vertical cutoff frequencies. Each pattern is Gaussian noise in the
// frequencies up to its cutoffs, taken through RDD 5's 64-point inverse
// transform; the horizontal edges of each 8x8 block are then attenuated.
// Pattern (h, v) is stored at x = 64 h, y = 64 v of a single texture.
static std::vector<int8_t> build_pattern_database() {
    // RDD 5's integer transform matrix, worked out from its formula.
    const double pi = 3.14159265358979323846;
    int transform[64][64];
    for (int y = 0; y < 64; y++)
        for (int k = 0; k < 64; k++)
            transform[y][k] = k == 0 ? 32 : (int)std::lround(32 * std::sqrt(2.0) * std::cos(pi * (2 * y + 1) * k / 128));

    std::vector<int8_t> database(DATABASE_SIZE * DATABASE_SIZE, 0);
    int8_t coefficients[64][64];  // [vertical frequency][horizontal frequency]
    int16_t half_transformed[64][64];
    for (int h = 0; h < CUTOFF_COUNT; h++) {
        for (int v = 0; v < CUTOFF_COUNT; v++) {
            const int last_horizontal = ((h + 3) << 2) - 1, last_vertical = ((v + 3) << 2) - 1;
            uint32_t random = h274_seeds[h + v * 13];
            std::memset(coefficients, 0, sizeof coefficients);
            for (int l = 0; l <= last_vertical; l++) {
                for (int k = 0; k <= last_horizontal; k += 4) {
                    const uint32_t start = random % 2048;
                    for (int i = 0; i < 4; i++)
                        coefficients[l][k + i] = h274_gaussian[(start + i) % 2048];
                    advance_random(&random);
                }
            }
            coefficients[0][0] = 0;  // no constant offset
            for (int y = 0; y < 64; y++) {
                for (int x = 0; x <= last_vertical; x++) {
                    int32_t sum = 0;
                    for (int p = 0; p <= last_horizontal; p++)
                        sum += transform[y][p] * coefficients[x][p];
                    half_transformed[y][x] = (int16_t)((sum + 128) >> 8);
                }
            }
            for (int y = 0; y < 64; y++) {
                for (int x = 0; x < 64; x++) {
                    int32_t sum = 0;
                    for (int p = 0; p <= last_vertical; p++)
                        sum += half_transformed[x][p] * transform[y][p];
                    int value = (sum + 128) >> 8;
                    value = value < -127 ? -127 : value > 127 ? 127 : value;
                    if ((y & 7) == 0 || (y & 7) == 7)
                        value = (value * h274_deblock[v]) >> 7;
                    database[(size_t)(v * PATTERN_SIZE + y) * DATABASE_SIZE + h * PATTERN_SIZE + x] = (int8_t)value;
                }
            }
        }
    }
    return database;
}

// The database is the same for every device, so it is built once per process.
static const std::vector<int8_t>& pattern_database() {
    static std::once_flag built;
    static std::vector<int8_t> database;
    std::call_once(built, [] { database = build_pattern_database(); });
    return database;
}

// A texture of this module's, with the views the shaders use.
struct WorkTexture {
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11UnorderedAccessView* uav = nullptr;
    UINT width = 0, height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};

static void release_texture(WorkTexture* t) {
    if (t->uav)
        t->uav->Release();
    if (t->srv)
        t->srv->Release();
    if (t->texture)
        t->texture->Release();
    *t = WorkTexture();
}

// Makes sure `t` is a texture of this format, at least this large. The
// planes of a frame share it: each uses as much of it as it needs.
static HRESULT ensure_texture(ID3D11Device* device, WorkTexture* t, UINT width, UINT height, DXGI_FORMAT format,
                              bool readable_by_shaders) {
    if (t->texture && t->width >= width && t->height >= height && t->format == format)
        return S_OK;
    release_texture(t);
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | (readable_by_shaders ? D3D11_BIND_SHADER_RESOURCE : 0);
    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &t->texture);
    if (SUCCEEDED(hr))
        hr = device->CreateUnorderedAccessView(t->texture, nullptr, &t->uav);
    if (SUCCEEDED(hr) && readable_by_shaders)
        hr = device->CreateShaderResourceView(t->texture, nullptr, &t->srv);
    if (FAILED(hr)) {
        release_texture(t);
        return hr;
    }
    t->width = width;
    t->height = height;
    t->format = format;
    return S_OK;
}

// A drawer for one stream. Its work buffers are shared by the frame's planes,
// which are drawn one after another.
struct GrainDrawer {
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    ID3D11ComputeShader* classify = nullptr;
    ID3D11ComputeShader* draw = nullptr;
    ID3D11Buffer* constants = nullptr;
    ID3D11Texture2D* patterns = nullptr;
    ID3D11ShaderResourceView* patterns_srv = nullptr;
    // Each 16x16 block's offsets, as worked out on the CPU and on the GPU.
    std::vector<uint32_t> offsets_on_cpu;
    ID3D11Buffer* offsets = nullptr;
    ID3D11ShaderResourceView* offsets_srv = nullptr;
    UINT offsets_capacity = 0;
    WorkTexture block_interval;  // each 8x8 block's interval
    WorkTexture output;          // the grained plane, before it is copied back
    // vmaf_grain_pack's shaders and constants, made on its first call.
    ID3D11VertexShader* pack_cover = nullptr;
    ID3D11PixelShader* pack_luma = nullptr;
    ID3D11PixelShader* pack_chroma = nullptr;
    ID3D11Buffer* pack_constants = nullptr;
    std::string error;
};

static void release_offsets(GrainDrawer* d) {
    if (d->offsets_srv)
        d->offsets_srv->Release();
    if (d->offsets)
        d->offsets->Release();
    d->offsets = nullptr;
    d->offsets_srv = nullptr;
    d->offsets_capacity = 0;
}

static void destroy_drawer(GrainDrawer* d) {
    release_offsets(d);
    release_texture(&d->block_interval);
    release_texture(&d->output);
    if (d->pack_constants)
        d->pack_constants->Release();
    if (d->pack_chroma)
        d->pack_chroma->Release();
    if (d->pack_luma)
        d->pack_luma->Release();
    if (d->pack_cover)
        d->pack_cover->Release();
    if (d->patterns_srv)
        d->patterns_srv->Release();
    if (d->patterns)
        d->patterns->Release();
    if (d->constants)
        d->constants->Release();
    if (d->draw)
        d->draw->Release();
    if (d->classify)
        d->classify->Release();
    if (d->context)
        d->context->Release();
    if (d->device)
        d->device->Release();
    delete d;
}

static GrainDrawer* create_drawer(ID3D11Device* device) {
    GrainDrawer* d = new (std::nothrow) GrainDrawer();
    if (!d)
        return nullptr;
    d->device = device;
    device->AddRef();
    device->GetImmediateContext(&d->context);
    ID3DBlob *classify, *draw;
    HRESULT hr = compile_shaders(&classify, &draw);
    if (SUCCEEDED(hr))
        hr = device->CreateComputeShader(classify->GetBufferPointer(), classify->GetBufferSize(), nullptr,
                                         &d->classify);
    if (SUCCEEDED(hr))
        hr = device->CreateComputeShader(draw->GetBufferPointer(), draw->GetBufferSize(), nullptr, &d->draw);
    if (SUCCEEDED(hr)) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = sizeof(ShaderConstants);
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = device->CreateBuffer(&desc, nullptr, &d->constants);
    }
    if (SUCCEEDED(hr)) {
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = desc.Height = DATABASE_SIZE;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8_SINT;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA data = {pattern_database().data(), DATABASE_SIZE, 0};
        hr = device->CreateTexture2D(&desc, &data, &d->patterns);
        if (SUCCEEDED(hr))
            hr = device->CreateShaderResourceView(d->patterns, nullptr, &d->patterns_srv);
    }
    if (FAILED(hr)) {
        destroy_drawer(d);
        return nullptr;
    }
    return d;
}

// Works out each 16x16 block's place in the pattern and its sign, in raster
// order, as x | y << 8 | sign << 16 (FFmpeg's ff_h274_apply_film_grain).
static void compute_block_offsets(std::vector<uint32_t>& offsets, int poc, int component, UINT blocks_across,
                                  UINT blocks_down) {
    static const int color_offset[3] = {0, 85, 170};
    uint32_t random = h274_seeds[(poc + color_offset[component]) & 0xFF];
    offsets.resize((size_t)blocks_across * blocks_down);
    for (uint32_t& offset : offsets) {
        const uint32_t x = ((random >> 16) % 52) & 0xFFFC;
        const uint32_t y = ((random & 0xFFFF) % 56) & 0xFFF8;
        offset = x | y << 8 | (random & 1) << 16;
        advance_random(&random);
    }
}

static HRESULT upload_block_offsets(GrainDrawer* d) {
    const UINT count = (UINT)d->offsets_on_cpu.size();
    if (count > d->offsets_capacity) {
        release_offsets(d);
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = count * sizeof(uint32_t);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(uint32_t);
        HRESULT hr = d->device->CreateBuffer(&desc, nullptr, &d->offsets);
        if (FAILED(hr))
            return hr;
        hr = d->device->CreateShaderResourceView(d->offsets, nullptr, &d->offsets_srv);
        if (FAILED(hr))
            return hr;
        d->offsets_capacity = count;
    }
    D3D11_BOX box = {0, 0, 0, count * (UINT)sizeof(uint32_t), 1, 1};
    d->context->UpdateSubresource(d->offsets, 0, &box, d->offsets_on_cpu.data(), 0, 0);
    return S_OK;
}

// Leaves the compute stage as the presenter expects to find it: nothing bound.
static void unbind_compute_stage(ID3D11DeviceContext* context) {
    ID3D11ShaderResourceView* no_views[4] = {};
    ID3D11UnorderedAccessView* no_targets[1] = {};
    ID3D11Buffer* no_buffers[1] = {};
    context->CSSetShaderResources(0, 4, no_views);
    context->CSSetUnorderedAccessViews(0, 1, no_targets, nullptr);
    context->CSSetConstantBuffers(0, 1, no_buffers);
    context->CSSetShader(nullptr, nullptr, 0);
}

static HRESULT write_constants(GrainDrawer* d, const GrainParams* params, const GrainPlane& plane, UINT blocks16_across,
                               UINT blocks8_across, DXGI_FORMAT format) {
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = d->context->Map(d->constants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr))
        return hr;
    ShaderConstants* k = (ShaderConstants*)mapped.pData;
    std::memset(k, 0, sizeof *k);
    k->width = (uint32_t)plane.width;
    k->height = (uint32_t)plane.height;
    k->depth = (uint32_t)params->depth;
    k->shift = (uint32_t)params->log2_scale_factor + 6;
    k->blocks16_across = blocks16_across;
    k->interval_count = (uint32_t)plane.intervals;
    k->blocks8_across = blocks8_across;
    k->sample_to_code = format == DXGI_FORMAT_R16_UNORM ? 65535.0f : 255.0f;
    for (int i = 0; i < plane.intervals; i++) {
        k->interval[i][0] = plane.lower[i] | (uint32_t)plane.upper[i] << 8;
        k->interval[i][1] = (uint32_t)(int32_t)plane.scale[i];
        k->interval[i][2] = plane.h[i] | (uint32_t)plane.v[i] << 8;
    }
    d->context->Unmap(d->constants, 0);
    return S_OK;
}

// Draws the grain of plane `c` on `texture` (its slice `subresource`), in place.
static HRESULT draw_plane(GrainDrawer* d, int c, ID3D11Texture2D* texture, UINT subresource, const GrainParams* params) {
    const GrainPlane& plane = params->plane[c];
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    if (desc.Format != DXGI_FORMAT_R8_UNORM && desc.Format != DXGI_FORMAT_R16_UNORM) {
        d->error = "a plane of DXGI format " + std::to_string((int)desc.Format);
        return E_INVALIDARG;
    }
    if (desc.ArraySize != 1 || plane.width <= 0 || plane.height <= 0 || (UINT)plane.width > desc.Width ||
        (UINT)plane.height > desc.Height || plane.intervals > 256) {
        d->error = "a plane of an unexpected size";
        return E_INVALIDARG;
    }
    const UINT width = (UINT)plane.width, height = (UINT)plane.height;
    const UINT blocks8_across = (width + 7) / 8, blocks8_down = (height + 7) / 8;
    const UINT blocks16_across = (width + 15) / 16, blocks16_down = (height + 15) / 16;

    compute_block_offsets(d->offsets_on_cpu, params->poc, c, blocks16_across, blocks16_down);
    HRESULT hr = upload_block_offsets(d);
    if (SUCCEEDED(hr))
        hr = ensure_texture(d->device, &d->block_interval, blocks8_across, blocks8_down, DXGI_FORMAT_R16_UINT, true);
    if (SUCCEEDED(hr))
        hr = ensure_texture(d->device, &d->output, width, height, desc.Format, false);
    if (SUCCEEDED(hr))
        hr = write_constants(d, params, plane, blocks16_across, blocks8_across, desc.Format);
    ID3D11ShaderResourceView* picture = nullptr;
    if (SUCCEEDED(hr))
        hr = d->device->CreateShaderResourceView(texture, nullptr, &picture);
    if (FAILED(hr)) {
        d->error = "could not set up the grain's GPU resources";
        return hr;
    }

    d->context->CSSetConstantBuffers(0, 1, &d->constants);
    d->context->CSSetShaderResources(0, 1, &picture);
    d->context->CSSetUnorderedAccessViews(0, 1, &d->block_interval.uav, nullptr);
    d->context->CSSetShader(d->classify, nullptr, 0);
    d->context->Dispatch((blocks8_across + 7) / 8, blocks8_down, 1);  // eight blocks per group

    ID3D11ShaderResourceView* draw_inputs[4] = {picture, d->patterns_srv, d->offsets_srv, d->block_interval.srv};
    d->context->CSSetUnorderedAccessViews(0, 1, &d->output.uav, nullptr);  // unbinds the block intervals as an output
    d->context->CSSetShaderResources(0, 4, draw_inputs);
    d->context->CSSetShader(d->draw, nullptr, 0);
    d->context->Dispatch(blocks8_across, blocks8_down, 1);

    unbind_compute_stage(d->context);
    picture->Release();
    D3D11_BOX box = {0, 0, 0, width, height, 1};
    d->context->CopySubresourceRegion(texture, subresource, 0, 0, 0, d->output.texture, 0, &box);
    return S_OK;
}

static int draw_planes(GrainDrawer* d, ID3D11Texture2D* const* textures, const UINT* subresources, int count,
                       const GrainParams* params) {
    if (params->depth < 8 || params->depth > 12 || params->log2_scale_factor < 0 || params->log2_scale_factor > 15) {
        d->error = "film grain parameters out of range";
        return -2;
    }
    for (int c = 0; c < count; c++) {
        if (params->plane[c].intervals == 0)
            continue;  // no grain on this plane
        if (FAILED(draw_plane(d, c, textures[c], subresources[c], params)))
            return -3;
    }
    return 0;
}

// Builds the pattern database and compiles the shaders ahead of the first
// drawer, so that opening a comparison does not wait for them. Returns 0 on
// success.
extern "C" __declspec(dllexport) int vmaf_grain_prepare() {
    pattern_database();
    return SUCCEEDED(compile_shaders(nullptr, nullptr)) ? 0 : -1;
}

// Creates a grain drawer on `device`, the app's D3D11 device. Returns null on failure.
extern "C" __declspec(dllexport) void* vmaf_grain_create(ID3D11Device* device) {
    return device ? create_drawer(device) : nullptr;
}

// Describes the last error, for the app's log.
extern "C" __declspec(dllexport) const char* vmaf_grain_error(void* drawer) {
    return drawer ? ((GrainDrawer*)drawer)->error.c_str() : "";
}

// Frees a grain drawer.
extern "C" __declspec(dllexport) void vmaf_grain_destroy(void* drawer) {
    if (drawer)
        destroy_drawer((GrainDrawer*)drawer);
}

// The app's frames come decoded in system memory (FFmpeg's VVC decoder) and
// are shown from NV12 or P010 textures. It uploads a frame's planes into
// textures of its own, one per plane (R8, or R16 with the codes in their low
// bits), draws the grain on them with vmaf_grain_apply_planes, and copies
// them into the texture it shows with vmaf_grain_pack. The caller holds the
// device's lock during both.

// Draws the grain of `params` on `planes` (each one's slice `subresources[c]`),
// in place. Returns 0 on success, or a negative number on failure (the frame
// may then be partly grained), with vmaf_grain_error saying why.
extern "C" __declspec(dllexport) int vmaf_grain_apply_planes(void* drawer, ID3D11Texture2D* const* planes,
                                                             const UINT* subresources, int count,
                                                             const GrainParams* params) {
    GrainDrawer* d = (GrainDrawer*)drawer;
    if (!d || !planes || !subresources || !params || count < 1 || count > 3)
        return -1;
    d->error.clear();
    return draw_planes(d, planes, subresources, count, params);
}

// The shaders that copy a frame's planes into an NV12 or P010 texture: a
// triangle covering the target, and a pixel shader for each of its planes.
// The planes hold their codes in their low bits, P010 in its high bits:
// `scale` moves them up (64 for 10-bit codes, 16 for 12-bit ones; 1 into
// NV12). Each of the target's chroma samples covers 2x2 luma samples: the
// planes' chroma samples in that square, 1, 2 or 4 of them (4:2:0, 4:2:2,
// 4:4:4: `shift` is their subsampling as powers of two), are averaged, the
// last column and row (`last`) repeated past an odd edge. A frame without
// chroma planes (4:0:0) gets `neutral` chroma.
static const char pack_shader_source[] = R"(
Texture2D<float> plane_y : register(t0);
Texture2D<float> plane_u : register(t1);
Texture2D<float> plane_v : register(t2);
cbuffer Pack : register(b0) {
    float scale;
    float neutral;
    int2 shift;
    int2 last;
    int has_chroma;
    int unused;
};

float4 cover(uint id : SV_VertexID) : SV_Position {
    float2 corner = float2((id << 1) & 2, id & 2);
    return float4(corner * float2(2, -2) + float2(-1, 1), 0, 1);
}
float luma(float4 at : SV_Position) : SV_Target {
    return plane_y.Load(int3(at.xy, 0)) * scale;
}
float2 chroma(float4 at : SV_Position) : SV_Target {
    if (has_chroma == 0)
        return float2(neutral, neutral);
    int2 step = int2(2, 2) >> shift;
    int2 first = int2(at.xy) * step;
    float2 sum = float2(0, 0);
    for (int y = 0; y < step.y; y++) {
        for (int x = 0; x < step.x; x++) {
            int3 place = int3(min(first + int2(x, y), last), 0);
            sum += float2(plane_u.Load(place), plane_v.Load(place));
        }
    }
    return sum / float(step.x * step.y) * scale;
}
)";

static HRESULT prepare_pack(GrainDrawer* d) {
    if (d->pack_chroma)
        return S_OK;
    static ID3DBlob* compiled[3] = {};
    const char* entries[3] = {"cover", "luma", "chroma"};
    const char* targets[3] = {"vs_5_0", "ps_5_0", "ps_5_0"};
    {
        std::lock_guard<std::mutex> guard(compile_lock);
        for (int i = 0; i < 3; i++) {
            if (compiled[i])
                continue;
            ID3DBlob* errors = nullptr;
            HRESULT hr = D3DCompile(pack_shader_source, sizeof(pack_shader_source) - 1, nullptr, nullptr, nullptr,
                                    entries[i], targets[i], D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &compiled[i], &errors);
            if (errors)
                errors->Release();
            if (FAILED(hr))
                return hr;
        }
    }
    HRESULT hr = d->device->CreateVertexShader(compiled[0]->GetBufferPointer(), compiled[0]->GetBufferSize(), nullptr,
                                               &d->pack_cover);
    if (SUCCEEDED(hr))
        hr = d->device->CreatePixelShader(compiled[1]->GetBufferPointer(), compiled[1]->GetBufferSize(), nullptr,
                                          &d->pack_luma);
    if (SUCCEEDED(hr))
        hr = d->device->CreatePixelShader(compiled[2]->GetBufferPointer(), compiled[2]->GetBufferSize(), nullptr,
                                          &d->pack_chroma);
    if (SUCCEEDED(hr)) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = 32;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        hr = d->device->CreateBuffer(&desc, nullptr, &d->pack_constants);
    }
    return hr;
}

// Copies a frame's `count` planes (Y, and Cb and Cr where `count` is 3, as
// vmaf_grain_apply_planes takes them: `depth`-bit codes, in R8 textures or
// in the low bits of R16 ones) into the two planes of `target`, an NV12
// texture for 8-bit codes or a P010 one above, made with
// D3D11_BIND_RENDER_TARGET, whose size is the luma plane's rounded up to
// even. Chroma subsampled less than 4:2:0 (`shift_x` and `shift_y`: the
// chroma planes' subsampling as powers of two, 1 or 0) is averaged down to
// it; a frame without chroma planes (`count` 1) gets neutral chroma.
// Returns 0 on success, or a negative number with vmaf_grain_error saying
// why.
extern "C" __declspec(dllexport) int vmaf_grain_pack(void* drawer, ID3D11Texture2D* const* planes, int count,
                                                     int depth, int shift_x, int shift_y, ID3D11Texture2D* target) {
    GrainDrawer* d = (GrainDrawer*)drawer;
    if (!d || !planes || !target || (count != 1 && count != 3) || depth < 8 || depth > 16 || shift_x < 0 ||
        shift_x > 1 || shift_y < 0 || shift_y > 1)
        return -1;
    d->error.clear();
    D3D11_TEXTURE2D_DESC desc;
    target->GetDesc(&desc);
    const bool ten_bit = desc.Format == DXGI_FORMAT_P010;
    if (ten_bit != (depth > 8) || (!ten_bit && desc.Format != DXGI_FORMAT_NV12) ||
        !(desc.BindFlags & D3D11_BIND_RENDER_TARGET)) {
        d->error = "a target of DXGI format " + std::to_string((int)desc.Format) + " that cannot take " +
                   std::to_string(depth) + "-bit codes";
        return -2;
    }
    // Where the chroma planes end, for the averages at an odd edge.
    int last[2] = {0, 0};
    if (count == 3) {
        D3D11_TEXTURE2D_DESC chroma;
        planes[1]->GetDesc(&chroma);
        last[0] = (int)chroma.Width - 1;
        last[1] = (int)chroma.Height - 1;
    }
    ID3D11RenderTargetView* views[2] = {};
    ID3D11ShaderResourceView* sources[3] = {};
    HRESULT hr = prepare_pack(d);
    D3D11_RENDER_TARGET_VIEW_DESC view = {};
    view.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    if (SUCCEEDED(hr)) {
        view.Format = ten_bit ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
        hr = d->device->CreateRenderTargetView(target, &view, &views[0]);
    }
    if (SUCCEEDED(hr)) {
        view.Format = ten_bit ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R8G8_UNORM;
        hr = d->device->CreateRenderTargetView(target, &view, &views[1]);
    }
    for (int c = 0; c < count && SUCCEEDED(hr); c++)
        hr = d->device->CreateShaderResourceView(planes[c], nullptr, &sources[c]);
    if (SUCCEEDED(hr)) {
        // The codes moved up into P010's high bits, and the middle code
        // (neutral chroma) as the target holds it.
        const float scale = depth > 8 ? (float)(1 << (16 - depth)) : 1.0f;
        const float neutral = (float)(1 << (depth - 1)) / (depth > 8 ? 65535.0f : 255.0f) * scale;
        struct {
            float scale, neutral;
            int shift[2], last[2], has_chroma, unused;
        } constants = {scale, neutral, {shift_x, shift_y}, {last[0], last[1]}, count == 3 ? 1 : 0, 0};
        ID3D11DeviceContext* c = d->context;
        c->UpdateSubresource(d->pack_constants, 0, nullptr, &constants, 0, 0);
        c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        c->IASetInputLayout(nullptr);
        c->VSSetShader(d->pack_cover, nullptr, 0);
        c->PSSetConstantBuffers(0, 1, &d->pack_constants);
        c->PSSetShaderResources(0, 3, sources);
        D3D11_VIEWPORT viewport = {0, 0, (float)desc.Width, (float)desc.Height, 0, 1};
        c->RSSetViewports(1, &viewport);
        c->OMSetRenderTargets(1, &views[0], nullptr);
        c->PSSetShader(d->pack_luma, nullptr, 0);
        c->Draw(3, 0);
        viewport.Width = (float)(desc.Width / 2);
        viewport.Height = (float)(desc.Height / 2);
        c->RSSetViewports(1, &viewport);
        c->OMSetRenderTargets(1, &views[1], nullptr);
        c->PSSetShader(d->pack_chroma, nullptr, 0);
        c->Draw(3, 0);
        // Nothing left bound, as the presenter expects to find the context.
        ID3D11ShaderResourceView* no_sources[3] = {};
        ID3D11Buffer* no_buffers[1] = {};
        c->OMSetRenderTargets(0, nullptr, nullptr);
        c->PSSetShaderResources(0, 3, no_sources);
        c->PSSetConstantBuffers(0, 1, no_buffers);
        c->PSSetShader(nullptr, nullptr, 0);
        c->VSSetShader(nullptr, nullptr, 0);
    } else {
        d->error = "could not set up the planes' copy into the frame";
    }
    for (ID3D11ShaderResourceView* source : sources)
        if (source)
            source->Release();
    for (ID3D11RenderTargetView* target_view : views)
        if (target_view)
            target_view->Release();
    return SUCCEEDED(hr) ? 0 : -3;
}

// Draws the grain of `params` on planes held in memory (8-bit samples, or
// 16-bit ones above 8 bits), on a device of its own: the GPU, or Windows'
// software renderer if `software` is set or there is no GPU. For the tests
// and the checks against FFmpeg; playback uses vmaf_grain_apply_planes.
extern "C" __declspec(dllexport) int vmaf_grain_planes(const void* const* in, void* const* out, int count,
                                                       const GrainParams* params, int software) {
    if (count < 1 || count > 3 || !params)
        return -1;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    HRESULT hr = E_FAIL;
    if (!software)
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, levels, 2, D3D11_SDK_VERSION, &device,
                               nullptr, &context);
    if (FAILED(hr))
        hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, levels, 2, D3D11_SDK_VERSION, &device,
                               nullptr, &context);
    if (FAILED(hr))
        return -4;
    const DXGI_FORMAT format = params->depth > 8 ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
    const UINT bytes_per_sample = params->depth > 8 ? 2 : 1;
    ID3D11Texture2D* textures[3] = {};
    UINT subresources[3] = {};
    int result = 0;
    for (int c = 0; c < count && result == 0; c++) {
        const GrainPlane& plane = params->plane[c];
        if (plane.width <= 0 || plane.height <= 0) {
            result = -1;
            break;
        }
        // Made as the app makes its planes: for sampling and rendering.
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = (UINT)plane.width;
        desc.Height = (UINT)plane.height;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        D3D11_SUBRESOURCE_DATA data = {in[c], desc.Width * bytes_per_sample, 0};
        if (FAILED(device->CreateTexture2D(&desc, &data, &textures[c])))
            result = -4;
    }
    GrainDrawer* d = result == 0 ? create_drawer(device) : nullptr;
    if (result == 0 && !d)
        result = -4;
    if (result == 0)
        result = draw_planes(d, textures, subresources, count, params);
    for (int c = 0; c < count && result == 0; c++) {
        const GrainPlane& plane = params->plane[c];
        D3D11_TEXTURE2D_DESC desc;
        textures[c]->GetDesc(&desc);
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* staging = nullptr;
        if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) {
            result = -4;
            break;
        }
        context->CopyResource(staging, textures[c]);
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped))) {
            const size_t row_bytes = (size_t)plane.width * bytes_per_sample;
            for (int y = 0; y < plane.height; y++)
                std::memcpy((uint8_t*)out[c] + y * row_bytes, (const uint8_t*)mapped.pData + (size_t)y * mapped.RowPitch,
                            row_bytes);
            context->Unmap(staging, 0);
        } else {
            result = -4;
        }
        staging->Release();
    }
    if (d)
        destroy_drawer(d);
    for (ID3D11Texture2D* t : textures)
        if (t)
            t->Release();
    context->Release();
    device->Release();
    return result;
}
