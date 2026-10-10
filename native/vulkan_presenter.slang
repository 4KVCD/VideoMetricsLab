// The Vulkan presenter's shaders (native/vulkan_presenter.cpp), compiled to
// SPIR-V by scripts/build_vulkan_presenter_shader.py.
//
// They draw a frame exactly as the Direct3D 11 presenter's shader does
// (native/d3d11_tonemap.cpp), with the same arithmetic in the same order:
// the vertex shader covers the target rectangle, and the fragment shader
// samples the frame's two planes, turns YUV into RGB, and shades HDR video
// for the display.

struct DrawConstants {
    // The part of the frame drawn: left, top, width, height in texture coordinates.
    float4 source;
    // Rows of (Y, U, V) coefficients, the offset in w.
    float4 to_rgb[3];
    // HDR: the source primaries' luminance weights; w: SDR white (nits).
    float4 luma;
    // HDR: linear source RGB to linear BT.709 (shade 1) or BT.2020 (shade 2), by rows.
    float4 to_output[3];
    // 0: the colours as they come; 1: HDR mapped to SDR BT.709; 2: HDR kept, as PQ in BT.2020.
    float shade;
    // HDR: 1 for PQ, 2 for HLG.
    float kind;
    // HDR: the light mapped to SDR white, and HLG's display peak (nits).
    float peak;
};

[[vk::push_constant]] ConstantBuffer<DrawConstants> draw;
[[vk::binding(0)]] Sampler2D<float> luma_plane;
[[vk::binding(1)]] Sampler2D<float2> chroma_plane;

struct Vertex {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Four vertices, drawn as a triangle strip: the corners of the target
// rectangle, which the viewport places in the window.
[shader("vertex")]
Vertex vertex_main(uint id: SV_VertexID) {
    float2 corner = float2(id & 1, id >> 1);
    Vertex v;
    // Vulkan's clip space points y down, so the top corners are at y = -1.
    v.position = float4(corner.x * 2 - 1, corner.y * 2 - 1, 0, 1);
    v.uv = draw.source.xy + corner * draw.source.zw;
    return v;
}

// PQ code values to light (nits).
float3 pq(float3 x) {
    float3 p = pow(max(x, 0), 1.0 / 78.84375);
    return 10000 * pow(max(p - 0.8359375, 0) / max(18.8515625 - 18.6875 * p, 1e-6), 1.0 / 0.1593017578125);
}

// Light (nits) to PQ code values.
float3 pq_code(float3 nits) {
    float3 y = pow(saturate(nits / 10000), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * y) / (1 + 18.6875 * y), 78.84375);
}

// HLG code values to scene light (0 to 1).
float3 hlg_scene(float3 x) {
    float3 lo = x * x / 3;
    float3 hi = (exp((x - 0.55991073) / 0.17883277) + 0.28466892) / 12;
    return float3(x.r <= 0.5 ? lo.r : hi.r, x.g <= 0.5 ? lo.g : hi.g, x.b <= 0.5 ? lo.b : hi.b);
}

// HLG code values to display light (nits), at the display peak.
float3 hlg(float3 x) {
    float3 scene = hlg_scene(x);
    return scene * pow(max(dot(scene, draw.luma.rgb), 1e-8), .2) * draw.peak;
}

// Linear source RGB to linear BT.709 or BT.2020.
float3 converted(float3 x) {
    return float3(dot(x, draw.to_output[0].rgb), dot(x, draw.to_output[1].rgb), dot(x, draw.to_output[2].rgb));
}

// Linear light (0 to 1) to sRGB code values.
float3 srgb(float3 x) {
    float3 lo = 12.92 * x, hi = 1.055 * pow(max(x, 0), 1.0 / 2.4) - .055;
    return float3(x.r <= .0031308 ? lo.r : hi.r, x.g <= .0031308 ? lo.g : hi.g, x.b <= .0031308 ? lo.b : hi.b);
}

[shader("fragment")]
float4 fragment_main(Vertex v) : SV_Target {
    float3 c = float3(luma_plane.Sample(v.uv), chroma_plane.Sample(v.uv));
    float3 rgb = saturate(float3(dot(c, draw.to_rgb[0].xyz), dot(c, draw.to_rgb[1].xyz), dot(c, draw.to_rgb[2].xyz))
                          + float3(draw.to_rgb[0].w, draw.to_rgb[1].w, draw.to_rgb[2].w));
    if (draw.shade < 0.5)
        return float4(rgb, 1);
    float3 light = draw.kind < 1.5 ? pq(rgb) : hlg(rgb);
    if (draw.shade > 1.5) {
        // HDR kept, for an HDR display, which takes PQ in BT.2020: the light
        // converted to BT.2020 and coded as PQ (HLG's at its display peak).
        return float4(pq_code(converted(light)), 1);
    }
    // A fixed extended-Reinhard luminance curve, the same for every video compared.
    float white = draw.luma.w;
    float y = max(dot(light, draw.luma.rgb) / white, 0);
    float w = draw.peak / white;
    float mapped = y * (1 + y / (w * w)) / (1 + y);
    light = light / white * (y > 1e-8 ? mapped / y : 0);
    // Linear source primaries to BT.709, then gamut clipping and sRGB coding.
    return float4(srgb(saturate(converted(light))), 1);
}
