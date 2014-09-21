#include <metal_stdlib>
#include <metal_graphics>
#include <metal_matrix>
#include <metal_geometric>
#include <metal_math>
#include <metal_texture>
#include <metal_integer>

using namespace metal;

#include "MyShaderTypes.hpp"

struct VertexOut {
    float4 position [[position]];
    float2 texcoord [[user(texturecoord)]];
};

vertex VertexOut myVertexShader(const device MyShaderTypes::QuardVertex *vertices [[ buffer(0) ]],
                                unsigned int vid [[vertex_id]])
{
    VertexOut out;
    out.position = float4(vertices[vid].position, 0.0f, 1.0f);
    out.texcoord = vertices[vid].texcoord;
    
    return out;
}

fragment float4 myFragmentShader(VertexOut interpolated [[stage_in]],
                                 texture2d<float> texture [[texture(0)]])
{
    constexpr sampler basic_sampler(coord::normalized, filter::linear, address::clamp_to_edge);
    return texture.sample(basic_sampler, interpolated.texcoord);
}

inline float calc_next_w(float Lux, float Rux, float Tuy, float Buy, float centerW)
{
    float v_in = Lux - Rux;
    float h_in = Tuy - Buy;
    float w = centerW + v_in + h_in;
    return mix(w, 1.0f, 0.01f);
}

// ゲームプログラミングのためのリアルタイム衝突判定 より
// 点cと線分abの間の距離の平方を返す
inline float SqDistPointSegment(float2 a, float2 b, float2 c)
{
    float2 ab = b - a, ac = c - a, bc = c - b;
    float e = dot(ac, ab);
    // cがabの外側に射影される場合を扱う
    if (e <= 0.0f)
        return dot(ac, ac);
    
    float f = dot(ab, ab);
    if (e >= f)
        return dot(bc, bc);
    
    // cがab上に射影される場合を扱う
    return dot(ac, ac) - e * e / f;
}

kernel void step_fuild(texture2d<float, access::sample>  input [[ texture(0) ]],
                       texture2d<float, access::write> output [[ texture(1) ]],
                       uint2 gid                              [[ thread_position_in_grid ]],
                       const device MyShaderTypes::FuildConstant *fuildConstant [[ buffer(0) ]],
                       texture2d<float, access::sample> inputImage [[ texture(2) ]],
                       texture2d<float, access::write> outputImage [[ texture(3) ]])
{
    float2 gidf = static_cast<float2>(gid);
    
    constexpr sampler fluid_sampler(coord::pixel, filter::nearest, address::clamp_to_edge);
    
    float3 c = input.sample(fluid_sampler, gidf).xyz;
    float3 w = input.sample(fluid_sampler, gidf, int2(0, -1)).xyz;
    float3 a = input.sample(fluid_sampler, gidf, int2(-1, 0)).xyz;
    float3 s = input.sample(fluid_sampler, gidf, int2(0, 1)).xyz;
    float3 d = input.sample(fluid_sampler, gidf, int2(1, 0)).xyz;
    
    float3 aw = input.sample(fluid_sampler, gidf, int2(- 1, - 1)).xyz;
    float3 as = input.sample(fluid_sampler, gidf, int2(- 1, + 1)).xyz;
    float3 sd = input.sample(fluid_sampler, gidf, int2(+ 1, + 1)).xyz;
    float3 wd = input.sample(fluid_sampler, gidf, int2(+ 1, - 1)).xyz;
    
    // 移流
    float nextW = calc_next_w(a.x, d.x, w.y, s.y, c.z);
    
    float2 nextU = c.xy;
    
    // 圧力
    float dxw = (a.z - nextW) + (nextW - d.z);
    float dyw = (w.z - nextW) + (nextW - s.z);
    nextU += float2(dxw, dyw) * 1.2f;
    
    // 外力
    float dist = sqrt(SqDistPointSegment(fuildConstant->a, fuildConstant->b, gidf));
    nextU += fuildConstant->force * (1.0f - smoothstep(5.0f, 25.0f, dist));
    
    // 減衰
    nextU = nextU - nextU * 0.03f;

    // まとめる
    float3 nextC = float3(nextU.x, nextU.y, nextW);
    
    // 粘性
    constexpr float one = 0.0625f;
    constexpr float v1 = one;
    constexpr float v2 = one * 2.0f;
    constexpr float v4 = one * 4.0f;
    
    float3 blur = nextC * v4 /**/ + w * v2 + a * v2 + s * v2  + d * v2 /**/ + aw * v1 + as * v1 + sd * v1 + wd * v1;
    
    // write
    output.write(float4(blur.x, blur.y, blur.z, 1.0f), gid);
    
    
    // image
    float2 velocity = -blur.xy;
    float2 uv = gidf + velocity;
    float2 origin = float2(floor(uv.x), floor(uv.y));
    float2 mod = float2(fract(uv.x), fract(uv.y));
    float4 c0 = inputImage.sample(fluid_sampler, origin);
    float4 c1 = inputImage.sample(fluid_sampler, float2(origin.x + 1.0f, origin.y));
    float4 c2 = inputImage.sample(fluid_sampler, float2(origin.x, origin.y + 1.0f));
    float4 c3 = inputImage.sample(fluid_sampler, float2(origin.x + 1.0f, origin.y + 1.0f));
    
    float4 color = mix(mix(c0, c1, mod.x), mix(c2, c3, mod.x), mod.y);
    outputImage.write(color, gid);
    
//    float2 velocity = -blur.xy;
//    float2 uv = gidf + velocity;
//    constexpr sampler image_sampler(coord::normalized, filter::linear, address::repeat);
//    float4 color = inputImage.sample(image_sampler, uv / FUILD_SIZE);
//    outputImage.write(color, gid);
}