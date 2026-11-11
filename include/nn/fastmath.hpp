#pragma once

#include <cstdint>
#include <cstring>

#include "config.hpp"

/// Branch-free single precision approximations (Cephes based, ~1-2 ulp in
/// the relevant range). They are written so that GCC/Clang can auto-vectorise
/// loops calling them without -ffast-math, which std::exp/std::tanh prevent.
namespace nn::fastmath {

NN_INLINE float bits_to_float(std::int32_t i) {
    float f;
    std::memcpy(&f, &i, sizeof f);
    return f;
}
NN_INLINE std::int32_t float_to_bits(float f) {
    std::int32_t i;
    std::memcpy(&i, &f, sizeof i);
    return i;
}

/// e^x. Inputs are clamped to [-86, 88] so neither the result nor any
/// if-converted intermediate is subnormal (subnormals cost ~100x on x86).
NN_INLINE float exp(float x) {
    x = x > 88.0f ? 88.0f : x;
    x = x < -86.0f ? -86.0f : x;
    float fx = x * 1.44269504088896341f + 0.5f;
    // floor() via truncation + correction (vectorises with plain SSE2)
    float t = static_cast<float>(static_cast<std::int32_t>(fx));
    fx = t > fx ? t - 1.0f : t;
    x -= fx * 0.693359375f;
    x -= fx * -2.12194440e-4f;
    const float z = x * x;
    float y = 1.9875691500E-4f;
    y = y * x + 1.3981999507E-3f;
    y = y * x + 8.3334519073E-3f;
    y = y * x + 4.1665795894E-2f;
    y = y * x + 1.6666665459E-1f;
    y = y * x + 5.0000001201E-1f;
    y = y * z + x + 1.0f;
    const std::int32_t n = static_cast<std::int32_t>(fx);
    return y * bits_to_float((n + 127) << 23);
}

/// Natural log for x > 0 (returns a large negative number for x <= 0).
NN_INLINE float log(float x) {
    x = x < 1.17549435e-38f ? 1.17549435e-38f : x;
    std::int32_t i = float_to_bits(x);
    std::int32_t e = ((i >> 23) & 0xff) - 126;
    float m = bits_to_float((i & 0x007fffff) | 0x3f000000);  // [0.5, 1)
    const bool small = m < 0.707106781186547524f;
    e = small ? e - 1 : e;
    m = small ? m + m - 1.0f : m - 1.0f;
    const float z = m * m;
    float y = 7.0376836292E-2f;
    y = y * m - 1.1514610310E-1f;
    y = y * m + 1.1676998740E-1f;
    y = y * m - 1.2420140846E-1f;
    y = y * m + 1.4249322787E-1f;
    y = y * m - 1.6668057665E-1f;
    y = y * m + 2.0000714765E-1f;
    y = y * m - 2.4999993993E-1f;
    y = y * m + 3.3333331174E-1f;
    y = y * m * z;
    const float fe = static_cast<float>(e);
    y += fe * -2.12194440e-4f;
    y += -0.5f * z;
    return m + y + fe * 0.693359375f;
}

NN_INLINE float sigmoid(float x) {
    x = x < -80.0f ? -80.0f : (x > 80.0f ? 80.0f : x);  // keeps 1/(1+e^-x) normal
    return 1.0f / (1.0f + exp(-x));
}

NN_INLINE float tanh(float x) {
    const float ax = x < 0 ? -x : x;
    // small |x|: odd polynomial (avoids cancellation)
    const float z = x * x;
    float p = -5.70498872745E-3f;
    p = p * z + 2.06390887954E-2f;
    p = p * z - 5.37397155531E-2f;
    p = p * z + 1.33314422036E-1f;
    p = p * z - 3.33332819422E-1f;
    const float small = p * z * x + x;
    // large |x|: 1 - 2/(e^{2|x|}+1)
    const float e = exp(2.0f * (ax > 9.0f ? 9.0f : ax));
    float big = 1.0f - 2.0f / (e + 1.0f);
    big = x < 0 ? -big : big;
    return ax < 0.625f ? small : big;
}

/// log(1 + exp(x)), numerically stable
NN_INLINE float softplus(float x) {
    const float ax = x < 0 ? -x : x;
    const float m = x > 0 ? x : 0.0f;
    return m + log(1.0f + exp(-ax));
}

}  // namespace nn::fastmath
