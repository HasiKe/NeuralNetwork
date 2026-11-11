#include "nn/activation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "nn/fastmath.hpp"

namespace nn {

namespace fm = fastmath;

namespace {
constexpr real kSeluLambda = 1.0507009873554805f;
constexpr real kSeluAlpha = 1.6732632423543772f;
constexpr real kGeluC = 0.7978845608028654f;  // sqrt(2/pi)
constexpr real kGeluK = 0.044715f;

const char* const kNames[] = {"linear",   "sigmoid",     "tanh",      "relu",       "leaky_relu",
                              "elu",      "selu",        "gelu",      "swish",      "mish",
                              "softplus", "softsign",    "hard_sigmoid", "hard_tanh", "hard_swish",
                              "relu6",    "gaussian",    "sine",      "step",       "softmax"};
static_assert(sizeof(kNames) / sizeof(kNames[0]) == static_cast<std::size_t>(ActivationKind::Count_),
              "activation name table out of sync");

void softmax_rows(std::size_t rows, std::size_t cols, const real* z, real* y) {
    for (std::size_t r = 0; r < rows; ++r) {
        const real* NN_RESTRICT zr = z + r * cols;
        real* yr = y + r * cols;
        real mx = zr[0];
        for (std::size_t c = 1; c < cols; ++c) mx = zr[c] > mx ? zr[c] : mx;
        real sum = 0;
        NN_SIMD_REDUCE(+, sum)
        for (std::size_t c = 0; c < cols; ++c) {
            const real e = fm::exp(zr[c] - mx);
            yr[c] = e;
            sum += e;
        }
        const real inv = 1.0f / sum;
        NN_SIMD
        for (std::size_t c = 0; c < cols; ++c) yr[c] *= inv;
    }
}
}  // namespace

real Activation::default_alpha(ActivationKind k) {
    switch (k) {
        case ActivationKind::LeakyReLU: return 0.01f;
        case ActivationKind::ELU: return 1.0f;
        case ActivationKind::Swish: return 1.0f;
        default: return 0.0f;
    }
}

const char* to_string(ActivationKind k) {
    auto i = static_cast<std::size_t>(k);
    return i < static_cast<std::size_t>(ActivationKind::Count_) ? kNames[i] : "unknown";
}

ActivationKind activation_from_string(const std::string& name) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(ActivationKind::Count_); ++i)
        if (name == kNames[i]) return static_cast<ActivationKind>(i);
    if (name == "silu") return ActivationKind::Swish;
    if (name == "identity") return ActivationKind::Linear;
    throw std::invalid_argument("nn: unknown activation '" + name + "'");
}

bool activation_backward_needs_z(ActivationKind k) {
    switch (k) {
        case ActivationKind::Linear:
        case ActivationKind::Sigmoid:
        case ActivationKind::Tanh:
        case ActivationKind::ReLU:
        case ActivationKind::ELU:
        case ActivationKind::SELU:
        case ActivationKind::HardSigmoid:
        case ActivationKind::HardTanh:
        case ActivationKind::ReLU6:
        case ActivationKind::Softmax:
            return false;
        default:
            return true;
    }
}

namespace {
void forward_serial(const Activation& act, std::size_t rows, std::size_t cols, const real* z, real* y) {
    const std::size_t n = rows * cols;
    const real a = act.alpha;
    // Each case is a tight, branch-free loop the compiler vectorises.
    switch (act.kind) {
        case ActivationKind::Linear:
            if (y != z) std::copy(z, z + n, y);
            break;
        case ActivationKind::Sigmoid:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) y[i] = fm::sigmoid(z[i]);
            break;
        case ActivationKind::Tanh:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) y[i] = fm::tanh(z[i]);
            break;
        case ActivationKind::ReLU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) y[i] = z[i] > 0 ? z[i] : 0.0f;
            break;
        case ActivationKind::LeakyReLU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) y[i] = z[i] > 0 ? z[i] : a * z[i];
            break;
        case ActivationKind::ELU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real x = z[i];
                const real e = a * (fm::exp(x < 0 ? x : 0.0f) - 1.0f);
                y[i] = x > 0 ? x : e;
            }
            break;
        case ActivationKind::SELU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real x = z[i];
                const real e = kSeluAlpha * (fm::exp(x < 0 ? x : 0.0f) - 1.0f);
                y[i] = kSeluLambda * (x > 0 ? x : e);
            }
            break;
        case ActivationKind::GELU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real x = z[i];
                y[i] = 0.5f * x * (1.0f + fm::tanh(kGeluC * (x + kGeluK * x * x * x)));
            }
            break;
        case ActivationKind::Swish:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) y[i] = z[i] * fm::sigmoid(a * z[i]);
            break;
        case ActivationKind::Mish:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) y[i] = z[i] * fm::tanh(fm::softplus(z[i]));
            break;
        case ActivationKind::Softplus:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) y[i] = fm::softplus(z[i]);
            break;
        case ActivationKind::Softsign:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real x = z[i];
                y[i] = x / (1.0f + (x < 0 ? -x : x));
            }
            break;
        case ActivationKind::HardSigmoid:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real v = z[i] * (1.0f / 6.0f) + 0.5f;
                y[i] = v < 0 ? 0.0f : (v > 1 ? 1.0f : v);
            }
            break;
        case ActivationKind::HardTanh:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real v = z[i];
                y[i] = v < -1 ? -1.0f : (v > 1 ? 1.0f : v);
            }
            break;
        case ActivationKind::HardSwish:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real x = z[i];
                real r = x + 3.0f;
                r = r < 0 ? 0.0f : (r > 6 ? 6.0f : r);
                y[i] = x * r * (1.0f / 6.0f);
            }
            break;
        case ActivationKind::ReLU6:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real v = z[i];
                y[i] = v < 0 ? 0.0f : (v > 6 ? 6.0f : v);
            }
            break;
        case ActivationKind::Gaussian:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) y[i] = fm::exp(-z[i] * z[i]);
            break;
        case ActivationKind::Sine:
            for (std::size_t i = 0; i < n; ++i) y[i] = std::sin(z[i]);
            break;
        case ActivationKind::Step:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) y[i] = z[i] > 0 ? 1.0f : 0.0f;
            break;
        case ActivationKind::Softmax:
            softmax_rows(rows, cols, z, y);
            break;
        default:
            throw std::invalid_argument("nn: invalid activation");
    }
}

void backward_serial(const Activation& act, std::size_t rows, std::size_t cols, const real* z,
                     const real* y, const real* dy, real* dz) {
    const std::size_t n = rows * cols;
    const real a = act.alpha;
    switch (act.kind) {
        case ActivationKind::Linear:
            if (dz != dy) std::copy(dy, dy + n, dz);
            break;
        case ActivationKind::Sigmoid:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) dz[i] = dy[i] * y[i] * (1.0f - y[i]);
            break;
        case ActivationKind::Tanh:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) dz[i] = dy[i] * (1.0f - y[i] * y[i]);
            break;
        case ActivationKind::ReLU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) dz[i] = y[i] > 0 ? dy[i] : 0.0f;
            break;
        case ActivationKind::LeakyReLU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) dz[i] = z[i] > 0 ? dy[i] : a * dy[i];
            break;
        case ActivationKind::ELU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) dz[i] = y[i] > 0 ? dy[i] : dy[i] * (y[i] + a);
            break;
        case ActivationKind::SELU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i)
                dz[i] = y[i] > 0 ? dy[i] * kSeluLambda : dy[i] * (y[i] + kSeluLambda * kSeluAlpha);
            break;
        case ActivationKind::GELU:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real x = z[i];
                const real t = fm::tanh(kGeluC * (x + kGeluK * x * x * x));
                const real du = kGeluC * (1.0f + 3.0f * kGeluK * x * x);
                dz[i] = dy[i] * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du);
            }
            break;
        case ActivationKind::Swish:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real s = fm::sigmoid(a * z[i]);
                dz[i] = dy[i] * (s + a * z[i] * s * (1.0f - s));
            }
            break;
        case ActivationKind::Mish:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real x = z[i];
                const real t = fm::tanh(fm::softplus(x));
                dz[i] = dy[i] * (t + x * (1.0f - t * t) * fm::sigmoid(x));
            }
            break;
        case ActivationKind::Softplus:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) dz[i] = dy[i] * fm::sigmoid(z[i]);
            break;
        case ActivationKind::Softsign:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real d = 1.0f + (z[i] < 0 ? -z[i] : z[i]);
                dz[i] = dy[i] / (d * d);
            }
            break;
        case ActivationKind::HardSigmoid:
            NN_SIMD for (std::size_t i = 0; i < n; ++i)
                dz[i] = (y[i] > 0 && y[i] < 1) ? dy[i] * (1.0f / 6.0f) : 0.0f;
            break;
        case ActivationKind::HardTanh:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) dz[i] = (y[i] > -1 && y[i] < 1) ? dy[i] : 0.0f;
            break;
        case ActivationKind::HardSwish:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real x = z[i];
                const real g = x < -3 ? 0.0f : (x > 3 ? 1.0f : (2.0f * x + 3.0f) * (1.0f / 6.0f));
                dz[i] = dy[i] * g;
            }
            break;
        case ActivationKind::ReLU6:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) dz[i] = (y[i] > 0 && y[i] < 6) ? dy[i] : 0.0f;
            break;
        case ActivationKind::Gaussian:
            NN_SIMD for (std::size_t i = 0; i < n; ++i) dz[i] = dy[i] * -2.0f * z[i] * y[i];
            break;
        case ActivationKind::Sine:
            for (std::size_t i = 0; i < n; ++i) dz[i] = dy[i] * std::cos(z[i]);
            break;
        case ActivationKind::Step:
            // straight-through estimator, clipped to |z| <= 1
            NN_SIMD for (std::size_t i = 0; i < n; ++i)
                dz[i] = (z[i] >= -1 && z[i] <= 1) ? dy[i] : 0.0f;
            break;
        case ActivationKind::Softmax:
            for (std::size_t r = 0; r < rows; ++r) {
                const real* yr = y + r * cols;
                const real* dyr = dy + r * cols;
                real* dzr = dz + r * cols;
                real s = 0;
                NN_SIMD_REDUCE(+, s)
                for (std::size_t c = 0; c < cols; ++c) s += yr[c] * dyr[c];
                NN_SIMD for (std::size_t c = 0; c < cols; ++c) dzr[c] = yr[c] * (dyr[c] - s);
            }
            break;
        default:
            throw std::invalid_argument("nn: invalid activation");
    }
}


// Split large blocks into row chunks processed by different threads.
[[maybe_unused]] constexpr std::size_t kParallelElems = std::size_t(1) << 16;

template <class F>
void for_row_chunks(std::size_t rows, std::size_t cols, F&& f) {
#ifdef _OPENMP
    if (rows > 1 && rows * cols >= kParallelElems) {
        const std::size_t chunk = std::max<std::size_t>(1, (kParallelElems / 4) / std::max<std::size_t>(cols, 1));
        const std::size_t nchunks = (rows + chunk - 1) / chunk;
#pragma omp parallel for schedule(static)
        for (std::size_t c = 0; c < nchunks; ++c) {
            const std::size_t r0 = c * chunk;
            f(r0, std::min(chunk, rows - r0));
        }
        return;
    }
#endif
    (void)cols;
    f(std::size_t(0), rows);
}
}  // namespace

void activation_forward(const Activation& act, std::size_t rows, std::size_t cols, const real* z,
                        real* y) {
    for_row_chunks(rows, cols, [&](std::size_t r0, std::size_t n) {
        forward_serial(act, n, cols, z + r0 * cols, y + r0 * cols);
    });
}

void activation_backward(const Activation& act, std::size_t rows, std::size_t cols, const real* z,
                         const real* y, const real* dy, real* dz) {
    for_row_chunks(rows, cols, [&](std::size_t r0, std::size_t n) {
        const std::size_t o = r0 * cols;
        backward_serial(act, n, cols, z ? z + o : nullptr, y ? y + o : nullptr, dy + o, dz + o);
    });
}

}  // namespace nn
