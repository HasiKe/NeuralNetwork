#pragma once

#include <string>

#include "config.hpp"

namespace nn {

/// Neuron transfer functions. Every value is a different "neuron type" that a
/// Dense / Conv2D / SimpleRNN layer can use.
enum class ActivationKind : std::uint8_t {
    Linear = 0,     ///< identity (regression outputs, linear neurons)
    Sigmoid,        ///< logistic 1/(1+e^-x)
    Tanh,           ///< hyperbolic tangent
    ReLU,           ///< max(0, x)
    LeakyReLU,      ///< x if x>0 else alpha*x            (alpha default 0.01)
    ELU,            ///< x if x>0 else alpha*(e^x-1)      (alpha default 1)
    SELU,           ///< scaled ELU (self-normalising)
    GELU,           ///< Gaussian error linear unit (tanh approximation)
    Swish,          ///< x*sigmoid(beta*x), a.k.a. SiLU   (alpha = beta, default 1)
    Mish,           ///< x*tanh(softplus(x))
    Softplus,       ///< log(1+e^x)
    Softsign,       ///< x/(1+|x|)
    HardSigmoid,    ///< clamp(x/6 + 1/2, 0, 1)
    HardTanh,       ///< clamp(x, -1, 1)
    HardSwish,      ///< x*hardsigmoid(x)
    ReLU6,          ///< clamp(x, 0, 6)
    Gaussian,       ///< e^(-x^2)  (radial neuron)
    Sine,           ///< sin(x)    (periodic / SIREN neuron)
    Step,           ///< Heaviside threshold (classic perceptron, straight-through gradient)
    Softmax,        ///< normalised exponential over the layer (probability outputs)
    Count_
};

struct Activation {
    ActivationKind kind = ActivationKind::Linear;
    /// Shape parameter: alpha for LeakyReLU/ELU, beta for Swish. NaN = default.
    real alpha = -1.0f;

    Activation() = default;
    Activation(ActivationKind k) : kind(k), alpha(default_alpha(k)) {}  // NOLINT implicit
    Activation(ActivationKind k, real a) : kind(k), alpha(a) {}

    static real default_alpha(ActivationKind k);

    // Convenience constructors: nn::Activation::relu() etc.
    static Activation linear() { return ActivationKind::Linear; }
    static Activation sigmoid() { return ActivationKind::Sigmoid; }
    static Activation tanh() { return ActivationKind::Tanh; }
    static Activation relu() { return ActivationKind::ReLU; }
    static Activation leaky_relu(real a = 0.01f) { return {ActivationKind::LeakyReLU, a}; }
    static Activation elu(real a = 1.0f) { return {ActivationKind::ELU, a}; }
    static Activation selu() { return ActivationKind::SELU; }
    static Activation gelu() { return ActivationKind::GELU; }
    static Activation swish(real beta = 1.0f) { return {ActivationKind::Swish, beta}; }
    static Activation mish() { return ActivationKind::Mish; }
    static Activation softplus() { return ActivationKind::Softplus; }
    static Activation softsign() { return ActivationKind::Softsign; }
    static Activation hard_sigmoid() { return ActivationKind::HardSigmoid; }
    static Activation hard_tanh() { return ActivationKind::HardTanh; }
    static Activation hard_swish() { return ActivationKind::HardSwish; }
    static Activation relu6() { return ActivationKind::ReLU6; }
    static Activation gaussian() { return ActivationKind::Gaussian; }
    static Activation sine() { return ActivationKind::Sine; }
    static Activation step() { return ActivationKind::Step; }
    static Activation softmax() { return ActivationKind::Softmax; }
};

const char* to_string(ActivationKind k);
ActivationKind activation_from_string(const std::string& name);

/// Apply activation to a rows x cols block: y = f(z). `z` and `y` may alias.
void activation_forward(const Activation& a, std::size_t rows, std::size_t cols, const real* z,
                        real* y);

/// Back-propagate through the activation: dz = dy * f'(z).
/// Needs both pre-activation z and output y (whichever is cheaper is used).
/// `dz` may alias `dy`.
void activation_backward(const Activation& a, std::size_t rows, std::size_t cols, const real* z,
                         const real* y, const real* dy, real* dz);

/// Whether backward only needs y (then z need not be stored).
bool activation_backward_needs_z(ActivationKind k);

}  // namespace nn
