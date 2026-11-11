#pragma once

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "matrix.hpp"
#include "random.hpp"

namespace nn {

/// Stable identifiers used in the binary model format. Never renumber.
enum class LayerType : std::uint32_t {
    Dense = 1,
    Dropout = 2,
    RBF = 3,
    SimpleRNN = 4,
    LSTM = 5,
    GRU = 6,
    Conv2D = 7,
    MaxPool2D = 8,
    LayerNorm = 9,
};

/// A trainable parameter buffer and its gradient (same size).
struct ParamRef {
    real* value;
    real* grad;
    std::size_t size;
    bool decay;  ///< apply weight decay (false for biases / norms)
};

/// Base class for all layers.
///
/// Data flows as matrices of shape [batch x features]. Structured layers
/// (recurrent, convolutional) interpret each row as a flattened tensor
/// (time-major sequences, CHW images), so every layer composes with every
/// other one without reshape layers.
class Layer {
public:
    virtual ~Layer() = default;

    virtual LayerType type() const = 0;
    virtual std::string name() const = 0;

    /// Allocate parameters for the given input width. Called by Network::add.
    virtual void build(std::size_t input_dim, Rng& rng) = 0;
    std::size_t input_dim() const { return m_in; }
    std::size_t output_dim() const { return m_out; }

    /// out = f(in). The layer may cache what it needs for backward() when
    /// `training` is true. `out` is resized by the layer.
    virtual void forward(const Matrix& in, Matrix& out, bool training) = 0;

    /// Given dL/d(out), compute parameter gradients (overwriting them) and,
    /// if grad_in != nullptr, dL/d(in). `in`/`out` are the tensors of the
    /// matching forward call.
    virtual void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                          Matrix* grad_in) = 0;

    /// Collect trainable parameters.
    virtual void parameters(std::vector<ParamRef>&) {}
    std::size_t parameter_count();

    /// Serialise configuration + weights (binary, little endian).
    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;

    /// Reseed internal RNG (dropout masks etc.).
    virtual void seed(std::uint64_t) {}

    /// Output layer whose activation is fused with the loss: backward receives
    /// dL/dz directly (softmax + cross entropy, sigmoid + binary CE).
    virtual bool can_fuse_output(int /*loss*/) const { return false; }
    void set_fused_output(bool f) { m_fused_output = f; }

protected:
    std::size_t m_in = 0, m_out = 0;
    bool m_fused_output = false;
};

using LayerPtr = std::unique_ptr<Layer>;

/// Create an empty layer of the given type (for deserialisation).
LayerPtr make_layer(LayerType t);

}  // namespace nn
