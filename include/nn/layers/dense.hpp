#pragma once

#include "../activation.hpp"
#include "../layer.hpp"

namespace nn {

enum class Init : std::uint8_t {
    Auto = 0,       ///< pick from activation (He for ReLU family, LeCun for SELU, Glorot otherwise)
    GlorotUniform,
    GlorotNormal,
    HeUniform,
    HeNormal,
    LeCunNormal,
    Uniform01,      ///< legacy behaviour of the original implementation
    Zeros,
};

/// Fully connected layer of neurons: y = f(x W + b).
/// W is stored [in x out] so the forward pass is a single NN-GEMM.
class Dense final : public Layer {
public:
    Dense() = default;
    Dense(std::size_t units, Activation act = ActivationKind::Linear, Init init = Init::Auto,
          bool use_bias = true)
        : m_units(units), m_act(act), m_init(init), m_use_bias(use_bias) {}

    LayerType type() const override { return LayerType::Dense; }
    std::string name() const override;
    void build(std::size_t input_dim, Rng& rng) override;
    void forward(const Matrix& in, Matrix& out, bool training) override;
    void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                  Matrix* grad_in) override;
    void parameters(std::vector<ParamRef>& p) override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;
    bool can_fuse_output(int loss) const override;

    const Activation& activation() const { return m_act; }
    Matrix& weights() { return m_w; }
    Matrix& bias() { return m_b; }

private:
    std::size_t m_units = 0;
    Activation m_act;
    Init m_init = Init::Auto;
    bool m_use_bias = true;
    Matrix m_w, m_b, m_dw, m_db;
    Matrix m_z, m_dz;  // cached pre-activation / scratch
};

/// Initialise a [fan_in x fan_out] weight matrix.
void init_weights(Matrix& w, std::size_t fan_in, std::size_t fan_out, Init init, ActivationKind act,
                  Rng& rng);

}  // namespace nn
