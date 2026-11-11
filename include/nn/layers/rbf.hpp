#pragma once

#include "../layer.hpp"

namespace nn {

/// Radial basis function neurons: y_j = exp(-gamma_j * ||x - c_j||^2).
/// Centres and widths are trainable.
class RBF final : public Layer {
public:
    RBF() = default;
    explicit RBF(std::size_t units, real initial_gamma = 1.0f)
        : m_units(units), m_gamma0(initial_gamma) {}

    LayerType type() const override { return LayerType::RBF; }
    std::string name() const override;
    void build(std::size_t input_dim, Rng& rng) override;
    void forward(const Matrix& in, Matrix& out, bool training) override;
    void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                  Matrix* grad_in) override;
    void parameters(std::vector<ParamRef>& p) override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    Matrix& centers() { return m_c; }  ///< [units x in]

private:
    std::size_t m_units = 0;
    real m_gamma0 = 1.0f;
    Matrix m_c, m_gamma, m_dc, m_dgamma;
    Matrix m_d2, m_g, m_xn, m_tmp;  // scratch
};

}  // namespace nn
