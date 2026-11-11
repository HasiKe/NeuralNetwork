#pragma once

#include "../layer.hpp"

namespace nn {

/// Layer normalisation over the features of each sample: y = g * (x - mu) / sigma + b.
class LayerNorm final : public Layer {
public:
    LayerNorm() = default;
    explicit LayerNorm(real eps) : m_eps(eps) {}

    LayerType type() const override { return LayerType::LayerNorm; }
    std::string name() const override;
    void build(std::size_t input_dim, Rng& rng) override;
    void forward(const Matrix& in, Matrix& out, bool training) override;
    void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                  Matrix* grad_in) override;
    void parameters(std::vector<ParamRef>& p) override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

private:
    real m_eps = 1e-5f;
    Matrix m_g, m_b, m_dg, m_db;
    Matrix m_xhat, m_rstd;
};

}  // namespace nn
