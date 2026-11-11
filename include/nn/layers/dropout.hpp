#pragma once

#include "../layer.hpp"

namespace nn {

/// Inverted dropout: during training zeroes each unit with probability `rate`
/// and scales the rest by 1/(1-rate). Identity at inference time.
class Dropout final : public Layer {
public:
    Dropout() = default;
    explicit Dropout(real rate) : m_rate(rate) {}

    LayerType type() const override { return LayerType::Dropout; }
    std::string name() const override;
    void build(std::size_t input_dim, Rng& rng) override;
    void forward(const Matrix& in, Matrix& out, bool training) override;
    void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                  Matrix* grad_in) override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;
    void seed(std::uint64_t s) override { m_rng.reseed(s); }

private:
    real m_rate = 0.5f;
    Rng m_rng;
    Matrix m_mask;
};

}  // namespace nn
