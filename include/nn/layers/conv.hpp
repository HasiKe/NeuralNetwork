#pragma once

#include "../activation.hpp"
#include "../layer.hpp"

namespace nn {

/// Shape of an image tensor flattened as CHW inside one matrix row.
struct Shape3 {
    std::size_t channels = 1, height = 1, width = 1;
    std::size_t size() const { return channels * height * width; }
};

/// 2D convolution neurons (im2col + GEMM over the whole batch).
class Conv2D final : public Layer {
public:
    Conv2D() = default;
    Conv2D(Shape3 input, std::size_t filters, std::size_t kernel, std::size_t stride = 1,
           std::size_t padding = 0, Activation act = ActivationKind::ReLU)
        : m_shape(input), m_filters(filters), m_k(kernel), m_stride(stride), m_pad(padding),
          m_act(act) {}

    LayerType type() const override { return LayerType::Conv2D; }
    std::string name() const override;
    void build(std::size_t input_dim, Rng& rng) override;
    void forward(const Matrix& in, Matrix& out, bool training) override;
    void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                  Matrix* grad_in) override;
    void parameters(std::vector<ParamRef>& p) override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    Shape3 input_shape() const { return m_shape; }
    Shape3 output_shape() const { return {m_filters, m_oh, m_ow}; }

private:
    Shape3 m_shape;
    std::size_t m_filters = 0, m_k = 3, m_stride = 1, m_pad = 0, m_oh = 0, m_ow = 0;
    Activation m_act;
    Matrix m_w, m_b, m_dw, m_db;  // W: [F x C*K*K]
    Matrix m_col;                 // [C*K*K x B*OH*OW]
    Matrix m_y, m_dy, m_dcol;     // [F x B*OH*OW]
    Matrix m_z;                   // pre-activation in output layout
};

/// 2D max pooling (no parameters).
class MaxPool2D final : public Layer {
public:
    MaxPool2D() = default;
    MaxPool2D(Shape3 input, std::size_t pool, std::size_t stride = 0)
        : m_shape(input), m_pool(pool), m_stride(stride ? stride : pool) {}

    LayerType type() const override { return LayerType::MaxPool2D; }
    std::string name() const override;
    void build(std::size_t input_dim, Rng& rng) override;
    void forward(const Matrix& in, Matrix& out, bool training) override;
    void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                  Matrix* grad_in) override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    Shape3 input_shape() const { return m_shape; }
    Shape3 output_shape() const { return {m_shape.channels, m_oh, m_ow}; }

private:
    Shape3 m_shape;
    std::size_t m_pool = 2, m_stride = 2, m_oh = 0, m_ow = 0;
    std::vector<std::uint32_t> m_argmax;
};

}  // namespace nn
