#include "nn/layers/dense.hpp"

#include <cmath>
#include <sstream>

#include "../serialize.hpp"
#include "nn/kernels.hpp"
#include "nn/loss.hpp"

namespace nn {

void init_weights(Matrix& w, std::size_t fan_in, std::size_t fan_out, Init init, ActivationKind act,
                  Rng& rng) {
    if (init == Init::Auto) {
        switch (act) {
            case ActivationKind::ReLU:
            case ActivationKind::LeakyReLU:
            case ActivationKind::ELU:
            case ActivationKind::GELU:
            case ActivationKind::Swish:
            case ActivationKind::Mish:
            case ActivationKind::HardSwish:
            case ActivationKind::ReLU6:
                init = Init::HeNormal;
                break;
            case ActivationKind::SELU:
                init = Init::LeCunNormal;
                break;
            default:
                init = Init::GlorotUniform;
        }
    }
    const real fi = static_cast<real>(fan_in ? fan_in : 1);
    const real fo = static_cast<real>(fan_out ? fan_out : 1);
    const std::size_t n = w.size();
    real* p = w.data();
    switch (init) {
        case Init::GlorotUniform: {
            const real l = std::sqrt(6.0f / (fi + fo));
            for (std::size_t i = 0; i < n; ++i) p[i] = rng.uniform(-l, l);
            break;
        }
        case Init::GlorotNormal: {
            const real s = std::sqrt(2.0f / (fi + fo));
            for (std::size_t i = 0; i < n; ++i) p[i] = rng.normal() * s;
            break;
        }
        case Init::HeUniform: {
            const real l = std::sqrt(6.0f / fi);
            for (std::size_t i = 0; i < n; ++i) p[i] = rng.uniform(-l, l);
            break;
        }
        case Init::HeNormal: {
            const real s = std::sqrt(2.0f / fi);
            for (std::size_t i = 0; i < n; ++i) p[i] = rng.normal() * s;
            break;
        }
        case Init::LeCunNormal: {
            const real s = std::sqrt(1.0f / fi);
            for (std::size_t i = 0; i < n; ++i) p[i] = rng.normal() * s;
            break;
        }
        case Init::Uniform01:
            for (std::size_t i = 0; i < n; ++i) p[i] = rng.uniform();
            break;
        case Init::Zeros:
        default:
            w.zero();
    }
}

std::string Dense::name() const {
    std::ostringstream s;
    s << "Dense(" << m_units << ", " << to_string(m_act.kind) << ")";
    return s.str();
}

void Dense::build(std::size_t input_dim, Rng& rng) {
    if (m_units == 0) throw std::invalid_argument("Dense: units must be > 0");
    m_in = input_dim;
    m_out = m_units;
    m_w.resize(m_in, m_out);
    init_weights(m_w, m_in, m_out, m_init, m_act.kind, rng);
    m_b = Matrix(1, m_out);
    m_dw = Matrix(m_in, m_out);
    m_db = Matrix(1, m_out);
}

bool Dense::can_fuse_output(int loss) const {
    return (static_cast<Loss>(loss) == Loss::CategoricalCrossEntropy &&
            m_act.kind == ActivationKind::Softmax) ||
           (static_cast<Loss>(loss) == Loss::BinaryCrossEntropy &&
            m_act.kind == ActivationKind::Sigmoid);
}

void Dense::forward(const Matrix& in, Matrix& out, bool training) {
    if (in.cols() != m_in) throw std::invalid_argument("Dense: input width mismatch");
    const std::size_t B = in.rows();
    const bool keep_z = training && activation_backward_needs_z(m_act.kind) && !m_fused_output;
    Matrix& z = keep_z ? m_z : out;
    z.resize(B, m_out);
    out.resize(B, m_out);
    kernels::gemm(false, false, B, m_out, m_in, 1.0f, in.data(), m_in, m_w.data(), m_out, 0.0f,
                  z.data(), m_out);
    if (m_use_bias) kernels::add_row_vector(B, m_out, z.data(), m_out, m_b.data());
    activation_forward(m_act, B, m_out, z.data(), out.data());
}

void Dense::backward(const Matrix& in, const Matrix& out, const Matrix& grad_out, Matrix* grad_in) {
    const std::size_t B = in.rows();
    const real* dz = grad_out.data();
    if (!m_fused_output && m_act.kind != ActivationKind::Linear) {
        m_dz.resize(B, m_out);
        activation_backward(m_act, B, m_out, m_z.data(), out.data(), grad_out.data(), m_dz.data());
        dz = m_dz.data();
    }
    // dW = X^T dZ, db = sum(dZ), dX = dZ W^T
    kernels::gemm(true, false, m_in, m_out, B, 1.0f, in.data(), m_in, dz, m_out, 0.0f, m_dw.data(),
                  m_out);
    if (m_use_bias) kernels::col_sums(B, m_out, dz, m_out, m_db.data());
    if (grad_in) {
        grad_in->resize(B, m_in);
        kernels::gemm(false, true, B, m_in, m_out, 1.0f, dz, m_out, m_w.data(), m_out, 0.0f,
                      grad_in->data(), m_in);
    }
}

void Dense::parameters(std::vector<ParamRef>& p) {
    p.push_back({m_w.data(), m_dw.data(), m_w.size(), true});
    if (m_use_bias) p.push_back({m_b.data(), m_db.data(), m_b.size(), false});
}

void Dense::save(std::ostream& os) const {
    io::put_size(os, m_units);
    io::put<std::uint8_t>(os, static_cast<std::uint8_t>(m_act.kind));
    io::put<real>(os, m_act.alpha);
    io::put<std::uint8_t>(os, m_use_bias ? 1 : 0);
    io::put_matrix(os, m_w);
    io::put_matrix(os, m_b);
}

void Dense::load(std::istream& is) {
    m_units = io::get_size(is);
    auto k = io::get<std::uint8_t>(is);
    io::expect(k < static_cast<std::uint8_t>(ActivationKind::Count_), "Dense: bad activation");
    m_act.kind = static_cast<ActivationKind>(k);
    m_act.alpha = io::get<real>(is);
    m_use_bias = io::get<std::uint8_t>(is) != 0;
    io::get_matrix(is, m_w);
    io::get_matrix(is, m_b);
    io::expect(m_w.cols() == m_units && m_b.size() == m_units, "Dense: shape mismatch");
    m_in = m_w.rows();
    m_out = m_units;
    m_dw = Matrix(m_in, m_out);
    m_db = Matrix(1, m_out);
}

}  // namespace nn
