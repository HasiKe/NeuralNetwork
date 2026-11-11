#include "nn/layers/dropout.hpp"

#include <sstream>

#include "../serialize.hpp"

namespace nn {

std::string Dropout::name() const {
    std::ostringstream s;
    s << "Dropout(" << m_rate << ")";
    return s.str();
}

void Dropout::build(std::size_t input_dim, Rng& rng) {
    if (m_rate < 0 || m_rate >= 1) throw std::invalid_argument("Dropout: rate must be in [0, 1)");
    m_in = m_out = input_dim;
    m_rng.reseed(rng.next_u64());
}

void Dropout::forward(const Matrix& in, Matrix& out, bool training) {
    out.resize(in.rows(), in.cols());
    const std::size_t n = in.size();
    if (!training || m_rate == 0) {
        if (&out != &in) std::copy(in.data(), in.data() + n, out.data());
        return;
    }
    m_mask.resize(in.rows(), in.cols());
    const real keep = 1.0f - m_rate, scale = 1.0f / keep;
    // 64-bit random -> two 24-bit uniforms per draw
    const std::uint32_t thresh = static_cast<std::uint32_t>(keep * 16777216.0f);
    real* m = m_mask.data();
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        std::uint64_t r = m_rng.next_u64();
        m[i] = (static_cast<std::uint32_t>(r & 0xFFFFFF) < thresh) ? scale : 0.0f;
        m[i + 1] = (static_cast<std::uint32_t>((r >> 32) & 0xFFFFFF) < thresh) ? scale : 0.0f;
    }
    for (; i < n; ++i) m[i] = m_rng.uniform() < keep ? scale : 0.0f;
    const real* x = in.data();
    real* y = out.data();
    NN_SIMD for (std::size_t j = 0; j < n; ++j) y[j] = x[j] * m[j];
}

void Dropout::backward(const Matrix&, const Matrix&, const Matrix& grad_out, Matrix* grad_in) {
    if (!grad_in) return;
    grad_in->resize(grad_out.rows(), grad_out.cols());
    const std::size_t n = grad_out.size();
    const real* g = grad_out.data();
    real* d = grad_in->data();
    if (m_rate == 0 || m_mask.size() != n) {
        std::copy(g, g + n, d);
        return;
    }
    const real* m = m_mask.data();
    NN_SIMD for (std::size_t j = 0; j < n; ++j) d[j] = g[j] * m[j];
}

void Dropout::save(std::ostream& os) const {
    io::put<real>(os, m_rate);
    io::put_size(os, m_in);
}

void Dropout::load(std::istream& is) {
    m_rate = io::get<real>(is);
    m_in = m_out = io::get_size(is);
}

}  // namespace nn
