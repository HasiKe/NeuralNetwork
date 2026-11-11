#include "nn/layers/layernorm.hpp"

#include <cmath>
#include <sstream>

#include "../serialize.hpp"

namespace nn {

std::string LayerNorm::name() const { return "LayerNorm"; }

void LayerNorm::build(std::size_t input_dim, Rng&) {
    m_in = m_out = input_dim;
    m_g = Matrix(1, input_dim, 1.0f);
    m_b = Matrix(1, input_dim);
    m_dg = Matrix(1, input_dim);
    m_db = Matrix(1, input_dim);
}

void LayerNorm::forward(const Matrix& in, Matrix& out, bool) {
    if (in.cols() != m_in) throw std::invalid_argument("LayerNorm: input width mismatch");
    const std::size_t B = in.rows(), D = m_in;
    out.resize(B, D);
    m_xhat.resize(B, D);
    m_rstd.resize(B, 1);
    const real invD = 1.0f / static_cast<real>(D);
    for (std::size_t i = 0; i < B; ++i) {
        const real* x = in.row(i);
        real mean = 0, var = 0;
        NN_SIMD_REDUCE(+, mean) for (std::size_t j = 0; j < D; ++j) mean += x[j];
        mean *= invD;
        NN_SIMD_REDUCE(+, var) for (std::size_t j = 0; j < D; ++j) var += (x[j] - mean) * (x[j] - mean);
        const real rstd = 1.0f / std::sqrt(var * invD + m_eps);
        m_rstd[i] = rstd;
        real* xh = m_xhat.row(i);
        real* y = out.row(i);
        const real* g = m_g.data();
        const real* b = m_b.data();
        NN_SIMD for (std::size_t j = 0; j < D; ++j) {
            xh[j] = (x[j] - mean) * rstd;
            y[j] = xh[j] * g[j] + b[j];
        }
    }
}

void LayerNorm::backward(const Matrix& in, const Matrix&, const Matrix& grad_out, Matrix* grad_in) {
    const std::size_t B = in.rows(), D = m_in;
    m_dg.zero();
    m_db.zero();
    if (grad_in) grad_in->resize(B, D);
    const real invD = 1.0f / static_cast<real>(D);
    for (std::size_t i = 0; i < B; ++i) {
        const real* dy = grad_out.row(i);
        const real* xh = m_xhat.row(i);
        real* dg = m_dg.data();
        real* db = m_db.data();
        const real* g = m_g.data();
        real s1 = 0, s2 = 0;
        for (std::size_t j = 0; j < D; ++j) {
            dg[j] += dy[j] * xh[j];
            db[j] += dy[j];
            const real dxh = dy[j] * g[j];
            s1 += dxh;
            s2 += dxh * xh[j];
        }
        if (grad_in) {
            real* dx = grad_in->row(i);
            const real r = m_rstd[i];
            NN_SIMD for (std::size_t j = 0; j < D; ++j)
                dx[j] = r * (dy[j] * g[j] - invD * s1 - xh[j] * invD * s2);
        }
    }
}

void LayerNorm::parameters(std::vector<ParamRef>& p) {
    p.push_back({m_g.data(), m_dg.data(), m_g.size(), false});
    p.push_back({m_b.data(), m_db.data(), m_b.size(), false});
}

void LayerNorm::save(std::ostream& os) const {
    io::put<real>(os, m_eps);
    io::put_matrix(os, m_g);
    io::put_matrix(os, m_b);
}

void LayerNorm::load(std::istream& is) {
    m_eps = io::get<real>(is);
    io::get_matrix(is, m_g);
    io::get_matrix(is, m_b);
    m_in = m_out = m_g.size();
    m_dg = Matrix(1, m_in);
    m_db = Matrix(1, m_in);
}

}  // namespace nn
