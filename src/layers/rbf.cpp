#include "nn/layers/rbf.hpp"

#include <cmath>
#include <sstream>

#include "../serialize.hpp"
#include "nn/fastmath.hpp"
#include "nn/kernels.hpp"

namespace nn {

// Width parameter is stored as log(gamma) so gamma stays positive.

std::string RBF::name() const {
    std::ostringstream s;
    s << "RBF(" << m_units << ")";
    return s.str();
}

void RBF::build(std::size_t input_dim, Rng& rng) {
    if (m_units == 0) throw std::invalid_argument("RBF: units must be > 0");
    m_in = input_dim;
    m_out = m_units;
    m_c.resize(m_units, m_in);
    for (std::size_t i = 0; i < m_c.size(); ++i) m_c[i] = rng.uniform();
    m_gamma = Matrix(1, m_units, std::log(m_gamma0));
    m_dc = Matrix(m_units, m_in);
    m_dgamma = Matrix(1, m_units);
}

void RBF::forward(const Matrix& in, Matrix& out, bool) {
    if (in.cols() != m_in) throw std::invalid_argument("RBF: input width mismatch");
    const std::size_t B = in.rows(), U = m_units, D = m_in;
    m_d2.resize(B, U);
    out.resize(B, U);
    // d2 = |x|^2 + |c|^2 - 2 x c^T
    kernels::gemm(false, true, B, U, D, -2.0f, in.data(), D, m_c.data(), D, 0.0f, m_d2.data(), U);
    m_tmp.resize(1, U);
    for (std::size_t j = 0; j < U; ++j) m_tmp[j] = kernels::dot(D, m_c.row(j), m_c.row(j));
    for (std::size_t i = 0; i < B; ++i) {
        const real xn = kernels::dot(D, in.row(i), in.row(i));
        real* d = m_d2.row(i);
        real* y = out.row(i);
        const real* cn = m_tmp.data();
        const real* lg = m_gamma.data();
        NN_SIMD for (std::size_t j = 0; j < U; ++j) {
            real v = d[j] + xn + cn[j];
            v = v < 0 ? 0.0f : v;
            d[j] = v;
            y[j] = fastmath::exp(-fastmath::exp(lg[j]) * v);
        }
    }
}

void RBF::backward(const Matrix& in, const Matrix& out, const Matrix& grad_out, Matrix* grad_in) {
    const std::size_t B = in.rows(), U = m_units, D = m_in;
    // E = dL/d(d2) = -gamma * dy * y
    m_g.resize(B, U);
    m_tmp.resize(1, U);
    for (std::size_t j = 0; j < U; ++j) m_tmp[j] = std::exp(m_gamma[j]);
    m_dgamma.zero();
    for (std::size_t i = 0; i < B; ++i) {
        const real* dy = grad_out.row(i);
        const real* y = out.row(i);
        const real* d2 = m_d2.row(i);
        real* e = m_g.row(i);
        real* dg = m_dgamma.data();
        const real* g = m_tmp.data();
        NN_SIMD for (std::size_t j = 0; j < U; ++j) {
            e[j] = -g[j] * dy[j] * y[j];
            dg[j] += e[j] * d2[j];  // dL/dlog(gamma) = dL/dgamma * gamma
        }
    }
    // dC = -2 E^T X + 2 c * colsum(E)
    kernels::gemm(true, false, U, D, B, -2.0f, m_g.data(), U, in.data(), D, 0.0f, m_dc.data(), D);
    m_xn.resize(1, U);
    kernels::col_sums(B, U, m_g.data(), U, m_xn.data());
    for (std::size_t j = 0; j < U; ++j) kernels::axpy(D, 2.0f * m_xn[j], m_c.row(j), m_dc.row(j));
    if (grad_in) {
        // dX = 2 x * rowsum(E) - 2 E C
        grad_in->resize(B, D);
        kernels::gemm(false, false, B, D, U, -2.0f, m_g.data(), U, m_c.data(), D, 0.0f,
                      grad_in->data(), D);
        for (std::size_t i = 0; i < B; ++i) {
            real rs = 0;
            const real* e = m_g.row(i);
            for (std::size_t j = 0; j < U; ++j) rs += e[j];
            kernels::axpy(D, 2.0f * rs, in.row(i), grad_in->row(i));
        }
    }
}

void RBF::parameters(std::vector<ParamRef>& p) {
    p.push_back({m_c.data(), m_dc.data(), m_c.size(), false});
    p.push_back({m_gamma.data(), m_dgamma.data(), m_gamma.size(), false});
}

void RBF::save(std::ostream& os) const {
    io::put_size(os, m_units);
    io::put_matrix(os, m_c);
    io::put_matrix(os, m_gamma);
}

void RBF::load(std::istream& is) {
    m_units = io::get_size(is);
    io::get_matrix(is, m_c);
    io::get_matrix(is, m_gamma);
    io::expect(m_c.rows() == m_units && m_gamma.size() == m_units, "RBF: shape mismatch");
    m_in = m_c.cols();
    m_out = m_units;
    m_dc = Matrix(m_units, m_in);
    m_dgamma = Matrix(1, m_units);
}

}  // namespace nn
