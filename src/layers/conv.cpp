#include "nn/layers/conv.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

#include "../serialize.hpp"
#include "nn/kernels.hpp"
#include "nn/layers/dense.hpp"

namespace nn {

namespace {
std::size_t conv_out(std::size_t in, std::size_t k, std::size_t s, std::size_t p) {
    if (in + 2 * p < k) throw std::invalid_argument("Conv: kernel larger than padded input");
    return (in + 2 * p - k) / s + 1;
}
void put_shape(std::ostream& os, const Shape3& s) {
    io::put_size(os, s.channels);
    io::put_size(os, s.height);
    io::put_size(os, s.width);
}
Shape3 get_shape(std::istream& is) {
    Shape3 s;
    s.channels = io::get_size(is);
    s.height = io::get_size(is);
    s.width = io::get_size(is);
    return s;
}
}  // namespace

// ---------------------------------------------------------------- Conv2D

std::string Conv2D::name() const {
    std::ostringstream s;
    s << "Conv2D(" << m_filters << ", " << m_k << "x" << m_k << ", s=" << m_stride << ", p=" << m_pad
      << ", " << to_string(m_act.kind) << ") -> " << m_filters << "x" << m_oh << "x" << m_ow;
    return s.str();
}

void Conv2D::build(std::size_t input_dim, Rng& rng) {
    if (m_shape.size() != input_dim)
        throw std::invalid_argument("Conv2D: input shape does not match previous layer width");
    if (m_act.kind == ActivationKind::Softmax)
        throw std::invalid_argument("Conv2D: softmax activation not supported");
    m_oh = conv_out(m_shape.height, m_k, m_stride, m_pad);
    m_ow = conv_out(m_shape.width, m_k, m_stride, m_pad);
    m_in = input_dim;
    m_out = m_filters * m_oh * m_ow;
    const std::size_t ckk = m_shape.channels * m_k * m_k;
    m_w.resize(m_filters, ckk);
    init_weights(m_w, ckk, m_filters * m_k * m_k, Init::Auto, m_act.kind, rng);
    m_b = Matrix(1, m_filters);
    m_dw = Matrix(m_filters, ckk);
    m_db = Matrix(1, m_filters);
}

void Conv2D::forward(const Matrix& in, Matrix& out, bool training) {
    if (in.cols() != m_in) throw std::invalid_argument("Conv2D: input width mismatch");
    const std::size_t B = in.rows(), C = m_shape.channels, IH = m_shape.height, IW = m_shape.width;
    const std::size_t K = m_k, P = m_oh * m_ow, BP = B * P, CKK = C * K * K, Fn = m_filters;
    m_col.resize(CKK, BP);
    // im2col
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (CKK * BP > 65536)
#endif
    for (std::size_t r = 0; r < CKK; ++r) {
        const std::size_t c = r / (K * K), ky = (r / K) % K, kx = r % K;
        real* dst = m_col.row(r);
        for (std::size_t b = 0; b < B; ++b) {
            const real* src = in.row(b) + c * IH * IW;
            for (std::size_t oy = 0; oy < m_oh; ++oy) {
                const long iy = static_cast<long>(oy * m_stride + ky) - static_cast<long>(m_pad);
                real* d = dst + b * P + oy * m_ow;
                if (iy < 0 || iy >= static_cast<long>(IH)) {
                    std::memset(d, 0, m_ow * sizeof(real));
                    continue;
                }
                const real* srow = src + iy * IW;
                for (std::size_t ox = 0; ox < m_ow; ++ox) {
                    const long ix = static_cast<long>(ox * m_stride + kx) - static_cast<long>(m_pad);
                    d[ox] = (ix < 0 || ix >= static_cast<long>(IW)) ? 0.0f : srow[ix];
                }
            }
        }
    }
    m_y.resize(Fn, BP);
    kernels::gemm(false, false, Fn, BP, CKK, 1.0f, m_w.data(), CKK, m_col.data(), BP, 0.0f, m_y.data(), BP);
    // reorder [F x B*P] -> [B x F*P] and add bias
    const bool keep_z = training && activation_backward_needs_z(m_act.kind);
    Matrix& z = keep_z ? m_z : out;
    z.resize(B, Fn * P);
    out.resize(B, Fn * P);
    for (std::size_t b = 0; b < B; ++b)
        for (std::size_t f = 0; f < Fn; ++f) {
            const real* s = m_y.row(f) + b * P;
            real* d = z.row(b) + f * P;
            const real bias = m_b[f];
            NN_SIMD for (std::size_t p = 0; p < P; ++p) d[p] = s[p] + bias;
        }
    activation_forward(m_act, B, Fn * P, z.data(), out.data());
}

void Conv2D::backward(const Matrix& in, const Matrix& out, const Matrix& grad_out, Matrix* grad_in) {
    const std::size_t B = in.rows(), C = m_shape.channels, IH = m_shape.height, IW = m_shape.width;
    const std::size_t K = m_k, P = m_oh * m_ow, BP = B * P, CKK = C * K * K, Fn = m_filters;
    Matrix& dz = m_y;  // forward output buffer is free during backward
    dz.resize(B, Fn * P);
    activation_backward(m_act, B, Fn * P, m_z.data(), out.data(), grad_out.data(), dz.data());
    m_dy.resize(Fn, BP);
    for (std::size_t b = 0; b < B; ++b)
        for (std::size_t f = 0; f < Fn; ++f)
            std::memcpy(m_dy.row(f) + b * P, dz.row(b) + f * P, P * sizeof(real));
    kernels::gemm(false, true, Fn, CKK, BP, 1.0f, m_dy.data(), BP, m_col.data(), BP, 0.0f, m_dw.data(), CKK);
    for (std::size_t f = 0; f < Fn; ++f) {
        real s = 0;
        const real* d = m_dy.row(f);
        NN_SIMD_REDUCE(+, s) for (std::size_t i = 0; i < BP; ++i) s += d[i];
        m_db[f] = s;
    }
    if (!grad_in) return;
    m_dcol.resize(CKK, BP);
    kernels::gemm(true, false, CKK, BP, Fn, 1.0f, m_w.data(), CKK, m_dy.data(), BP, 0.0f, m_dcol.data(), BP);
    grad_in->resize(B, m_in);
    grad_in->zero();
    // col2im (parallel over samples: no write conflicts)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (CKK * BP > 65536)
#endif
    for (std::size_t b = 0; b < B; ++b) {
        real* gi = grad_in->row(b);
        for (std::size_t r = 0; r < CKK; ++r) {
            const std::size_t c = r / (K * K), ky = (r / K) % K, kx = r % K;
            const real* src = m_dcol.row(r) + b * P;
            real* dst = gi + c * IH * IW;
            for (std::size_t oy = 0; oy < m_oh; ++oy) {
                const long iy = static_cast<long>(oy * m_stride + ky) - static_cast<long>(m_pad);
                if (iy < 0 || iy >= static_cast<long>(IH)) continue;
                for (std::size_t ox = 0; ox < m_ow; ++ox) {
                    const long ix = static_cast<long>(ox * m_stride + kx) - static_cast<long>(m_pad);
                    if (ix >= 0 && ix < static_cast<long>(IW)) dst[iy * IW + ix] += src[oy * m_ow + ox];
                }
            }
        }
    }
}

void Conv2D::parameters(std::vector<ParamRef>& p) {
    p.push_back({m_w.data(), m_dw.data(), m_w.size(), true});
    p.push_back({m_b.data(), m_db.data(), m_b.size(), false});
}

void Conv2D::save(std::ostream& os) const {
    put_shape(os, m_shape);
    io::put_size(os, m_filters);
    io::put_size(os, m_k);
    io::put_size(os, m_stride);
    io::put_size(os, m_pad);
    io::put<std::uint8_t>(os, static_cast<std::uint8_t>(m_act.kind));
    io::put<real>(os, m_act.alpha);
    io::put_matrix(os, m_w);
    io::put_matrix(os, m_b);
}

void Conv2D::load(std::istream& is) {
    m_shape = get_shape(is);
    m_filters = io::get_size(is);
    m_k = io::get_size(is);
    m_stride = io::get_size(is);
    m_pad = io::get_size(is);
    auto k = io::get<std::uint8_t>(is);
    io::expect(k < static_cast<std::uint8_t>(ActivationKind::Count_), "Conv2D: bad activation");
    m_act.kind = static_cast<ActivationKind>(k);
    m_act.alpha = io::get<real>(is);
    io::get_matrix(is, m_w);
    io::get_matrix(is, m_b);
    io::expect(m_stride > 0 && m_w.rows() == m_filters &&
                   m_w.cols() == m_shape.channels * m_k * m_k && m_b.size() == m_filters,
               "Conv2D: shape mismatch");
    m_oh = conv_out(m_shape.height, m_k, m_stride, m_pad);
    m_ow = conv_out(m_shape.width, m_k, m_stride, m_pad);
    m_in = m_shape.size();
    m_out = m_filters * m_oh * m_ow;
    m_dw = Matrix(m_w.rows(), m_w.cols());
    m_db = Matrix(1, m_filters);
}

// ---------------------------------------------------------------- MaxPool2D

std::string MaxPool2D::name() const {
    std::ostringstream s;
    s << "MaxPool2D(" << m_pool << ", s=" << m_stride << ") -> " << m_shape.channels << "x" << m_oh
      << "x" << m_ow;
    return s.str();
}

void MaxPool2D::build(std::size_t input_dim, Rng&) {
    if (m_shape.size() != input_dim)
        throw std::invalid_argument("MaxPool2D: input shape does not match previous layer width");
    m_oh = conv_out(m_shape.height, m_pool, m_stride, 0);
    m_ow = conv_out(m_shape.width, m_pool, m_stride, 0);
    m_in = input_dim;
    m_out = m_shape.channels * m_oh * m_ow;
}

void MaxPool2D::forward(const Matrix& in, Matrix& out, bool) {
    if (in.cols() != m_in) throw std::invalid_argument("MaxPool2D: input width mismatch");
    const std::size_t B = in.rows(), C = m_shape.channels, IH = m_shape.height, IW = m_shape.width;
    out.resize(B, m_out);
    m_argmax.resize(B * m_out);
    for (std::size_t b = 0; b < B; ++b) {
        const real* x = in.row(b);
        real* y = out.row(b);
        std::uint32_t* am = m_argmax.data() + b * m_out;
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t oy = 0; oy < m_oh; ++oy)
                for (std::size_t ox = 0; ox < m_ow; ++ox) {
                    real best = -std::numeric_limits<real>::infinity();
                    std::size_t bi = 0;
                    for (std::size_t ky = 0; ky < m_pool; ++ky)
                        for (std::size_t kx = 0; kx < m_pool; ++kx) {
                            const std::size_t idx =
                                c * IH * IW + (oy * m_stride + ky) * IW + ox * m_stride + kx;
                            if (x[idx] > best) { best = x[idx]; bi = idx; }
                        }
                    const std::size_t o = (c * m_oh + oy) * m_ow + ox;
                    y[o] = best;
                    am[o] = static_cast<std::uint32_t>(bi);
                }
    }
}

void MaxPool2D::backward(const Matrix& in, const Matrix&, const Matrix& grad_out, Matrix* grad_in) {
    if (!grad_in) return;
    const std::size_t B = in.rows();
    grad_in->resize(B, m_in);
    grad_in->zero();
    for (std::size_t b = 0; b < B; ++b) {
        const real* g = grad_out.row(b);
        real* d = grad_in->row(b);
        const std::uint32_t* am = m_argmax.data() + b * m_out;
        for (std::size_t o = 0; o < m_out; ++o) d[am[o]] += g[o];
    }
}

void MaxPool2D::save(std::ostream& os) const {
    put_shape(os, m_shape);
    io::put_size(os, m_pool);
    io::put_size(os, m_stride);
}

void MaxPool2D::load(std::istream& is) {
    m_shape = get_shape(is);
    m_pool = io::get_size(is);
    m_stride = io::get_size(is);
    io::expect(m_pool > 0 && m_stride > 0, "MaxPool2D: bad config");
    m_oh = conv_out(m_shape.height, m_pool, m_stride, 0);
    m_ow = conv_out(m_shape.width, m_pool, m_stride, 0);
    m_in = m_shape.size();
    m_out = m_shape.channels * m_oh * m_ow;
}

}  // namespace nn
