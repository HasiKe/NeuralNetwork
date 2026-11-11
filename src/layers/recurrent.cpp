#include "nn/layers/recurrent.hpp"

#include <cmath>
#include <cstring>
#include <sstream>

#include "../serialize.hpp"
#include "nn/fastmath.hpp"
#include "nn/kernels.hpp"

namespace nn {

namespace fm = fastmath;
using kernels::gemm;

// ---------------------------------------------------------------- common

void RecurrentBase::build_common(std::size_t input_dim, std::size_t G, Rng& rng) {
    if (m_units == 0 || m_T == 0) throw std::invalid_argument("Recurrent: units and timesteps must be > 0");
    if (input_dim % m_T != 0)
        throw std::invalid_argument("Recurrent: input width must be divisible by timesteps");
    m_F = input_dim / m_T;
    m_in = input_dim;
    m_out = m_return_seq ? m_T * m_units : m_units;
    const std::size_t H = m_units;
    const real k = 1.0f / std::sqrt(static_cast<real>(H));
    m_wx.resize(m_F, G * H);
    m_wh.resize(H, G * H);
    for (std::size_t i = 0; i < m_wx.size(); ++i) m_wx[i] = rng.uniform(-k, k);
    for (std::size_t i = 0; i < m_wh.size(); ++i) m_wh[i] = rng.uniform(-k, k);
    m_b = Matrix(1, G * H);
    m_dwx = Matrix(m_F, G * H);
    m_dwh = Matrix(H, G * H);
    m_db = Matrix(1, G * H);
}

void RecurrentBase::save_common(std::ostream& os) const {
    io::put_size(os, m_units);
    io::put_size(os, m_T);
    io::put_size(os, m_F);
    io::put<std::uint8_t>(os, m_return_seq ? 1 : 0);
    io::put_matrix(os, m_wx);
    io::put_matrix(os, m_wh);
    io::put_matrix(os, m_b);
}

void RecurrentBase::load_common(std::istream& is, std::size_t G) {
    m_units = io::get_size(is);
    m_T = io::get_size(is);
    m_F = io::get_size(is);
    m_return_seq = io::get<std::uint8_t>(is) != 0;
    io::get_matrix(is, m_wx);
    io::get_matrix(is, m_wh);
    io::get_matrix(is, m_b);
    const std::size_t H = m_units;
    io::expect(m_wx.rows() == m_F && m_wx.cols() == G * H && m_wh.rows() == H &&
                   m_wh.cols() == G * H && m_b.size() == G * H,
               "Recurrent: shape mismatch");
    m_in = m_T * m_F;
    m_out = m_return_seq ? m_T * H : H;
    m_dwx = Matrix(m_F, G * H);
    m_dwh = Matrix(H, G * H);
    m_db = Matrix(1, G * H);
}

void RecurrentBase::params_common(std::vector<ParamRef>& p) {
    p.push_back({m_wx.data(), m_dwx.data(), m_wx.size(), true});
    p.push_back({m_wh.data(), m_dwh.data(), m_wh.size(), true});
    p.push_back({m_b.data(), m_db.data(), m_b.size(), false});
}

void RecurrentBase::add_output_grad(const Matrix& grad_out, std::size_t t, std::size_t B,
                                    real* dh) const {
    const std::size_t H = m_units;
    if (m_return_seq) {
        for (std::size_t b = 0; b < B; ++b) kernels::axpy(H, 1.0f, grad_out.row(b) + t * H, dh + b * H);
    } else if (t == m_T - 1) {
        for (std::size_t b = 0; b < B; ++b) kernels::axpy(H, 1.0f, grad_out.row(b), dh + b * H);
    }
}

namespace {

// Write hidden states into the layer output.
void emit_output(const Matrix& hs, std::size_t B, std::size_t T, std::size_t H, bool seq, Matrix& out) {
    if (seq) {
        out.resize(B, T * H);
        for (std::size_t b = 0; b < B; ++b)
            std::memcpy(out.row(b), hs.row(b) + H, T * H * sizeof(real));
    } else {
        out.resize(B, H);
        for (std::size_t b = 0; b < B; ++b)
            std::memcpy(out.row(b), hs.row(b) + T * H, H * sizeof(real));
    }
}

void init_states(Matrix& s, std::size_t B, std::size_t T, std::size_t H) {
    s.resize(B, (T + 1) * H);
    for (std::size_t b = 0; b < B; ++b) std::memset(s.row(b), 0, H * sizeof(real));
}

}  // namespace

// ---------------------------------------------------------------- SimpleRNN

std::string SimpleRNN::name() const {
    std::ostringstream s;
    s << "SimpleRNN(" << m_units << ", T=" << m_T << ", " << to_string(m_act.kind)
      << (m_return_seq ? ", seq" : "") << ")";
    return s.str();
}

void SimpleRNN::build(std::size_t input_dim, Rng& rng) {
    if (m_act.kind == ActivationKind::Softmax)
        throw std::invalid_argument("SimpleRNN: softmax activation not supported");
    build_common(input_dim, 1, rng);
}

void SimpleRNN::forward(const Matrix& in, Matrix& out, bool) {
    if (in.cols() != m_in) throw std::invalid_argument("SimpleRNN: input width mismatch");
    const std::size_t B = in.rows(), T = m_T, H = m_units, F = m_F;
    m_z.resize(B * T, H);
    gemm(false, false, B * T, H, F, 1.0f, in.data(), F, m_wx.data(), H, 0.0f, m_z.data(), H);
    kernels::add_row_vector(B * T, H, m_z.data(), H, m_b.data());
    init_states(m_hs, B, T, H);
    const std::size_t lh = (T + 1) * H, lz = T * H;
    for (std::size_t t = 0; t < T; ++t) {
        real* z = m_z.data() + t * H;
        if (t > 0) gemm(false, false, B, H, H, 1.0f, m_hs.data() + t * H, lh, m_wh.data(), H, 1.0f, z, lz);
        for (std::size_t b = 0; b < B; ++b)
            activation_forward(m_act, 1, H, z + b * lz, m_hs.data() + b * lh + (t + 1) * H);
    }
    emit_output(m_hs, B, T, H, m_return_seq, out);
}

void SimpleRNN::backward(const Matrix& in, const Matrix&, const Matrix& grad_out, Matrix* grad_in) {
    const std::size_t B = in.rows(), T = m_T, H = m_units, F = m_F;
    const std::size_t lh = (T + 1) * H, lz = T * H;
    m_dgates.resize(B * T, H);
    m_dh.resize(B, H);
    m_dh_next = Matrix(B, H);
    m_dwh.zero();
    for (std::size_t t = T; t-- > 0;) {
        std::memcpy(m_dh.data(), m_dh_next.data(), B * H * sizeof(real));
        add_output_grad(grad_out, t, B, m_dh.data());
        real* dz = m_dgates.data() + t * H;
        for (std::size_t b = 0; b < B; ++b)
            activation_backward(m_act, 1, H, m_z.data() + b * lz + t * H,
                                m_hs.data() + b * lh + (t + 1) * H, m_dh.row(b), dz + b * lz);
        if (t > 0) {
            gemm(false, true, B, H, H, 1.0f, dz, lz, m_wh.data(), H, 0.0f, m_dh_next.data(), H);
            gemm(true, false, H, H, B, 1.0f, m_hs.data() + t * H, lh, dz, lz, 1.0f, m_dwh.data(), H);
        }
    }
    gemm(true, false, F, H, B * T, 1.0f, in.data(), F, m_dgates.data(), H, 0.0f, m_dwx.data(), H);
    kernels::col_sums(B * T, H, m_dgates.data(), H, m_db.data());
    if (grad_in) {
        grad_in->resize(B, T * F);
        gemm(false, true, B * T, F, H, 1.0f, m_dgates.data(), H, m_wx.data(), H, 0.0f, grad_in->data(), F);
    }
}

void SimpleRNN::save(std::ostream& os) const {
    save_common(os);
    io::put<std::uint8_t>(os, static_cast<std::uint8_t>(m_act.kind));
    io::put<real>(os, m_act.alpha);
}

void SimpleRNN::load(std::istream& is) {
    load_common(is, 1);
    auto k = io::get<std::uint8_t>(is);
    io::expect(k < static_cast<std::uint8_t>(ActivationKind::Count_), "SimpleRNN: bad activation");
    m_act.kind = static_cast<ActivationKind>(k);
    m_act.alpha = io::get<real>(is);
}

// ---------------------------------------------------------------- LSTM

std::string LSTM::name() const {
    std::ostringstream s;
    s << "LSTM(" << m_units << ", T=" << m_T << (m_return_seq ? ", seq" : "") << ")";
    return s.str();
}

void LSTM::build(std::size_t input_dim, Rng& rng) {
    build_common(input_dim, 4, rng);
    for (std::size_t j = m_units; j < 2 * m_units; ++j) m_b[j] = 1.0f;  // forget gate bias
}

void LSTM::forward(const Matrix& in, Matrix& out, bool) {
    if (in.cols() != m_in) throw std::invalid_argument("LSTM: input width mismatch");
    const std::size_t B = in.rows(), T = m_T, H = m_units, F = m_F, G = 4 * H;
    m_xproj.resize(B * T, G);
    gemm(false, false, B * T, G, F, 1.0f, in.data(), F, m_wx.data(), G, 0.0f, m_xproj.data(), G);
    kernels::add_row_vector(B * T, G, m_xproj.data(), G, m_b.data());
    init_states(m_hs, B, T, H);
    init_states(m_cs, B, T, H);
    m_tanhc.resize(B * T, H);
    const std::size_t lh = (T + 1) * H, lg = T * G;
    for (std::size_t t = 0; t < T; ++t) {
        real* gt = m_xproj.data() + t * G;
        if (t > 0) gemm(false, false, B, G, H, 1.0f, m_hs.data() + t * H, lh, m_wh.data(), G, 1.0f, gt, lg);
        for (std::size_t b = 0; b < B; ++b) {
            real* NN_RESTRICT g = gt + b * lg;
            const real* NN_RESTRICT cp = m_cs.data() + b * lh + t * H;
            real* NN_RESTRICT c = m_cs.data() + b * lh + (t + 1) * H;
            real* NN_RESTRICT h = m_hs.data() + b * lh + (t + 1) * H;
            real* NN_RESTRICT tc = m_tanhc.data() + (b * T + t) * H;
            NN_SIMD for (std::size_t j = 0; j < H; ++j) {
                const real ig = fm::sigmoid(g[j]);
                const real fg = fm::sigmoid(g[H + j]);
                const real gg = fm::tanh(g[2 * H + j]);
                const real og = fm::sigmoid(g[3 * H + j]);
                g[j] = ig;
                g[H + j] = fg;
                g[2 * H + j] = gg;
                g[3 * H + j] = og;
                const real cn = fg * cp[j] + ig * gg;
                c[j] = cn;
                const real tcn = fm::tanh(cn);
                tc[j] = tcn;
                h[j] = og * tcn;
            }
        }
    }
    emit_output(m_hs, B, T, H, m_return_seq, out);
}

void LSTM::backward(const Matrix& in, const Matrix&, const Matrix& grad_out, Matrix* grad_in) {
    const std::size_t B = in.rows(), T = m_T, H = m_units, F = m_F, G = 4 * H;
    const std::size_t lh = (T + 1) * H, lg = T * G;
    m_dgates.resize(B * T, G);
    m_dh.resize(B, H);
    m_dh_next = Matrix(B, H);
    m_dc = Matrix(B, H);
    m_dwh.zero();
    for (std::size_t t = T; t-- > 0;) {
        std::memcpy(m_dh.data(), m_dh_next.data(), B * H * sizeof(real));
        add_output_grad(grad_out, t, B, m_dh.data());
        real* dgt = m_dgates.data() + t * G;
        for (std::size_t b = 0; b < B; ++b) {
            const real* NN_RESTRICT g = m_xproj.data() + (b * T + t) * G;
            const real* NN_RESTRICT cp = m_cs.data() + b * lh + t * H;
            const real* NN_RESTRICT tc = m_tanhc.data() + (b * T + t) * H;
            const real* NN_RESTRICT dh = m_dh.row(b);
            real* NN_RESTRICT dcn = m_dc.row(b);
            real* NN_RESTRICT dg = dgt + b * lg;
            NN_SIMD for (std::size_t j = 0; j < H; ++j) {
                const real ig = g[j], fg = g[H + j], gg = g[2 * H + j], og = g[3 * H + j];
                const real dc = dh[j] * og * (1.0f - tc[j] * tc[j]) + dcn[j];
                dg[j] = dc * gg * ig * (1.0f - ig);
                dg[H + j] = dc * cp[j] * fg * (1.0f - fg);
                dg[2 * H + j] = dc * ig * (1.0f - gg * gg);
                dg[3 * H + j] = dh[j] * tc[j] * og * (1.0f - og);
                dcn[j] = dc * fg;
            }
        }
        if (t > 0) {
            gemm(false, true, B, H, G, 1.0f, dgt, lg, m_wh.data(), G, 0.0f, m_dh_next.data(), H);
            gemm(true, false, H, G, B, 1.0f, m_hs.data() + t * H, lh, dgt, lg, 1.0f, m_dwh.data(), G);
        }
    }
    gemm(true, false, F, G, B * T, 1.0f, in.data(), F, m_dgates.data(), G, 0.0f, m_dwx.data(), G);
    kernels::col_sums(B * T, G, m_dgates.data(), G, m_db.data());
    if (grad_in) {
        grad_in->resize(B, T * F);
        gemm(false, true, B * T, F, G, 1.0f, m_dgates.data(), G, m_wx.data(), G, 0.0f, grad_in->data(), F);
    }
}

// ---------------------------------------------------------------- GRU

std::string GRU::name() const {
    std::ostringstream s;
    s << "GRU(" << m_units << ", T=" << m_T << (m_return_seq ? ", seq" : "") << ")";
    return s.str();
}

void GRU::build(std::size_t input_dim, Rng& rng) {
    build_common(input_dim, 3, rng);
    m_bh = Matrix(1, 3 * m_units);
    m_dbh = Matrix(1, 3 * m_units);
}

void GRU::parameters(std::vector<ParamRef>& p) {
    params_common(p);
    p.push_back({m_bh.data(), m_dbh.data(), m_bh.size(), false});
}

void GRU::forward(const Matrix& in, Matrix& out, bool) {
    if (in.cols() != m_in) throw std::invalid_argument("GRU: input width mismatch");
    const std::size_t B = in.rows(), T = m_T, H = m_units, F = m_F, G = 3 * H;
    m_xproj.resize(B * T, G);
    gemm(false, false, B * T, G, F, 1.0f, in.data(), F, m_wx.data(), G, 0.0f, m_xproj.data(), G);
    kernels::add_row_vector(B * T, G, m_xproj.data(), G, m_b.data());
    m_hproj.resize(B * T, G);
    init_states(m_hs, B, T, H);
    const std::size_t lh = (T + 1) * H, lg = T * G;
    for (std::size_t t = 0; t < T; ++t) {
        real* hp_t = m_hproj.data() + t * G;
        if (t > 0) {
            gemm(false, false, B, G, H, 1.0f, m_hs.data() + t * H, lh, m_wh.data(), G, 0.0f, hp_t, lg);
        } else {
            for (std::size_t b = 0; b < B; ++b) std::memset(hp_t + b * lg, 0, G * sizeof(real));
        }
        for (std::size_t b = 0; b < B; ++b) {
            real* NN_RESTRICT x = m_xproj.data() + (b * T + t) * G;
            real* NN_RESTRICT hp = hp_t + b * lg;
            const real* NN_RESTRICT hprev = m_hs.data() + b * lh + t * H;
            real* NN_RESTRICT h = m_hs.data() + b * lh + (t + 1) * H;
            const real* NN_RESTRICT bh = m_bh.data();
            NN_SIMD for (std::size_t j = 0; j < G; ++j) hp[j] += bh[j];
            NN_SIMD for (std::size_t j = 0; j < H; ++j) {
                const real r = fm::sigmoid(x[j] + hp[j]);
                const real z = fm::sigmoid(x[H + j] + hp[H + j]);
                const real n = fm::tanh(x[2 * H + j] + r * hp[2 * H + j]);
                x[j] = r;
                x[H + j] = z;
                x[2 * H + j] = n;
                h[j] = (1.0f - z) * n + z * hprev[j];
            }
        }
    }
    emit_output(m_hs, B, T, H, m_return_seq, out);
}

void GRU::backward(const Matrix& in, const Matrix&, const Matrix& grad_out, Matrix* grad_in) {
    const std::size_t B = in.rows(), T = m_T, H = m_units, F = m_F, G = 3 * H;
    const std::size_t lh = (T + 1) * H, lg = T * G;
    m_dgates.resize(B * T, G);
    m_dhproj.resize(B * T, G);
    m_dh.resize(B, H);
    m_dh_next = Matrix(B, H);
    m_dwh.zero();
    for (std::size_t t = T; t-- > 0;) {
        std::memcpy(m_dh.data(), m_dh_next.data(), B * H * sizeof(real));
        add_output_grad(grad_out, t, B, m_dh.data());
        real* dxt = m_dgates.data() + t * G;
        real* dht = m_dhproj.data() + t * G;
        for (std::size_t b = 0; b < B; ++b) {
            const real* NN_RESTRICT x = m_xproj.data() + (b * T + t) * G;  // r, z, n
            const real* NN_RESTRICT hp = m_hproj.data() + (b * T + t) * G;
            const real* NN_RESTRICT hprev = m_hs.data() + b * lh + t * H;
            const real* NN_RESTRICT dh = m_dh.row(b);
            real* NN_RESTRICT dnext = m_dh_next.row(b);
            real* NN_RESTRICT dx = dxt + b * lg;
            real* NN_RESTRICT dhp = dht + b * lg;
            NN_SIMD for (std::size_t j = 0; j < H; ++j) {
                const real r = x[j], z = x[H + j], n = x[2 * H + j];
                const real dn = dh[j] * (1.0f - z) * (1.0f - n * n);
                const real dz = dh[j] * (hprev[j] - n) * z * (1.0f - z);
                const real dr = dn * hp[2 * H + j] * r * (1.0f - r);
                dx[j] = dr;
                dx[H + j] = dz;
                dx[2 * H + j] = dn;
                dhp[j] = dr;
                dhp[H + j] = dz;
                dhp[2 * H + j] = dn * r;
                dnext[j] = dh[j] * z;
            }
        }
        if (t > 0) {
            gemm(false, true, B, H, G, 1.0f, dht, lg, m_wh.data(), G, 1.0f, m_dh_next.data(), H);
            gemm(true, false, H, G, B, 1.0f, m_hs.data() + t * H, lh, dht, lg, 1.0f, m_dwh.data(), G);
        }
    }
    kernels::col_sums(B * T, G, m_dhproj.data(), G, m_dbh.data());
    gemm(true, false, F, G, B * T, 1.0f, in.data(), F, m_dgates.data(), G, 0.0f, m_dwx.data(), G);
    kernels::col_sums(B * T, G, m_dgates.data(), G, m_db.data());
    if (grad_in) {
        grad_in->resize(B, T * F);
        gemm(false, true, B * T, F, G, 1.0f, m_dgates.data(), G, m_wx.data(), G, 0.0f, grad_in->data(), F);
    }
}

void GRU::save(std::ostream& os) const {
    save_common(os);
    io::put_matrix(os, m_bh);
}

void GRU::load(std::istream& is) {
    load_common(is, 3);
    io::get_matrix(is, m_bh);
    io::expect(m_bh.size() == 3 * m_units, "GRU: shape mismatch");
    m_dbh = Matrix(1, 3 * m_units);
}

}  // namespace nn
