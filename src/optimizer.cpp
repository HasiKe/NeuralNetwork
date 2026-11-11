#include "nn/optimizer.hpp"

#include <cmath>

namespace nn {

namespace {
void ensure_state(std::vector<std::vector<real>>& s, const std::vector<ParamRef>& params) {
    if (s.size() == params.size()) return;
    s.resize(params.size());
    for (std::size_t i = 0; i < params.size(); ++i) s[i].assign(params[i].size, 0.0f);
}
}  // namespace

real Optimizer::clip_scale(const std::vector<ParamRef>& params) const {
    if (m_clip <= 0) return 1.0f;
    double sq = 0;
    for (const auto& p : params)
        for (std::size_t i = 0; i < p.size; ++i) sq += static_cast<double>(p.grad[i]) * p.grad[i];
    const double norm = std::sqrt(sq);
    return norm > m_clip ? static_cast<real>(m_clip / norm) : 1.0f;
}

void SGD::step(const std::vector<ParamRef>& params) {
    ++m_t;
    const real cs = clip_scale(params);
    if (m_mu != 0) ensure_state(m_v, params);
    for (std::size_t k = 0; k < params.size(); ++k) {
        const ParamRef& p = params[k];
        real* NN_RESTRICT w = p.value;
        const real* NN_RESTRICT g = p.grad;
        const real wd = p.decay ? m_wd : 0.0f, lr = m_lr, mu = m_mu;
        if (mu == 0) {
            NN_SIMD for (std::size_t i = 0; i < p.size; ++i) w[i] -= lr * (cs * g[i] + wd * w[i]);
        } else {
            real* NN_RESTRICT v = m_v[k].data();
            if (m_nesterov) {
                NN_SIMD for (std::size_t i = 0; i < p.size; ++i) {
                    const real gi = cs * g[i] + wd * w[i];
                    v[i] = mu * v[i] + gi;
                    w[i] -= lr * (gi + mu * v[i]);
                }
            } else {
                NN_SIMD for (std::size_t i = 0; i < p.size; ++i) {
                    v[i] = mu * v[i] + cs * g[i] + wd * w[i];
                    w[i] -= lr * v[i];
                }
            }
        }
    }
}

void Adam::step(const std::vector<ParamRef>& params) {
    ++m_t;
    const real cs = clip_scale(params);
    ensure_state(m_m, params);
    ensure_state(m_v, params);
    const real b1 = m_b1, b2 = m_b2, eps = m_eps;
    const real c1 = 1.0f / (1.0f - static_cast<real>(std::pow(b1, static_cast<double>(m_t))));
    const real c2 = 1.0f / (1.0f - static_cast<real>(std::pow(b2, static_cast<double>(m_t))));
    const real lr = m_lr;
    for (std::size_t k = 0; k < params.size(); ++k) {
        const ParamRef& p = params[k];
        real* NN_RESTRICT w = p.value;
        const real* NN_RESTRICT g = p.grad;
        real* NN_RESTRICT m = m_m[k].data();
        real* NN_RESTRICT v = m_v[k].data();
        const real wd = p.decay ? m_wd : 0.0f;
        const real l2 = m_decoupled ? 0.0f : wd;   // classic L2 in the gradient
        const real dw = m_decoupled ? lr * wd : 0.0f;  // AdamW decoupled decay
        NN_SIMD for (std::size_t i = 0; i < p.size; ++i) {
            const real gi = cs * g[i] + l2 * w[i];
            m[i] = b1 * m[i] + (1.0f - b1) * gi;
            v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
            w[i] -= lr * (m[i] * c1) / (std::sqrt(v[i] * c2) + eps) + dw * w[i];
        }
    }
}

void RMSProp::step(const std::vector<ParamRef>& params) {
    ++m_t;
    const real cs = clip_scale(params);
    ensure_state(m_s, params);
    const real rho = m_rho, eps = m_eps, lr = m_lr;
    for (std::size_t k = 0; k < params.size(); ++k) {
        const ParamRef& p = params[k];
        real* NN_RESTRICT w = p.value;
        const real* NN_RESTRICT g = p.grad;
        real* NN_RESTRICT s = m_s[k].data();
        const real wd = p.decay ? m_wd : 0.0f;
        NN_SIMD for (std::size_t i = 0; i < p.size; ++i) {
            const real gi = cs * g[i] + wd * w[i];
            s[i] = rho * s[i] + (1.0f - rho) * gi * gi;
            w[i] -= lr * gi / (std::sqrt(s[i]) + eps);
        }
    }
}

void Adagrad::step(const std::vector<ParamRef>& params) {
    ++m_t;
    const real cs = clip_scale(params);
    ensure_state(m_s, params);
    const real eps = m_eps, lr = m_lr;
    for (std::size_t k = 0; k < params.size(); ++k) {
        const ParamRef& p = params[k];
        real* NN_RESTRICT w = p.value;
        const real* NN_RESTRICT g = p.grad;
        real* NN_RESTRICT s = m_s[k].data();
        const real wd = p.decay ? m_wd : 0.0f;
        NN_SIMD for (std::size_t i = 0; i < p.size; ++i) {
            const real gi = cs * g[i] + wd * w[i];
            s[i] += gi * gi;
            w[i] -= lr * gi / (std::sqrt(s[i]) + eps);
        }
    }
}

}  // namespace nn
