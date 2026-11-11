#pragma once

#include <memory>
#include <vector>

#include "layer.hpp"

namespace nn {

/// Base optimizer. State buffers are allocated lazily on the first step and
/// matched to parameters by position.
class Optimizer {
public:
    explicit Optimizer(real lr, real weight_decay = 0) : m_lr(lr), m_wd(weight_decay) {}
    virtual ~Optimizer() = default;

    virtual void step(const std::vector<ParamRef>& params) = 0;
    virtual void reset() { m_t = 0; }

    real learning_rate() const { return m_lr; }
    void set_learning_rate(real lr) { m_lr = lr; }
    real weight_decay() const { return m_wd; }
    /// Clip gradients by global L2 norm before the update (0 = off).
    void set_clip_norm(real c) { m_clip = c; }

protected:
    /// Applies gradient clipping; returns the scale factor for gradients.
    real clip_scale(const std::vector<ParamRef>& params) const;
    real m_lr, m_wd, m_clip = 0;
    std::size_t m_t = 0;
};

/// Stochastic gradient descent with optional (Nesterov) momentum.
class SGD final : public Optimizer {
public:
    explicit SGD(real lr = 0.01f, real momentum = 0, bool nesterov = false, real weight_decay = 0)
        : Optimizer(lr, weight_decay), m_mu(momentum), m_nesterov(nesterov) {}
    void step(const std::vector<ParamRef>& params) override;
    void reset() override { Optimizer::reset(); m_v.clear(); }

private:
    real m_mu;
    bool m_nesterov;
    std::vector<std::vector<real>> m_v;
};

/// Adam / AdamW (decoupled weight decay when `decoupled` is true).
class Adam final : public Optimizer {
public:
    explicit Adam(real lr = 1e-3f, real beta1 = 0.9f, real beta2 = 0.999f, real eps = 1e-8f,
                  real weight_decay = 0, bool decoupled = true)
        : Optimizer(lr, weight_decay), m_b1(beta1), m_b2(beta2), m_eps(eps), m_decoupled(decoupled) {}
    void step(const std::vector<ParamRef>& params) override;
    void reset() override { Optimizer::reset(); m_m.clear(); m_v.clear(); }

private:
    real m_b1, m_b2, m_eps;
    bool m_decoupled;
    std::vector<std::vector<real>> m_m, m_v;
};

class RMSProp final : public Optimizer {
public:
    explicit RMSProp(real lr = 1e-3f, real rho = 0.9f, real eps = 1e-7f, real weight_decay = 0)
        : Optimizer(lr, weight_decay), m_rho(rho), m_eps(eps) {}
    void step(const std::vector<ParamRef>& params) override;
    void reset() override { Optimizer::reset(); m_s.clear(); }

private:
    real m_rho, m_eps;
    std::vector<std::vector<real>> m_s;
};

class Adagrad final : public Optimizer {
public:
    explicit Adagrad(real lr = 1e-2f, real eps = 1e-7f, real weight_decay = 0)
        : Optimizer(lr, weight_decay), m_eps(eps) {}
    void step(const std::vector<ParamRef>& params) override;
    void reset() override { Optimizer::reset(); m_s.clear(); }

private:
    real m_eps;
    std::vector<std::vector<real>> m_s;
};

/// Learning-rate schedules applied once per epoch by Network::fit.
struct LRSchedule {
    enum class Kind { Constant, StepDecay, Exponential, Cosine, ReduceOnPlateau } kind = Kind::Constant;
    real factor = 0.5f;          ///< StepDecay/Exponential/Plateau multiplier
    std::size_t step_size = 10;  ///< StepDecay: epochs between decays
    std::size_t patience = 5;    ///< Plateau: epochs without improvement
    real min_lr = 1e-6f;

    static LRSchedule constant() { return {}; }
    static LRSchedule step_decay(std::size_t every, real factor = 0.5f) {
        LRSchedule s; s.kind = Kind::StepDecay; s.step_size = every; s.factor = factor; return s;
    }
    static LRSchedule exponential(real factor = 0.95f) {
        LRSchedule s; s.kind = Kind::Exponential; s.factor = factor; return s;
    }
    static LRSchedule cosine(real min_lr = 1e-6f) {
        LRSchedule s; s.kind = Kind::Cosine; s.min_lr = min_lr; return s;
    }
    /// Automatic learning rate: shrink when the monitored loss stops improving.
    static LRSchedule reduce_on_plateau(std::size_t patience = 5, real factor = 0.5f,
                                        real min_lr = 1e-6f) {
        LRSchedule s; s.kind = Kind::ReduceOnPlateau; s.patience = patience; s.factor = factor;
        s.min_lr = min_lr; return s;
    }
};

}  // namespace nn
