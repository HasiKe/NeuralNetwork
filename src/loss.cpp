#include "nn/loss.hpp"

#include <cmath>
#include <stdexcept>

namespace nn {

const char* to_string(Loss l) {
    switch (l) {
        case Loss::MeanSquaredError: return "mse";
        case Loss::MeanAbsoluteError: return "mae";
        case Loss::Huber: return "huber";
        case Loss::CategoricalCrossEntropy: return "categorical_crossentropy";
        case Loss::BinaryCrossEntropy: return "binary_crossentropy";
    }
    return "unknown";
}

real compute_loss(Loss loss, const Matrix& pred, const Matrix& target, Matrix* grad, bool fused) {
    if (pred.rows() != target.rows() || pred.cols() != target.cols())
        throw std::invalid_argument("nn: prediction / target shape mismatch");
    const std::size_t n = pred.size();
    if (n == 0) return 0;
    const real* p = pred.data();
    const real* t = target.data();
    real* g = nullptr;
    if (grad) {
        grad->resize(pred.rows(), pred.cols());
        g = grad->data();
    }
    const real inv_n = 1.0f / static_cast<real>(n);
    const real inv_b = 1.0f / static_cast<real>(pred.rows());
    constexpr real eps = 1e-7f;
    double total = 0;

    switch (loss) {
        case Loss::MeanSquaredError:
            for (std::size_t i = 0; i < n; ++i) {
                const real d = p[i] - t[i];
                total += d * d;
                if (g) g[i] = 2.0f * d * inv_n;
            }
            return static_cast<real>(total * inv_n);
        case Loss::MeanAbsoluteError:
            for (std::size_t i = 0; i < n; ++i) {
                const real d = p[i] - t[i];
                total += std::fabs(d);
                if (g) g[i] = (d > 0 ? 1.0f : (d < 0 ? -1.0f : 0.0f)) * inv_n;
            }
            return static_cast<real>(total * inv_n);
        case Loss::Huber:
            for (std::size_t i = 0; i < n; ++i) {
                const real d = p[i] - t[i];
                const real a = std::fabs(d);
                total += a <= 1 ? 0.5f * d * d : a - 0.5f;
                if (g) g[i] = (d > 1 ? 1.0f : (d < -1 ? -1.0f : d)) * inv_n;
            }
            return static_cast<real>(total * inv_n);
        case Loss::CategoricalCrossEntropy:
            for (std::size_t i = 0; i < n; ++i) {
                const real pi = p[i] < eps ? eps : p[i];
                if (t[i] != 0) total -= t[i] * std::log(pi);
                if (g) g[i] = fused ? (p[i] - t[i]) * inv_b : -t[i] / pi * inv_b;
            }
            return static_cast<real>(total * inv_b);
        case Loss::BinaryCrossEntropy:
            for (std::size_t i = 0; i < n; ++i) {
                real pi = p[i] < eps ? eps : (p[i] > 1 - eps ? 1 - eps : p[i]);
                total -= t[i] * std::log(pi) + (1 - t[i]) * std::log(1 - pi);
                if (g) g[i] = fused ? (p[i] - t[i]) * inv_n : (pi - t[i]) / (pi * (1 - pi)) * inv_n;
            }
            return static_cast<real>(total * inv_n);
    }
    throw std::invalid_argument("nn: unknown loss");
}

}  // namespace nn
