#pragma once

/// Automatic learning rate (opt-in): see FitOptions::auto_lr.

#include <cstddef>
#include <vector>

#include "config.hpp"

namespace nn {

/// Options of the automatic learning rate. Disabled by default; `AutoLR::on()`
/// enables it with settings that need no tuning. While enabled, the learning rate
/// given to the optimizer is ignored and overwritten.
struct AutoLR {
    bool enabled = false;

    // ---- start value: LR range test (weights and optimizer state are not changed)
    real min_lr = 1e-6f;           ///< first learning rate of the range test
    real max_lr = 10.0f;           ///< last learning rate of the range test
    std::size_t range_steps = 100; ///< mini-batches of the range test
    /// Start value = (learning rate at the lowest smoothed loss) / range_divisor.
    real range_divisor = 3.0f;
    /// > 0: use this start value and skip the range test.
    real initial_lr = 0;

    // ---- schedule
    real warmup_fraction = 0.05f;  ///< share of all steps for the linear warmup (at most 5 epochs)
    std::size_t patience = 0;      ///< epochs without improvement before lr *= factor (0 = epochs / 10, 5..20)
    real factor = 0.5f;
    real floor_ratio = 1e-3f;      ///< lowest lr = start value * floor_ratio
    std::size_t stop_patience = 0; ///< early stopping, epochs without improvement (0 = 4 * patience)

    // ---- safety
    /// Held out for monitoring if fit() gets no validation data (0 = monitor the training loss).
    real validation_split = 0.15f;
    real divergence_factor = 4.0f; ///< loss above best * factor counts as divergence
    std::size_t max_rollbacks = 6;

    static AutoLR on() {
        AutoLR a;
        a.enabled = true;
        return a;
    }
};

/// Result of the LR range test.
struct LRFindResult {
    real lr = 0;           ///< suggested start learning rate
    real min_loss_lr = 0;  ///< learning rate at the lowest smoothed loss
    bool diverged = false; ///< the test stopped because the loss blew up
    std::vector<real> lrs, losses;  ///< curve: learning rate and smoothed loss per step
};

/// What the automatic learning rate chose and did (History::auto_lr).
struct AutoLRReport {
    bool used = false;
    LRFindResult range;  ///< empty if AutoLR::initial_lr was given
    real start_lr = 0, final_lr = 0;
    std::size_t warmup_steps = 0, patience = 0, stop_patience = 0;
    std::size_t reductions = 0, rollbacks = 0;
    std::size_t validation_rows = 0;  ///< 0 = training loss was monitored
    /// 1-based epoch of the restored weights (0 = initial weights).
    std::size_t best_epoch = 0;
    real best_loss = 0;  ///< monitored loss of the restored weights
    bool stopped_early = false;
};

namespace detail {
/// Reduce-on-plateau bookkeeping shared by LRSchedule::reduce_on_plateau and AutoLR.
struct Plateau {
    real best;
    std::size_t wait = 0;
    /// True when `patience` epochs passed without a relative improvement of 1e-4.
    bool update(real monitored, std::size_t patience) {
        if (monitored < best * (1 - 1e-4f)) {
            best = monitored;
            wait = 0;
        } else if (++wait >= patience) {
            wait = 0;
            return true;
        }
        return false;
    }
};
}  // namespace detail

}  // namespace nn
