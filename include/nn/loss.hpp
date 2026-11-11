#pragma once

#include "matrix.hpp"

namespace nn {

enum class Loss : int {
    MeanSquaredError = 0,
    MeanAbsoluteError,
    Huber,                    ///< delta = 1
    CategoricalCrossEntropy,  ///< targets one-hot (or distributions); fused with Softmax output
    BinaryCrossEntropy,       ///< targets in [0,1]; fused with Sigmoid output
};

const char* to_string(Loss l);

/// Mean loss over the batch. If `grad` != nullptr it receives dL/d(pred)
/// (or dL/dz when `fused` is true, i.e. softmax/sigmoid output + CE).
real compute_loss(Loss loss, const Matrix& pred, const Matrix& target, Matrix* grad, bool fused);

}  // namespace nn
