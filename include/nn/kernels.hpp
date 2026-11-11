#pragma once

#include "config.hpp"

/// Low level, cache-blocked and vectorised compute kernels.
/// All matrices are row-major with explicit leading dimensions.
namespace nn::kernels {

/// C = alpha * op(A) * op(B) + beta * C
/// op(A) is M x K, op(B) is K x N, C is M x N.
/// With beta == 0, C need not be initialised (NaNs in C are ignored).
void gemm(bool trans_a, bool trans_b, std::size_t M, std::size_t N, std::size_t K, real alpha,
          const real* A, std::size_t lda, const real* B, std::size_t ldb, real beta, real* C,
          std::size_t ldc);

/// y += alpha * x
void axpy(std::size_t n, real alpha, const real* x, real* y);
/// x *= alpha
void scale(std::size_t n, real alpha, real* x);
/// dot product
real dot(std::size_t n, const real* x, const real* y);
/// out[c] = sum_r M[r, c]  (column sums of a rows x cols block, leading dim ld)
void col_sums(std::size_t rows, std::size_t cols, const real* M, std::size_t ld, real* out,
              bool accumulate = false);
/// M[r, :] += v  for each row
void add_row_vector(std::size_t rows, std::size_t cols, real* M, std::size_t ld, const real* v);

}  // namespace nn::kernels
