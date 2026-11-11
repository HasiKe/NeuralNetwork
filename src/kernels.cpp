#include "nn/kernels.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef NN_USE_BLAS
#include <cblas.h>
#endif

namespace nn {

namespace {
int g_threads = 0;
}

void set_num_threads(int n) {
    g_threads = n;
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#endif
#ifdef NN_USE_BLAS
#if defined(OPENBLAS_VERSION) || defined(NN_OPENBLAS)
    if (n > 0) openblas_set_num_threads(n);
#endif
#endif
}

int num_threads() {
#ifdef _OPENMP
    return g_threads > 0 ? g_threads : omp_get_max_threads();
#else
    return 1;
#endif
}

const char* build_info() {
    return "nn " NN_STR(NN_VERSION_MAJOR) "." NN_STR(NN_VERSION_MINOR) "." NN_STR(NN_VERSION_PATCH)
#ifdef _OPENMP
           " +openmp"
#endif
#if defined(NN_HAS_OPENMP_SIMD) || defined(_OPENMP)
           " +simd"
#endif
#ifdef __AVX512F__
           " avx512"
#elif defined(__AVX2__)
           " avx2"
#elif defined(__AVX__)
           " avx"
#elif defined(__SSE2__)
           " sse2"
#elif defined(__ARM_NEON)
           " neon"
#endif
#ifdef __FMA__
           " fma"
#endif
#ifdef NN_USE_BLAS
           " +blas"
#endif
        ;
}

}  // namespace nn

namespace nn::kernels {

namespace {

constexpr std::size_t MB = 32;   // rows per tile
constexpr std::size_t NB = 256;  // cols per tile (B panel KB x NB fits in L2)
constexpr std::size_t KB = 256;  // depth per block
constexpr std::size_t kParallelFlops = std::size_t(1) << 18;

// C[0..rows, 0..nb] += alpha * a(i, k) * b(k, j) for k in [k0, k0+kb)
// a(i,k) = A[i*sai + k*sak], b(k, j) = Bp[(k - k0)*ldbp + j]
void micro_tile(std::size_t rows, std::size_t nb, std::size_t kb, real alpha,
                const real* NN_RESTRICT A, std::size_t sai, std::size_t sak,
                const real* NN_RESTRICT Bp, std::size_t ldbp, real* NN_RESTRICT C, std::size_t ldc) {
    std::size_t i = 0;
    for (; i + 4 <= rows; i += 4) {
        real* NN_RESTRICT c0 = C + (i + 0) * ldc;
        real* NN_RESTRICT c1 = C + (i + 1) * ldc;
        real* NN_RESTRICT c2 = C + (i + 2) * ldc;
        real* NN_RESTRICT c3 = C + (i + 3) * ldc;
        const real* a = A + i * sai;
        // Register-blocked 4 x 16 kernel: the C block stays in vector registers
        // for the whole k loop (8 accumulators with AVX2), B is streamed once.
        constexpr std::size_t JB = 16;
        std::size_t j0 = 0;
        for (; j0 + JB <= nb; j0 += JB) {
            real acc0[JB] = {}, acc1[JB] = {}, acc2[JB] = {}, acc3[JB] = {};
            for (std::size_t k = 0; k < kb; ++k) {
                const real a0 = a[k * sak], a1 = a[sai + k * sak];
                const real a2 = a[2 * sai + k * sak], a3 = a[3 * sai + k * sak];
                const real* NN_RESTRICT b = Bp + k * ldbp + j0;
                NN_SIMD
                for (std::size_t x = 0; x < JB; ++x) {
                    acc0[x] += a0 * b[x];
                    acc1[x] += a1 * b[x];
                    acc2[x] += a2 * b[x];
                    acc3[x] += a3 * b[x];
                }
            }
            NN_SIMD
            for (std::size_t x = 0; x < JB; ++x) {
                c0[j0 + x] += alpha * acc0[x];
                c1[j0 + x] += alpha * acc1[x];
                c2[j0 + x] += alpha * acc2[x];
                c3[j0 + x] += alpha * acc3[x];
            }
        }
        if (j0 == nb) continue;
        for (std::size_t k = 0; k < kb; ++k) {
            const real a0 = alpha * a[k * sak];
            const real a1 = alpha * a[sai + k * sak];
            const real a2 = alpha * a[2 * sai + k * sak];
            const real a3 = alpha * a[3 * sai + k * sak];
            const real* NN_RESTRICT b = Bp + k * ldbp;
            NN_SIMD
            for (std::size_t j = j0; j < nb; ++j) {
                const real bj = b[j];
                c0[j] += a0 * bj;
                c1[j] += a1 * bj;
                c2[j] += a2 * bj;
                c3[j] += a3 * bj;
            }
        }
    }
    for (; i < rows; ++i) {
        real* NN_RESTRICT c0 = C + i * ldc;
        const real* a = A + i * sai;
        for (std::size_t k = 0; k < kb; ++k) {
            const real a0 = alpha * a[k * sak];
            if (a0 == 0) continue;
            const real* NN_RESTRICT b = Bp + k * ldbp;
            NN_SIMD
            for (std::size_t j = 0; j < nb; ++j) c0[j] += a0 * b[j];
        }
    }
}

void scale_c(std::size_t M, std::size_t N, real beta, real* C, std::size_t ldc) {
    if (beta == 1) return;
    for (std::size_t i = 0; i < M; ++i) {
        real* c = C + i * ldc;
        if (beta == 0) {
            std::memset(c, 0, N * sizeof(real));
        } else {
            NN_SIMD
            for (std::size_t j = 0; j < N; ++j) c[j] *= beta;
        }
    }
}

}  // namespace

void gemm(bool ta, bool tb, std::size_t M, std::size_t N, std::size_t K, real alpha, const real* A,
          std::size_t lda, const real* B, std::size_t ldb, real beta, real* C, std::size_t ldc) {
    if (M == 0 || N == 0) return;
#ifdef NN_USE_BLAS
    if (K > 0) {
        cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), alpha, A,
                    static_cast<int>(lda), B, static_cast<int>(ldb), beta, C, static_cast<int>(ldc));
        return;
    }
#endif
    scale_c(M, N, beta, C, ldc);
    if (K == 0 || alpha == 0) return;

    // op(B) = B^T: transpose once into a contiguous K x N buffer so every tile
    // streams B rows (blocked transpose keeps both sides cache friendly).
    thread_local std::vector<real> bt;
    if (tb) {
        bt.resize(K * N);
        constexpr std::size_t TB = 32;
        real* NN_RESTRICT dst = bt.data();
        auto tr_block = [&](std::size_t j0) {
            const std::size_t je = std::min(N, j0 + TB);
            for (std::size_t k0 = 0; k0 < K; k0 += TB) {
                const std::size_t ke = std::min(K, k0 + TB);
                for (std::size_t j = j0; j < je; ++j)
                    for (std::size_t k = k0; k < ke; ++k) dst[k * N + j] = B[j * ldb + k];
            }
        };
        const std::size_t nblk = (N + TB - 1) / TB;
#ifdef _OPENMP
        if (N * K >= (std::size_t(1) << 16)) {
#pragma omp parallel for schedule(static)
            for (std::size_t b = 0; b < nblk; ++b) tr_block(b * TB);
        } else
#endif
        {
            for (std::size_t b = 0; b < nblk; ++b) tr_block(b * TB);
        }
        B = bt.data();
        ldb = N;
    }

    const std::size_t sai = ta ? 1 : lda;
    const std::size_t sak = ta ? lda : 1;
    const std::size_t tiles_m = (M + MB - 1) / MB;
    const std::size_t tiles_n = (N + NB - 1) / NB;
    const std::size_t tiles = tiles_m * tiles_n;
    auto run_tile = [&](std::size_t t) {
        const std::size_t tm = t / tiles_n, tn = t % tiles_n;
        const std::size_t i0 = tm * MB, j0 = tn * NB;
        const std::size_t rows = std::min(MB, M - i0), nb = std::min(NB, N - j0);
        for (std::size_t k0 = 0; k0 < K; k0 += KB) {
            const std::size_t kb = std::min(KB, K - k0);
            micro_tile(rows, nb, kb, alpha, A + i0 * sai + k0 * sak, sai, sak, B + k0 * ldb + j0, ldb,
                       C + i0 * ldc + j0, ldc);
        }
    };
#ifdef _OPENMP
    // Note: an `if(false)` clause still enters the OpenMP runtime, which costs
    // microseconds - far more than a small GEMM. Branch explicitly instead.
    if (M * N * K >= kParallelFlops && tiles > 1) {
#pragma omp parallel for schedule(static)
        for (std::size_t t = 0; t < tiles; ++t) run_tile(t);
        return;
    }
#endif
    for (std::size_t t = 0; t < tiles; ++t) run_tile(t);
}

void axpy(std::size_t n, real alpha, const real* NN_RESTRICT x, real* NN_RESTRICT y) {
    NN_SIMD
    for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

void scale(std::size_t n, real alpha, real* x) {
    NN_SIMD
    for (std::size_t i = 0; i < n; ++i) x[i] *= alpha;
}

real dot(std::size_t n, const real* x, const real* y) {
    real s = 0;
    NN_SIMD_REDUCE(+, s)
    for (std::size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

void col_sums(std::size_t rows, std::size_t cols, const real* M, std::size_t ld, real* NN_RESTRICT out,
              bool accumulate) {
    if (!accumulate) std::memset(out, 0, cols * sizeof(real));
    for (std::size_t r = 0; r < rows; ++r) {
        const real* NN_RESTRICT m = M + r * ld;
        NN_SIMD
        for (std::size_t c = 0; c < cols; ++c) out[c] += m[c];
    }
}

void add_row_vector(std::size_t rows, std::size_t cols, real* M, std::size_t ld,
                    const real* NN_RESTRICT v) {
    for (std::size_t r = 0; r < rows; ++r) {
        real* NN_RESTRICT m = M + r * ld;
        NN_SIMD
        for (std::size_t c = 0; c < cols; ++c) m[c] += v[c];
    }
}

}  // namespace nn::kernels
