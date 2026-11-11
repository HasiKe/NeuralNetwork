#pragma once

#include <cstddef>
#include <cstdint>

#define NN_VERSION_MAJOR 2
#define NN_VERSION_MINOR 0
#define NN_VERSION_PATCH 0
#define NN_STR_(x) #x
#define NN_STR(x) NN_STR_(x)

#if defined(__GNUC__) || defined(__clang__)
#define NN_RESTRICT __restrict__
#define NN_INLINE inline __attribute__((always_inline))
#define NN_LIKELY(x) __builtin_expect(!!(x), 1)
#define NN_UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
#define NN_RESTRICT __restrict
#define NN_INLINE __forceinline
#define NN_LIKELY(x) (x)
#define NN_UNLIKELY(x) (x)
#else
#define NN_RESTRICT
#define NN_INLINE inline
#define NN_LIKELY(x) (x)
#define NN_UNLIKELY(x) (x)
#endif

// Portable "vectorize this loop" hint. Works with -fopenmp or -fopenmp-simd and
// is ignored otherwise.
#define NN_PRAGMA(x) _Pragma(#x)
#if defined(_OPENMP) || defined(NN_HAS_OPENMP_SIMD)
#define NN_SIMD NN_PRAGMA(omp simd)
#define NN_SIMD_REDUCE(op, var) NN_PRAGMA(omp simd reduction(op : var))
#else
#define NN_SIMD
#define NN_SIMD_REDUCE(op, var)
#endif

namespace nn {

/// Scalar type used throughout the library.
using real = float;

/// Number of threads used by parallel kernels (0 = OpenMP default).
void set_num_threads(int n);
int num_threads();

/// Human readable build configuration (SIMD / OpenMP / BLAS).
const char* build_info();

}  // namespace nn
