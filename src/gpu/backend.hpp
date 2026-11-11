#pragma once

// Internal interface implemented by every GPU backend.

#include <memory>
#include <stdexcept>
#include <string>

#include "nn/config.hpp"

namespace nn::gpu {

class Device {
public:
    virtual ~Device() = default;
    virtual std::string name() const = 0;

    /// Row-major C = alpha * op(A) * op(B) + beta * C on host pointers.
    /// Implementations copy the operands to the device, run, and copy C back.
    /// With beta == 0, C is not read. Throws gpu::Error on failure.
    virtual void gemm(bool ta, bool tb, std::size_t M, std::size_t N, std::size_t K, real alpha,
                      const real* A, std::size_t lda, const real* B, std::size_t ldb, real beta,
                      real* C, std::size_t ldc) = 0;
};

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Number of floats spanned by a row-major rows x cols block with leading dim ld.
inline std::size_t span(std::size_t rows, std::size_t cols, std::size_t ld) {
    return rows == 0 || cols == 0 ? 0 : (rows - 1) * ld + cols;
}

// Factories; each returns nullptr if no device is available. Only the ones
// whose backend is compiled in are defined.
std::unique_ptr<Device> create_opencl(int index);
std::unique_ptr<Device> create_vulkan(int index);
std::unique_ptr<Device> create_cuda(int index);
std::unique_ptr<Device> create_hip(int index);
std::unique_ptr<Device> create_sycl(int index);

/// Called by kernels::gemm. Returns true if the product was computed on the GPU.
bool try_gemm(bool ta, bool tb, std::size_t M, std::size_t N, std::size_t K, real alpha,
              const real* A, std::size_t lda, const real* B, std::size_t ldb, real beta, real* C,
              std::size_t ldc);

}  // namespace nn::gpu
