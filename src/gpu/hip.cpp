// HIP backend (AMD ROCm; also NVIDIA through HIP's CUDA layer): GEMM via hipBLAS.
#include <hip/hip_runtime_api.h>
#if __has_include(<hipblas/hipblas.h>)
#include <hipblas/hipblas.h>
#else
#include <hipblas.h>
#endif

#include <string>

#include "backend.hpp"

namespace nn::gpu {

namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw Error(std::string("HIP ") + what + ": " + hipGetErrorString(e));
}
void check(hipblasStatus_t s, const char* what) {
    if (s != HIPBLAS_STATUS_SUCCESS) throw Error(std::string("hipBLAS ") + what + " failed (status " + std::to_string(static_cast<int>(s)) + ")");
}

class HipDevice final : public Device {
public:
    explicit HipDevice(int index) : m_index(index) {
        check(hipSetDevice(index), "hipSetDevice");
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, index), "hipGetDeviceProperties");
        m_name = std::string("HIP: ") + prop.name;
        check(hipStreamCreate(&m_stream), "hipStreamCreate");
        check(hipblasCreate(&m_handle), "hipblasCreate");
        check(hipblasSetStream(m_handle, m_stream), "hipblasSetStream");
    }

    ~HipDevice() override {
        (void)hipSetDevice(m_index);
        for (auto& b : m_buf)
            if (b.ptr) (void)hipFree(b.ptr);
        if (m_handle) hipblasDestroy(m_handle);
        if (m_stream) (void)hipStreamDestroy(m_stream);
    }

    std::string name() const override { return m_name; }

    void gemm(bool ta, bool tb, std::size_t M, std::size_t N, std::size_t K, real alpha,
              const real* A, std::size_t lda, const real* B, std::size_t ldb, real beta, real* C,
              std::size_t ldc) override {
        check(hipSetDevice(m_index), "hipSetDevice");
        const std::size_t na = ta ? span(K, M, lda) : span(M, K, lda);
        const std::size_t nb = tb ? span(N, K, ldb) : span(K, N, ldb);
        const std::size_t nc = span(M, N, ldc);
        float* a = buffer(0, na);
        float* b = buffer(1, nb);
        float* c = buffer(2, nc);
        check(hipMemcpyAsync(a, A, na * sizeof(real), hipMemcpyHostToDevice, m_stream), "copy A");
        check(hipMemcpyAsync(b, B, nb * sizeof(real), hipMemcpyHostToDevice, m_stream), "copy B");
        if (beta != 0 || ldc != N)  // keep padding between rows intact
            check(hipMemcpyAsync(c, C, nc * sizeof(real), hipMemcpyHostToDevice, m_stream), "copy C");
        // Row-major C = op(A) op(B)  <=>  column-major C^T = op(B)^T op(A)^T
        check(hipblasSgemm(m_handle, tb ? HIPBLAS_OP_T : HIPBLAS_OP_N, ta ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                           static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha, b,
                           static_cast<int>(ldb), a, static_cast<int>(lda), &beta, c, static_cast<int>(ldc)),
              "sgemm");
        check(hipMemcpyAsync(C, c, nc * sizeof(real), hipMemcpyDeviceToHost, m_stream), "copy C back");
        check(hipStreamSynchronize(m_stream), "hipStreamSynchronize");
    }

private:
    struct Buf { float* ptr = nullptr; std::size_t cap = 0; };

    float* buffer(int slot, std::size_t n) {
        Buf& b = m_buf[slot];
        n = n ? n : 1;
        if (n > b.cap) {
            if (b.ptr) (void)hipFree(b.ptr);
            b.ptr = nullptr;
            b.cap = 0;
            check(hipMalloc(reinterpret_cast<void**>(&b.ptr), n * sizeof(real)), "hipMalloc");
            b.cap = n;
        }
        return b.ptr;
    }

    int m_index;
    hipStream_t m_stream = nullptr;
    hipblasHandle_t m_handle = nullptr;
    Buf m_buf[3];
    std::string m_name;
};

}  // namespace

std::unique_ptr<Device> create_hip(int index) {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || index < 0 || index >= n) return nullptr;
    return std::make_unique<HipDevice>(index);
}

}  // namespace nn::gpu
