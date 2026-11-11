// CUDA backend (NVIDIA): GEMM via cuBLAS. Host-only C++, no nvcc required.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>

#include "backend.hpp"

namespace nn::gpu {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw Error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}
void check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw Error(std::string("cuBLAS ") + what + " failed (status " + std::to_string(static_cast<int>(s)) + ")");
}

class CudaDevice final : public Device {
public:
    explicit CudaDevice(int index) : m_index(index) {
        check(cudaSetDevice(index), "cudaSetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, index), "cudaGetDeviceProperties");
        m_name = std::string("CUDA: ") + prop.name;
        check(cudaStreamCreate(&m_stream), "cudaStreamCreate");
        check(cublasCreate(&m_handle), "cublasCreate");
        check(cublasSetStream(m_handle, m_stream), "cublasSetStream");
    }

    ~CudaDevice() override {
        cudaSetDevice(m_index);
        for (auto& b : m_buf)
            if (b.ptr) cudaFree(b.ptr);
        if (m_handle) cublasDestroy(m_handle);
        if (m_stream) cudaStreamDestroy(m_stream);
    }

    std::string name() const override { return m_name; }

    void gemm(bool ta, bool tb, std::size_t M, std::size_t N, std::size_t K, real alpha,
              const real* A, std::size_t lda, const real* B, std::size_t ldb, real beta, real* C,
              std::size_t ldc) override {
        check(cudaSetDevice(m_index), "cudaSetDevice");
        const std::size_t na = ta ? span(K, M, lda) : span(M, K, lda);
        const std::size_t nb = tb ? span(N, K, ldb) : span(K, N, ldb);
        const std::size_t nc = span(M, N, ldc);
        float* a = buffer(0, na);
        float* b = buffer(1, nb);
        float* c = buffer(2, nc);
        check(cudaMemcpyAsync(a, A, na * sizeof(real), cudaMemcpyHostToDevice, m_stream), "copy A");
        check(cudaMemcpyAsync(b, B, nb * sizeof(real), cudaMemcpyHostToDevice, m_stream), "copy B");
        if (beta != 0 || ldc != N)  // keep padding between rows intact
            check(cudaMemcpyAsync(c, C, nc * sizeof(real), cudaMemcpyHostToDevice, m_stream), "copy C");
        // Row-major C = op(A) op(B)  <=>  column-major C^T = op(B)^T op(A)^T
        check(cublasSgemm(m_handle, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                          static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), &alpha, b,
                          static_cast<int>(ldb), a, static_cast<int>(lda), &beta, c, static_cast<int>(ldc)),
              "sgemm");
        check(cudaMemcpyAsync(C, c, nc * sizeof(real), cudaMemcpyDeviceToHost, m_stream), "copy C back");
        check(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize");
    }

private:
    struct Buf { float* ptr = nullptr; std::size_t cap = 0; };

    float* buffer(int slot, std::size_t n) {
        Buf& b = m_buf[slot];
        n = n ? n : 1;
        if (n > b.cap) {
            if (b.ptr) cudaFree(b.ptr);
            b.ptr = nullptr;
            b.cap = 0;
            check(cudaMalloc(reinterpret_cast<void**>(&b.ptr), n * sizeof(real)), "cudaMalloc");
            b.cap = n;
        }
        return b.ptr;
    }

    int m_index;
    cudaStream_t m_stream = nullptr;
    cublasHandle_t m_handle = nullptr;
    Buf m_buf[3];
    std::string m_name;
};

}  // namespace

std::unique_ptr<Device> create_cuda(int index) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || index < 0 || index >= n) return nullptr;
    return std::make_unique<CudaDevice>(index);
}

}  // namespace nn::gpu
