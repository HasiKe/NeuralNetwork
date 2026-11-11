// OpenCL backend: vendor-neutral (Intel, AMD, NVIDIA, ARM Mali, Apple, ...).
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <vector>

#include "backend.hpp"

namespace nn::gpu {

namespace {

// 64x64 output tile per work-group of 16x16, each work-item computes 4x4.
const char* kSource = R"CLC(
#define TS 64
#define TK 16
#define WPT 4
#define RTS 16
__kernel __attribute__((reqd_work_group_size(RTS, RTS, 1)))
void sgemm(const int M, const int N, const int K, const float alpha,
           __global const float* A, const int lda, const int ta,
           __global const float* B, const int ldb, const int tb,
           const float beta, __global float* C, const int ldc) {
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int row0 = get_group_id(1) * TS, col0 = get_group_id(0) * TS;
    const int tid = ty * RTS + tx;
    __local float As[TK][TS + 1];
    __local float Bs[TK][TS + 1];
    float acc[WPT][WPT];
    for (int i = 0; i < WPT; ++i)
        for (int j = 0; j < WPT; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int l = tid; l < TS * TK; l += RTS * RTS) {
            const int i = l / TK, k = l % TK;
            const int gi = row0 + i, gk = k0 + k;
            As[k][i] = (gi < M && gk < K) ? (ta ? A[gk * lda + gi] : A[gi * lda + gk]) : 0.0f;
            const int j = l % TS, kk = l / TS;
            const int gj = col0 + j, gk2 = k0 + kk;
            Bs[kk][j] = (gj < N && gk2 < K) ? (tb ? B[gj * ldb + gk2] : B[gk2 * ldb + gj]) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; ++k) {
            float a[WPT], b[WPT];
            for (int w = 0; w < WPT; ++w) {
                a[w] = As[k][ty + RTS * w];
                b[w] = Bs[k][tx + RTS * w];
            }
            for (int wi = 0; wi < WPT; ++wi)
                for (int wj = 0; wj < WPT; ++wj) acc[wi][wj] = mad(a[wi], b[wj], acc[wi][wj]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int wi = 0; wi < WPT; ++wi) {
        const int r = row0 + ty + RTS * wi;
        if (r >= M) continue;
        for (int wj = 0; wj < WPT; ++wj) {
            const int c = col0 + tx + RTS * wj;
            if (c >= N) continue;
            const float v = alpha * acc[wi][wj];
            C[r * ldc + c] = beta != 0.0f ? v + beta * C[r * ldc + c] : v;
        }
    }
}
)CLC";

void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) throw Error(std::string("OpenCL ") + what + " failed (error " + std::to_string(err) + ")");
}

class OpenCLDevice final : public Device {
public:
    explicit OpenCLDevice(cl_device_id dev) : m_dev(dev) {
        cl_int err;
        m_ctx = clCreateContext(nullptr, 1, &m_dev, nullptr, nullptr, &err);
        check(err, "clCreateContext");
        m_queue = clCreateCommandQueue(m_ctx, m_dev, 0, &err);
        check(err, "clCreateCommandQueue");
        m_prog = clCreateProgramWithSource(m_ctx, 1, &kSource, nullptr, &err);
        check(err, "clCreateProgramWithSource");
        err = clBuildProgram(m_prog, 1, &m_dev, "-cl-mad-enable", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::size_t len = 0;
            clGetProgramBuildInfo(m_prog, m_dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            std::string log(len, '\0');
            clGetProgramBuildInfo(m_prog, m_dev, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
            throw Error("OpenCL kernel build failed: " + log);
        }
        m_kernel = clCreateKernel(m_prog, "sgemm", &err);
        check(err, "clCreateKernel");
        char buf[256] = {0};
        clGetDeviceInfo(m_dev, CL_DEVICE_NAME, sizeof buf - 1, buf, nullptr);
        m_name = std::string("OpenCL: ") + buf;
    }

    ~OpenCLDevice() override {
        for (auto& b : m_buf)
            if (b.mem) clReleaseMemObject(b.mem);
        if (m_kernel) clReleaseKernel(m_kernel);
        if (m_prog) clReleaseProgram(m_prog);
        if (m_queue) clReleaseCommandQueue(m_queue);
        if (m_ctx) clReleaseContext(m_ctx);
    }

    std::string name() const override { return m_name; }

    void gemm(bool ta, bool tb, std::size_t M, std::size_t N, std::size_t K, real alpha,
              const real* A, std::size_t lda, const real* B, std::size_t ldb, real beta, real* C,
              std::size_t ldc) override {
        const std::size_t na = ta ? span(K, M, lda) : span(M, K, lda);
        const std::size_t nb = tb ? span(N, K, ldb) : span(K, N, ldb);
        const std::size_t nc = span(M, N, ldc);
        cl_mem a = buffer(0, na), b = buffer(1, nb), c = buffer(2, nc);
        check(clEnqueueWriteBuffer(m_queue, a, CL_FALSE, 0, na * sizeof(real), A, 0, nullptr, nullptr), "write A");
        check(clEnqueueWriteBuffer(m_queue, b, CL_FALSE, 0, nb * sizeof(real), B, 0, nullptr, nullptr), "write B");
        if (beta != 0 || ldc != N)  // keep padding between rows intact
           
            check(clEnqueueWriteBuffer(m_queue, c, CL_FALSE, 0, nc * sizeof(real), C, 0, nullptr, nullptr), "write C");
        const cl_int iM = static_cast<cl_int>(M), iN = static_cast<cl_int>(N), iK = static_cast<cl_int>(K);
        const cl_int ilda = static_cast<cl_int>(lda), ildb = static_cast<cl_int>(ldb), ildc = static_cast<cl_int>(ldc);
        const cl_int ita = ta, itb = tb;
        cl_int err = 0;
        err |= clSetKernelArg(m_kernel, 0, sizeof iM, &iM);
        err |= clSetKernelArg(m_kernel, 1, sizeof iN, &iN);
        err |= clSetKernelArg(m_kernel, 2, sizeof iK, &iK);
        err |= clSetKernelArg(m_kernel, 3, sizeof alpha, &alpha);
        err |= clSetKernelArg(m_kernel, 4, sizeof a, &a);
        err |= clSetKernelArg(m_kernel, 5, sizeof ilda, &ilda);
        err |= clSetKernelArg(m_kernel, 6, sizeof ita, &ita);
        err |= clSetKernelArg(m_kernel, 7, sizeof b, &b);
        err |= clSetKernelArg(m_kernel, 8, sizeof ildb, &ildb);
        err |= clSetKernelArg(m_kernel, 9, sizeof itb, &itb);
        err |= clSetKernelArg(m_kernel, 10, sizeof beta, &beta);
        err |= clSetKernelArg(m_kernel, 11, sizeof c, &c);
        err |= clSetKernelArg(m_kernel, 12, sizeof ildc, &ildc);
        check(err, "clSetKernelArg");
        const std::size_t global[2] = {(N + 63) / 64 * 16, (M + 63) / 64 * 16};
        const std::size_t local[2] = {16, 16};
        check(clEnqueueNDRangeKernel(m_queue, m_kernel, 2, nullptr, global, local, 0, nullptr, nullptr), "launch");
        check(clEnqueueReadBuffer(m_queue, c, CL_TRUE, 0, nc * sizeof(real), C, 0, nullptr, nullptr), "read C");
    }

private:
    struct Buf { cl_mem mem = nullptr; std::size_t cap = 0; };

    cl_mem buffer(int slot, std::size_t n) {
        Buf& b = m_buf[slot];
        n = n ? n : 1;
        if (n > b.cap) {
            if (b.mem) clReleaseMemObject(b.mem);
            cl_int err;
            b.mem = clCreateBuffer(m_ctx, CL_MEM_READ_WRITE, n * sizeof(real), nullptr, &err);
            check(err, "clCreateBuffer");
            b.cap = n;
        }
        return b.mem;
    }

    cl_device_id m_dev;
    cl_context m_ctx = nullptr;
    cl_command_queue m_queue = nullptr;
    cl_program m_prog = nullptr;
    cl_kernel m_kernel = nullptr;
    Buf m_buf[3];
    std::string m_name;
};

}  // namespace

std::unique_ptr<Device> create_opencl(int index) {
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return nullptr;
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    // GPUs first, then accelerators/CPUs as a fallback
    std::vector<cl_device_id> gpus, others;
    for (auto p : plats) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
        for (auto d : devs) {
            cl_device_type t = 0;
            clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof t, &t, nullptr);
            (t & CL_DEVICE_TYPE_GPU ? gpus : others).push_back(d);
        }
    }
    gpus.insert(gpus.end(), others.begin(), others.end());
    if (index < 0 || static_cast<std::size_t>(index) >= gpus.size()) return nullptr;
    return std::make_unique<OpenCLDevice>(gpus[static_cast<std::size_t>(index)]);
}

}  // namespace nn::gpu
