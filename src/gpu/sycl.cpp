// SYCL backend (Intel oneAPI DPC++, AdaptiveCpp, ...): portable tiled GEMM kernel.
// This translation unit must be compiled by a SYCL compiler (e.g. icpx -fsycl).
#include <sycl/sycl.hpp>

#include <string>
#include <vector>

#include "backend.hpp"

namespace nn::gpu {

namespace {

constexpr int TS = 64, TK = 16, WPT = 4, RTS = 16;

class SyclDevice final : public Device {
public:
    explicit SyclDevice(const sycl::device& dev) : m_queue(dev, sycl::property::queue::in_order()) {
        m_name = "SYCL: " + dev.get_info<sycl::info::device::name>();
    }

    ~SyclDevice() override {
        for (auto& b : m_buf)
            if (b.ptr) sycl::free(b.ptr, m_queue);
    }

    std::string name() const override { return m_name; }

    void gemm(bool ta, bool tb, std::size_t M, std::size_t N, std::size_t K, real alpha,
              const real* A, std::size_t lda, const real* B, std::size_t ldb, real beta, real* C,
              std::size_t ldc) override {
        try {
            run(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } catch (const sycl::exception& e) {
            throw Error(std::string("SYCL: ") + e.what());
        }
    }

private:
    struct Buf { float* ptr = nullptr; std::size_t cap = 0; };

    float* buffer(int slot, std::size_t n) {
        Buf& b = m_buf[slot];
        n = n ? n : 1;
        if (n > b.cap) {
            if (b.ptr) sycl::free(b.ptr, m_queue);
            b.ptr = sycl::malloc_device<float>(n, m_queue);
            if (!b.ptr) throw Error("SYCL malloc_device failed");
            b.cap = n;
        }
        return b.ptr;
    }

    void run(bool ta_, bool tb_, std::size_t M_, std::size_t N_, std::size_t K_, real alpha,
             const real* A, std::size_t lda_, const real* B, std::size_t ldb_, real beta, real* C,
             std::size_t ldc_) {
        const std::size_t na = ta_ ? span(K_, M_, lda_) : span(M_, K_, lda_);
        const std::size_t nb = tb_ ? span(N_, K_, ldb_) : span(K_, N_, ldb_);
        const std::size_t nc = span(M_, N_, ldc_);
        float* a = buffer(0, na);
        float* b = buffer(1, nb);
        float* c = buffer(2, nc);
        m_queue.memcpy(a, A, na * sizeof(real));
        m_queue.memcpy(b, B, nb * sizeof(real));
        if (beta != 0 || ldc_ != N_) m_queue.memcpy(c, C, nc * sizeof(real));
        const int M = static_cast<int>(M_), N = static_cast<int>(N_), K = static_cast<int>(K_);
        const int lda = static_cast<int>(lda_), ldb = static_cast<int>(ldb_), ldc = static_cast<int>(ldc_);
        const int ta = ta_, tb = tb_;
        const sycl::range<2> global((M_ + TS - 1) / TS * RTS, (N_ + TS - 1) / TS * RTS);
        const sycl::range<2> local(RTS, RTS);
        m_queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 2> As(sycl::range<2>(TK, TS + 1), h);
            sycl::local_accessor<float, 2> Bs(sycl::range<2>(TK, TS + 1), h);
            h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
                const int ty = static_cast<int>(it.get_local_id(0)), tx = static_cast<int>(it.get_local_id(1));
                const int row0 = static_cast<int>(it.get_group(0)) * TS, col0 = static_cast<int>(it.get_group(1)) * TS;
                const int tid = ty * RTS + tx;
                float acc[WPT][WPT] = {};
                for (int k0 = 0; k0 < K; k0 += TK) {
                    for (int l = tid; l < TS * TK; l += RTS * RTS) {
                        const int i = l / TK, k = l % TK;
                        const int gi = row0 + i, gk = k0 + k;
                        As[k][i] = (gi < M && gk < K) ? (ta ? a[gk * lda + gi] : a[gi * lda + gk]) : 0.0f;
                        const int j = l % TS, kk = l / TS;
                        const int gj = col0 + j, gk2 = k0 + kk;
                        Bs[kk][j] = (gj < N && gk2 < K) ? (tb ? b[gj * ldb + gk2] : b[gk2 * ldb + gj]) : 0.0f;
                    }
                    sycl::group_barrier(it.get_group());
                    for (int k = 0; k < TK; ++k) {
                        float av[WPT], bv[WPT];
                        for (int w = 0; w < WPT; ++w) {
                            av[w] = As[k][ty + RTS * w];
                            bv[w] = Bs[k][tx + RTS * w];
                        }
                        for (int wi = 0; wi < WPT; ++wi)
                            for (int wj = 0; wj < WPT; ++wj) acc[wi][wj] = sycl::fma(av[wi], bv[wj], acc[wi][wj]);
                    }
                    sycl::group_barrier(it.get_group());
                }
                for (int wi = 0; wi < WPT; ++wi) {
                    const int r = row0 + ty + RTS * wi;
                    if (r >= M) continue;
                    for (int wj = 0; wj < WPT; ++wj) {
                        const int col = col0 + tx + RTS * wj;
                        if (col >= N) continue;
                        const float v = alpha * acc[wi][wj];
                        c[r * ldc + col] = beta != 0.0f ? v + beta * c[r * ldc + col] : v;
                    }
                }
            });
        });
        m_queue.memcpy(C, c, nc * sizeof(real)).wait_and_throw();
    }

    sycl::queue m_queue;
    Buf m_buf[3];
    std::string m_name;
};

}  // namespace

std::unique_ptr<Device> create_sycl(int index) {
    std::vector<sycl::device> gpus, others;
    for (const auto& d : sycl::device::get_devices())
        (d.is_gpu() ? gpus : others).push_back(d);
    gpus.insert(gpus.end(), others.begin(), others.end());
    if (index < 0 || static_cast<std::size_t>(index) >= gpus.size()) return nullptr;
    return std::make_unique<SyclDevice>(gpus[static_cast<std::size_t>(index)]);
}

}  // namespace nn::gpu
