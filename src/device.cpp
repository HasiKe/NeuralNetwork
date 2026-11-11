#include "nn/device.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "gpu/backend.hpp"

namespace nn {

namespace {

struct State {
    std::mutex mutex;
    std::unique_ptr<gpu::Device> device;
    Backend backend = Backend::CPU;
    std::atomic<bool> active{false};
    std::atomic<std::size_t> threshold{std::size_t(1) << 26};  // ~400^3; tune per GPU
    std::atomic<bool> env_checked{false};
};

State& state() {
    static State s;
    return s;
}

std::unique_ptr<gpu::Device> create(Backend b, [[maybe_unused]] int index) {
    switch (b) {
#ifdef NN_HAS_OPENCL
        case Backend::OpenCL: return gpu::create_opencl(index);
#endif
#ifdef NN_HAS_VULKAN
        case Backend::Vulkan: return gpu::create_vulkan(index);
#endif
#ifdef NN_HAS_CUDA
        case Backend::CUDA: return gpu::create_cuda(index);
#endif
#ifdef NN_HAS_HIP
        case Backend::HIP: return gpu::create_hip(index);
#endif
#ifdef NN_HAS_SYCL
        case Backend::SYCL: return gpu::create_sycl(index);
#endif
        default: return nullptr;
    }
}

bool activate_locked(State& s, Backend b, int index) {
    if (b == Backend::CPU) {
        s.active = false;
        s.device.reset();
        s.backend = Backend::CPU;
        return true;
    }
    std::unique_ptr<gpu::Device> dev;
    try {
        dev = create(b, index);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nn: %s backend unavailable: %s\n", to_string(b), e.what());
        return false;
    }
    if (!dev) return false;
    s.active = false;
    s.device = std::move(dev);
    s.backend = b;
    s.active = true;
    return true;
}

// Apply NN_BACKEND / NN_DEVICE once, lazily.
void check_env_locked(State& s) {
    if (s.env_checked) return;
    s.env_checked = true;
    if (const char* t = std::getenv("NN_OFFLOAD_THRESHOLD"))
        s.threshold = static_cast<std::size_t>(std::strtoull(t, nullptr, 10));
    const char* env = std::getenv("NN_BACKEND");
    if (!env || !*env) return;
    const char* dev = std::getenv("NN_DEVICE");
    const int index = dev ? std::atoi(dev) : 0;
    try {
        const Backend b = backend_from_string(env);
        if (!activate_locked(s, b, index))
            std::fprintf(stderr, "nn: NN_BACKEND=%s requested but not available, using CPU\n", env);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nn: %s\n", e.what());
    }
}

}  // namespace

const char* to_string(Backend b) {
    switch (b) {
        case Backend::CPU: return "cpu";
        case Backend::OpenCL: return "opencl";
        case Backend::Vulkan: return "vulkan";
        case Backend::CUDA: return "cuda";
        case Backend::HIP: return "hip";
        case Backend::SYCL: return "sycl";
    }
    return "unknown";
}

Backend backend_from_string(const std::string& name) {
    std::string n(name);
    std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return std::tolower(c); });
    for (Backend b : {Backend::CPU, Backend::OpenCL, Backend::Vulkan, Backend::CUDA, Backend::HIP,
                      Backend::SYCL})
        if (n == to_string(b)) return b;
    if (n == "rocm") return Backend::HIP;
    if (n == "oneapi") return Backend::SYCL;
    throw std::invalid_argument("nn: unknown backend '" + name + "'");
}

std::vector<Backend> compiled_backends() {
    std::vector<Backend> v{Backend::CPU};
#ifdef NN_HAS_OPENCL
    v.push_back(Backend::OpenCL);
#endif
#ifdef NN_HAS_VULKAN
    v.push_back(Backend::Vulkan);
#endif
#ifdef NN_HAS_CUDA
    v.push_back(Backend::CUDA);
#endif
#ifdef NN_HAS_HIP
    v.push_back(Backend::HIP);
#endif
#ifdef NN_HAS_SYCL
    v.push_back(Backend::SYCL);
#endif
    return v;
}

bool set_backend(Backend b, int device) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.env_checked = true;  // explicit choice overrides the environment
    return activate_locked(s, b, device);
}

Backend current_backend() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    check_env_locked(s);
    return s.backend;
}

std::string device_name() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    check_env_locked(s);
    return s.device ? s.device->name() : std::string("CPU (") + build_info() + ")";
}

void set_offload_threshold(std::size_t flops) { state().threshold = flops; }
std::size_t offload_threshold() { return state().threshold; }

namespace gpu {

bool try_gemm(bool ta, bool tb, std::size_t M, std::size_t N, std::size_t K, real alpha,
              const real* A, std::size_t lda, const real* B, std::size_t ldb, real beta, real* C,
              std::size_t ldc) {
    State& s = state();
    if (!s.active.load(std::memory_order_relaxed)) {
        if (s.env_checked) return false;
        std::lock_guard<std::mutex> lock(s.mutex);
        check_env_locked(s);
        if (!s.active) return false;
    }
    if (static_cast<double>(M) * N * K < static_cast<double>(s.threshold.load())) return false;
    // GPU kernels use 32-bit indices
    constexpr std::size_t kMax = 0x7fffffff;
    if ((ta ? span(K, M, lda) : span(M, K, lda)) > kMax || (tb ? span(N, K, ldb) : span(K, N, ldb)) > kMax ||
        span(M, N, ldc) > kMax)
        return false;
    std::lock_guard<std::mutex> lock(s.mutex);  // one queue per device
    if (!s.device) return false;
    try {
        s.device->gemm(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nn: %s gemm failed (%s), falling back to CPU\n", to_string(s.backend),
                     e.what());
        s.active = false;
        s.device.reset();
        s.backend = Backend::CPU;
        return false;
    }
}

}  // namespace gpu
}  // namespace nn
