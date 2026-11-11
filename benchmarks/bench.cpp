// Micro benchmarks: GEMM throughput, activation throughput, training speed, spiking simulation.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include <nn/nn.hpp>

using clk = std::chrono::steady_clock;

template <class F>
double time_it(F&& f, int reps) {
    f();  // warm-up
    auto t0 = clk::now();
    for (int i = 0; i < reps; ++i) f();
    return std::chrono::duration<double>(clk::now() - t0).count() / reps;
}

static nn::Matrix rnd(std::size_t r, std::size_t c, nn::Rng& rng) {
    nn::Matrix m(r, c);
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = rng.uniform(-1, 1);
    return m;
}

int main(int argc, char** argv) {
    if (argc > 1) nn::set_num_threads(std::stoi(argv[1]));
    std::printf("%s, threads=%d\n\n", nn::build_info(), nn::num_threads());
    nn::Rng rng(1);

    std::printf("GEMM (GFLOP/s)\n");
    for (std::size_t n : {128, 256, 512, 1024}) {
        nn::Matrix A = rnd(n, n, rng), B = rnd(n, n, rng), C(n, n);
        std::printf("  %4zu: ", n);
        for (int v = 0; v < 3; ++v) {
            const bool ta = v == 2, tb = v == 1;
            double t = time_it([&] { nn::kernels::gemm(ta, tb, n, n, n, 1, A.data(), n, B.data(), n, 0, C.data(), n); },
                               n >= 1024 ? 3 : 10);
            std::printf("%s %7.1f   ", ta ? "TN" : (tb ? "NT" : "NN"), 2.0 * n * n * n / t * 1e-9);
        }
        std::printf("\n");
    }

    std::printf("\nActivations (Gelem/s, 4M elements)\n");
    {
        nn::Matrix z = rnd(1024, 4096, rng), y(1024, 4096);
        for (auto k : {nn::ActivationKind::ReLU, nn::ActivationKind::Sigmoid, nn::ActivationKind::Tanh,
                       nn::ActivationKind::GELU, nn::ActivationKind::Swish, nn::ActivationKind::Mish,
                       nn::ActivationKind::ELU, nn::ActivationKind::Softmax}) {
            double t = time_it([&] { nn::activation_forward(k, z.rows(), z.cols(), z.data(), y.data()); }, 5);
            std::printf("  %-10s %6.2f\n", nn::to_string(k), z.size() / t * 1e-9);
        }
    }

    std::printf("\nMLP training 784-512-256-10 (batch 128)\n");
    {
        const std::size_t N = 4096;
        nn::Matrix x = rnd(N, 784, rng), y(N, 10);
        for (std::size_t i = 0; i < N; ++i) y(i, rng.below(10)) = 1;
        nn::Network net(784, 1);
        net.add<nn::Dense>(512, nn::Activation::relu());
        net.add<nn::Dense>(256, nn::Activation::relu());
        net.add<nn::Dense>(10, nn::Activation::softmax());
        net.compile(nn::Loss::CategoricalCrossEntropy, std::make_unique<nn::Adam>(1e-3f));
        nn::FitOptions o;
        o.epochs = 1;
        o.batch_size = 128;
        o.verbose = 0;
        double t = time_it([&] { net.fit(x, y, o); }, 3);
        std::printf("  %.0f samples/s train, ", N / t);
        t = time_it([&] { net.predict(x, 512); }, 3);
        std::printf("%.0f samples/s inference\n", N / t);
    }

    std::printf("\nLSTM training (T=32, F=16, H=128, batch 64)\n");
    {
        const std::size_t N = 1024, T = 32, F = 16;
        nn::Matrix x = rnd(N, T * F, rng), y = rnd(N, 1, rng);
        nn::Network net(T * F, 1);
        net.add<nn::LSTM>(128, T);
        net.add<nn::Dense>(1);
        net.compile(nn::Loss::MeanSquaredError, std::make_unique<nn::Adam>(1e-3f));
        nn::FitOptions o;
        o.epochs = 1;
        o.batch_size = 64;
        o.verbose = 0;
        double t = time_it([&] { net.fit(x, y, o); }, 2);
        std::printf("  %.0f sequences/s\n", N / t);
    }

    std::printf("\nSpiking: 10k Izhikevich neurons (80/20 exc/inh), ~1M synapses, dt=0.5ms\n");
    {
        nn::spiking::Network sn(0.5f, 5.0f, 7);
        auto exc = sn.add_population(8000, nn::spiking::IzhikevichParams::regular_spiking());
        auto inh = sn.add_population(2000, nn::spiking::IzhikevichParams::fast_spiking());
        sn.connect(exc, exc, 0.01f, 0.5f);
        sn.connect(exc, inh, 0.01f, 0.5f);
        sn.connect(inh, exc, 0.01f, -1.0f);
        sn.connect(inh, inh, 0.01f, -1.0f);
        sn.add_poisson_input(exc, 2000, 1.5f);
        sn.add_poisson_input(inh, 2000, 1.5f);
        sn.finalize();
        auto t0 = clk::now();
        sn.run(2000);  // 1 s biological time
        double t = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("  synapses %zu, 1 s simulated in %.3f s (%.1fx realtime), exc %.1f Hz, inh %.1f Hz\n",
                    sn.synapse_count(), t, 1.0 / t, sn.firing_rate(exc), sn.firing_rate(inh));
    }
    return 0;
}
