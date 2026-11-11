// Time-series forecasting with recurrent neurons: predict the next value of a
// noisy sum of sines from the previous T values. Compares SimpleRNN, GRU, LSTM.
#include <chrono>
#include <cmath>
#include <cstdio>

#include <nn/nn.hpp>

int main() {
    const std::size_t T = 24, N = 3000;
    nn::Rng rng(1);
    std::vector<float> series(N + T + 1);
    for (std::size_t i = 0; i < series.size(); ++i)
        series[i] = 0.6f * std::sin(0.11f * i) + 0.3f * std::sin(0.037f * i + 1) + 0.05f * rng.normal();

    // rows = windows of T consecutive values (time-major, 1 feature per step)
    nn::Matrix x(N, T), y(N, 1);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t t = 0; t < T; ++t) x(i, t) = series[i + t];
        y(i, 0) = series[i + T];
    }
    auto split = nn::data::train_test_split(x, y, 0.2f, rng);

    for (int which = 0; which < 3; ++which) {
        nn::Network net(T, 5);
        const char* name = which == 0 ? "SimpleRNN" : which == 1 ? "GRU" : "LSTM";
        if (which == 0) net.add<nn::SimpleRNN>(32, T, true);
        if (which == 1) net.add<nn::GRU>(32, T, true);
        if (which == 2) net.add<nn::LSTM>(32, T, true);
        // second recurrent layer consumes the full sequence of the first one
        if (which == 0) net.add<nn::SimpleRNN>(32, T);
        if (which == 1) net.add<nn::GRU>(32, T);
        if (which == 2) net.add<nn::LSTM>(32, T);
        net.add<nn::Dense>(1);
        net.compile(nn::Loss::MeanSquaredError, std::make_unique<nn::Adam>(3e-3f));
        nn::FitOptions o;
        o.epochs = 15;
        o.batch_size = 64;
        o.verbose = 0;
        auto t0 = std::chrono::steady_clock::now();
        net.fit(split.x_train, split.y_train, o);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%-9s params %6zu  test MSE %.5f  (%.2fs)\n", name, net.parameter_count(),
                    net.evaluate_loss(split.x_test, split.y_test), s);
    }
    return 0;
}
