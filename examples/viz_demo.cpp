// Records four networks for the 3D viewer (viz/viewer.html):
//   mlp.viz.json      Dense network learning a two-spiral classification
//   cnn.viz.json      Conv2D + MaxPool network recognising 12x12 shapes
//   lstm.viz.json     LSTM predicting the next value of a sine wave
//   spiking.viz.json  Izhikevich network (excitatory + inhibitory) firing
//
//   ./example_viz_demo [output_dir]
//   cd <output_dir> && python3 -m http.server   ->  http://localhost:8000/viewer.html
#include <cmath>
#include <cstdio>
#include <string>

#include <nn/nn.hpp>
#include <nn/viz.hpp>

namespace {

std::string out_dir = ".";

void save(const nn::viz::NetworkRecorder& rec, const char* name) {
    const std::string path = out_dir + "/" + name;
    std::printf("  -> %s (%zu snapshots) %s\n", path.c_str(), rec.snapshot_count(), rec.save(path) ? "" : "FAILED");
}

// Train `net`, taking a snapshot before training and after every `every` epochs.
void train_recorded(nn::Network& net, nn::viz::NetworkRecorder& rec, const nn::Matrix& x, const nn::Matrix& y,
                    std::size_t epochs, std::size_t every, bool classification) {
    auto acc = [&] { return classification ? net.evaluate_accuracy(x, y) : std::nanf(""); };
    rec.snapshot(x, &y, "untrained", net.evaluate_loss(x, y), acc());
    nn::FitOptions o;
    o.epochs = epochs;
    o.batch_size = 32;
    o.verbose = 0;
    o.on_epoch_end = [&](std::size_t e, float loss, float) {
        if ((e + 1) % every == 0) rec.snapshot(x, &y, "epoch " + std::to_string(e + 1), loss, acc());
        return true;
    };
    net.fit(x, y, o);
}

void mlp() {
    std::printf("MLP on two spirals\n");
    nn::Rng rng(3);
    const std::size_t N = 600;
    nn::Matrix x(N, 2), y(N, 2);
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t c = i % 2;
        const float t = 0.25f + 3.2f * rng.uniform();
        const float a = t * 2.2f + static_cast<float>(c) * 3.14159f;
        x(i, 0) = t * std::cos(a) / 3.5f + 0.03f * rng.normal();
        x(i, 1) = t * std::sin(a) / 3.5f + 0.03f * rng.normal();
        y(i, c) = 1;
    }
    nn::Network net(2, 7);
    net.add<nn::Dense>(32, nn::Activation::tanh());
    net.add<nn::Dense>(32, nn::Activation::tanh());
    net.add<nn::Dense>(16, nn::Activation::tanh());
    net.add<nn::Dense>(2, nn::Activation::softmax());
    net.compile(nn::Loss::CategoricalCrossEntropy, std::make_unique<nn::Adam>(3e-3f));
    nn::viz::NetworkRecorder rec(net);
    train_recorded(net, rec, x, y, 300, 6, true);
    std::printf("  accuracy %.1f%%\n", 100 * net.evaluate_accuracy(x, y));
    save(rec, "mlp.viz.json");
}

void cnn() {
    std::printf("CNN on 12x12 shapes (square / cross / circle)\n");
    nn::Rng rng(5);
    const std::size_t S = 12, N = 900;
    nn::Matrix x(N, S * S), y(N, 3);
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t c = i % 3;
        const float cx = 4.5f + 3 * rng.uniform(), cy = 4.5f + 3 * rng.uniform(), r = 2.2f + 1.3f * rng.uniform();
        for (std::size_t py = 0; py < S; ++py)
            for (std::size_t px = 0; px < S; ++px) {
                const float dx = px - cx, dy = py - cy;
                bool on = false;
                if (c == 0) on = std::fabs(std::max(std::fabs(dx), std::fabs(dy)) - r) < 0.6f;
                if (c == 1) on = (std::fabs(dx) < 0.7f || std::fabs(dy) < 0.7f) && std::max(std::fabs(dx), std::fabs(dy)) < r + 0.5f;
                if (c == 2) on = std::fabs(std::sqrt(dx * dx + dy * dy) - r) < 0.6f;
                x(i, py * S + px) = (on ? 1.0f : 0.0f) + 0.1f * rng.uniform();
            }
        y(i, c) = 1;
    }
    nn::Network net(S * S, 11);
    net.add<nn::Conv2D>(nn::Shape3{1, S, S}, 6, 3, 1, 1, nn::Activation::relu());
    net.add<nn::MaxPool2D>(nn::Shape3{6, S, S}, 2);
    net.add<nn::Conv2D>(nn::Shape3{6, 6, 6}, 8, 3, 1, 0, nn::Activation::relu());
    net.add<nn::Dense>(16, nn::Activation::relu());
    net.add<nn::Dense>(3, nn::Activation::softmax());
    net.compile(nn::Loss::CategoricalCrossEntropy, std::make_unique<nn::Adam>(3e-3f));
    nn::viz::NetworkRecorder rec(net, {9, 4096, 600});
    train_recorded(net, rec, x, y, 30, 1, true);
    std::printf("  accuracy %.1f%%\n", 100 * net.evaluate_accuracy(x, y));
    save(rec, "cnn.viz.json");
}

void lstm() {
    std::printf("LSTM forecasting a sine wave\n");
    const std::size_t T = 16, N = 800;
    nn::Rng rng(2);
    nn::Matrix x(N, T), y(N, 1);
    for (std::size_t i = 0; i < N; ++i) {
        const float phase = 6.283f * rng.uniform(), freq = 0.25f + 0.2f * rng.uniform();
        for (std::size_t t = 0; t < T; ++t) x(i, t) = std::sin(phase + freq * t);
        y(i, 0) = std::sin(phase + freq * T);
    }
    nn::Network net(T, 4);
    net.add<nn::LSTM>(12, T, true);
    net.add<nn::LSTM>(12, T);
    net.add<nn::Dense>(1);
    net.compile(nn::Loss::MeanSquaredError, std::make_unique<nn::Adam>(5e-3f));
    nn::viz::NetworkRecorder rec(net);
    train_recorded(net, rec, x, y, 40, 2, false);
    std::printf("  MSE %.5f\n", net.evaluate_loss(x, y));
    save(rec, "lstm.viz.json");
}

void spiking() {
    std::printf("Spiking network (Izhikevich, 1000 neurons, 1 s)\n");
    namespace sp = nn::spiking;
    sp::Network net(0.5f, 5.0f, 42);
    auto exc = net.add_population(800, sp::IzhikevichParams::regular_spiking(), "excitatory");
    auto inh = net.add_population(200, sp::IzhikevichParams::fast_spiking(), "inhibitory");
    net.connect(exc, exc, 0.1f, 0.5f, 0.1f);
    net.connect(exc, inh, 0.1f, 0.5f, 0.1f);
    net.connect(inh, exc, 0.1f, -1.0f, 0.2f);
    net.connect(inh, inh, 0.1f, -1.0f, 0.2f);
    net.add_poisson_input(exc, 2000, 1.5f);
    net.add_poisson_input(inh, 2000, 1.5f);
    net.finalize();
    nn::viz::SpikeRecorder rec(net);
    for (int s = 0; s < 2000; ++s) {
        net.step();
        rec.record_step();
    }
    std::printf("  exc %.1f Hz, inh %.1f Hz\n", net.firing_rate(exc), net.firing_rate(inh));
    const std::string path = out_dir + "/spiking.viz.json";
    std::printf("  -> %s %s\n", path.c_str(), rec.save(path) ? "" : "FAILED");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) out_dir = argv[1];
    mlp();
    cnn();
    lstm();
    spiking();
    std::printf("\nOpen viz/viewer.html and load one of the files (or serve the folder: python3 -m http.server).\n");
    return 0;
}
