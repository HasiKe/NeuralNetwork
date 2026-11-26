// Automatic learning rate vs. fixed learning rates.
//
//   nn_auto_lr_bench [seeds_per_setting] [path/to/seeds_dataset.csv] [first_seed]
//
// Every method gets the same data split, network, initial weights and epoch budget,
// and is scored on the test set with the weights of its best validation epoch.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nn/nn.hpp>

namespace {

struct Data {
    nn::Matrix x, y, xv, yv, xt, yt;
};

struct Problem {
    const char* name;
    bool classification;
    std::size_t epochs, batch;
    Data (*make)(std::uint64_t seed, const std::string& csv);
    void (*build)(nn::Network&);
};

void split3(const nn::Matrix& x, const nn::Matrix& y, std::size_t n_train, std::size_t n_val, nn::Rng& rng,
            Data& d) {
    std::vector<std::size_t> idx(x.rows());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    rng.shuffle(idx.begin(), idx.end());
    d.x.gather_rows(x, idx.data(), n_train);
    d.y.gather_rows(y, idx.data(), n_train);
    d.xv.gather_rows(x, idx.data() + n_train, n_val);
    d.yv.gather_rows(y, idx.data() + n_train, n_val);
    const std::size_t rest = x.rows() - n_train - n_val;
    d.xt.gather_rows(x, idx.data() + n_train + n_val, rest);
    d.yt.gather_rows(y, idx.data() + n_train + n_val, rest);
}

void standardise(Data& d) {
    nn::data::StandardScaler s;
    s.fit_transform(d.x);
    s.transform(d.xv);
    s.transform(d.xt);
}

Data make_seeds(std::uint64_t seed, const std::string& csv) {
    nn::Matrix raw = nn::data::load_csv(csv);
    auto [x, labels] = nn::data::split_xy(raw);
    const std::size_t k = nn::data::encode_labels(labels);
    nn::Rng rng(1000 + seed);
    Data d;
    split3(x, nn::data::one_hot(labels, k), 126, 42, rng, d);
    standardise(d);
    return d;
}

Data make_spirals(std::uint64_t seed, const std::string&) {
    const std::size_t n = 1800, k = 3;
    nn::Rng rng(2000 + seed);
    nn::Matrix x(n, 2), y(n, k);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t c = i % k;
        const float r = rng.uniform(0.05f, 1.0f);
        const float a = 4.0f * r + 2.0944f * static_cast<float>(c) + rng.normal() * 0.18f;
        x(i, 0) = r * std::cos(a);
        x(i, 1) = r * std::sin(a);
        y(i, c) = 1;
    }
    Data d;
    split3(x, y, 900, 300, rng, d);
    standardise(d);
    return d;
}

// Smooth 2-D surface with noise. Not used while choosing the AutoLR defaults.
Data make_surface(std::uint64_t seed, const std::string&) {
    const std::size_t n = 1500;
    nn::Rng rng(4000 + seed);
    nn::Matrix x(n, 2), y(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        const float a = rng.uniform(-1, 1), b = rng.uniform(-1, 1);
        x(i, 0) = a;
        x(i, 1) = b;
        y(i, 0) = std::sin(3 * a) * std::cos(2 * b) + 0.5f * a * b + 0.4f * std::exp(-8 * (a - 0.3f) * (a - 0.3f)) +
                  0.1f * rng.normal();
    }
    Data d;
    split3(x, y, 600, 200, rng, d);
    standardise(d);
    return d;
}

Data make_friedman(std::uint64_t seed, const std::string&) {
    const std::size_t n = 2000;
    nn::Rng rng(3000 + seed);
    nn::Matrix x(n, 10), y(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < 10; ++j) x(i, j) = rng.uniform();
        const float v = 10 * std::sin(3.14159265f * x(i, 0) * x(i, 1)) +
                        20 * (x(i, 2) - 0.5f) * (x(i, 2) - 0.5f) + 10 * x(i, 3) + 5 * x(i, 4) + rng.normal();
        y(i, 0) = (v - 14.4f) / 5.0f;
    }
    Data d;
    split3(x, y, 800, 300, rng, d);
    standardise(d);
    return d;
}

const Problem kProblems[] = {
    {"seeds (echte Daten, 3 Klassen)", true, 150, 16, make_seeds,
     [](nn::Network& n) {
         n.add<nn::Dense>(32, nn::Activation::gelu());
         n.add<nn::Dense>(16, nn::Activation::gelu());
         n.add<nn::Dense>(3, nn::Activation::softmax());
     }},
    {"spirals (synthetisch, 3 Klassen)", true, 100, 32, make_spirals,
     [](nn::Network& n) {
         n.add<nn::Dense>(48, nn::Activation::tanh());
         n.add<nn::Dense>(48, nn::Activation::tanh());
         n.add<nn::Dense>(3, nn::Activation::softmax());
     }},
    {"friedman1 (synthetisch, Regression)", false, 120, 32, make_friedman,
     [](nn::Network& n) {
         n.add<nn::Dense>(48, nn::Activation::gelu());
         n.add<nn::Dense>(48, nn::Activation::gelu());
         n.add<nn::Dense>(1);
     }},
    {"surface (synthetisch, Regression, nicht zur Abstimmung benutzt)", false, 200, 32, make_surface,
     [](nn::Network& n) {
         n.add<nn::Dense>(24, nn::Activation::tanh());
         n.add<nn::Dense>(24, nn::Activation::tanh());
         n.add<nn::Dense>(1);
     }},
};

struct Score {
    double loss = 0, acc = 0, secs = 0, lr = 0, epochs = 0;
};

enum class Mode { Fixed, Plateau, Auto };

Score run(const Problem& p, const Data& d, std::uint64_t seed, bool sgd, Mode mode, float lr) {
    nn::Network net(d.x.cols(), 100 + seed);
    p.build(net);
    std::unique_ptr<nn::Optimizer> o;
    if (sgd)
        o = std::make_unique<nn::SGD>(lr, 0.9f);
    else
        o = std::make_unique<nn::Adam>(lr);
    net.compile(p.classification ? nn::Loss::CategoricalCrossEntropy : nn::Loss::MeanSquaredError, std::move(o));
    nn::FitOptions f;
    f.epochs = p.epochs;
    f.batch_size = p.batch;
    f.verbose = 0;

    std::string best_model;
    float best = std::numeric_limits<float>::infinity();
    if (mode == Mode::Auto) {
        f.auto_lr = nn::AutoLR::on();
    } else {
        if (mode == Mode::Plateau) f.schedule = nn::LRSchedule::reduce_on_plateau(7, 0.5f);
        f.on_epoch_end = [&](std::size_t, float, float vloss) {
            if (vloss < best) {
                best = vloss;
                std::ostringstream s;
                net.save(s);
                best_model = s.str();
            }
            return true;
        };
    }
    const auto t0 = std::chrono::steady_clock::now();
    const nn::History h = net.fit(d.x, d.y, d.xv, d.yv, f);
    Score s;
    s.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    s.epochs = static_cast<double>(h.loss.size());
    s.lr = mode == Mode::Auto ? h.auto_lr.start_lr : lr;
    nn::Network* eval = &net;
    nn::Network loaded;
    if (mode != Mode::Auto && !best_model.empty()) {
        std::istringstream is(best_model);
        loaded = nn::Network::load(is);
        eval = &loaded;
    }
    s.loss = eval->evaluate_loss(d.xt, d.yt);
    if (p.classification) s.acc = eval->evaluate_accuracy(d.xt, d.yt);
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t seeds = argc > 1 ? static_cast<std::size_t>(std::stoul(argv[1])) : 5;
    const std::string csv = argc > 2 ? argv[2] : "data/seeds_dataset.csv";
    const std::size_t first = argc > 3 ? static_cast<std::size_t>(std::stoul(argv[3])) : 100;
    const float lrs[] = {1e-4f, 3e-4f, 1e-3f, 3e-3f, 1e-2f, 3e-2f, 1e-1f, 3e-1f};
    std::printf("%s\n%zu Seeds je Einstellung ab Seed %zu, Testwerte als Mittel ueber die Seeds (+- Standardabweichung)\n",
                nn::build_info(), seeds, first);

    for (const Problem& p : kProblems) {
        std::vector<Data> data;
        for (std::size_t s = 0; s < seeds; ++s) data.push_back(p.make(first + s, csv));
        for (const bool sgd : {false, true}) {
            std::printf("\n== %s, %s, max. %zu Epochen\n", p.name, sgd ? "SGD(momentum 0.9)" : "Adam", p.epochs);
            std::printf("%-26s %12s %10s %10s %8s %8s\n", "Methode", "Testloss", "+-",
                        p.classification ? "Acc %" : "", "Epochen", "Sek.");
            std::vector<std::pair<double, std::string>> fixed;
            double auto_loss = 0;
            auto row = [&](const std::string& name, Mode mode, float lr) {
                double l = 0, l2 = 0, acc = 0, sec = 0, ep = 0, lo = 1e30, hi = 0;
                for (std::size_t s = 0; s < seeds; ++s) {
                    const Score r = run(p, data[s], first + s, sgd, mode, lr);
                    l += r.loss;
                    l2 += r.loss * r.loss;
                    acc += r.acc;
                    sec += r.secs;
                    ep += r.epochs;
                    lo = std::min(lo, r.lr);
                    hi = std::max(hi, r.lr);
                }
                const double n = static_cast<double>(seeds);
                const double mean = l / n, sd = std::sqrt(std::max(0.0, l2 / n - mean * mean));
                std::printf("%-26s %12.4f %10.4f", name.c_str(), mean, sd);
                if (mode == Mode::Auto)
                    auto_loss = mean;
                else
                    fixed.emplace_back(std::isfinite(mean) ? mean : 1e30, name);
                if (p.classification)
                    std::printf(" %10.1f", 100 * acc / n);
                else
                    std::printf(" %10s", "");
                std::printf(" %8.0f %8.2f", ep / n, sec / n);
                if (mode == Mode::Auto) std::printf("   Start-LR %.3g .. %.3g", lo, hi);
                std::printf("\n");
                std::fflush(stdout);
            };
            for (float lr : lrs) {
                char name[64];
                std::snprintf(name, sizeof name, "fest %g", static_cast<double>(lr));
                row(name, Mode::Fixed, lr);
            }
            for (float lr : lrs) {
                char name[64];
                std::snprintf(name, sizeof name, "fest %g + plateau", static_cast<double>(lr));
                row(name, Mode::Plateau, lr);
            }
            row("auto", Mode::Auto, 1e-3f);
            std::sort(fixed.begin(), fixed.end());
            std::size_t better = 0;
            for (const auto& f : fixed) better += f.first < auto_loss;
            // "best fixed" is picked with hindsight on the test set, which favours the fixed rates
            std::printf("-> auto: Platz %zu von %zu; bestes fest (im Nachhinein gewaehlt): %s = %.4f; "
                        "mittleres fest (Median): %.4f\n",
                        better + 1, fixed.size() + 1, fixed.front().second.c_str(), fixed.front().first,
                        fixed[fixed.size() / 2].first);
        }
    }
    return 0;
}
