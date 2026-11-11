// Self-contained test suite (no external framework).
#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <nn/nn.hpp>

namespace {

int g_failed = 0, g_checks = 0;
std::string g_current;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_failed;                                                                   \
            std::printf("  FAIL [%s] %s:%d: %s\n", g_current.c_str(), __FILE__, __LINE__, #cond); \
        }                                                                                 \
    } while (0)

#define CHECK_MSG(cond, ...)                                                              \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(cond)) {                                                                    \
            ++g_failed;                                                                   \
            std::printf("  FAIL [%s] %s:%d: ", g_current.c_str(), __FILE__, __LINE__);    \
            std::printf(__VA_ARGS__);                                                     \
            std::printf("\n");                                                            \
        }                                                                                 \
    } while (0)

struct TestCase {
    const char* name;
    std::function<void()> fn;
};
std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
struct Reg {
    Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};
#define TEST(name)                       \
    void name();                         \
    Reg reg_##name(#name, name);         \
    void name()

nn::Matrix random_matrix(std::size_t r, std::size_t c, nn::Rng& rng, float lo = -1, float hi = 1) {
    nn::Matrix m(r, c);
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = rng.uniform(lo, hi);
    return m;
}

// ------------------------------------------------------------------ gradient checking

// L = sum(out .* R). Checks dL/dparams and dL/din against central differences.
// `tolerated` = fraction of entries allowed to fail (non-smooth activations).
void grad_check(nn::Layer& layer, std::size_t in_dim, std::size_t batch, const char* label,
                double tolerated = 0.0, float lo = -1, float hi = 1) {
    nn::Rng rng(123);
    nn::Matrix x = random_matrix(batch, in_dim, rng, lo, hi);
    nn::Matrix out, out2, gin;
    layer.forward(x, out, true);
    nn::Matrix R = random_matrix(out.rows(), out.cols(), rng);
    layer.backward(x, out, R, &gin);

    auto lossf = [&]() {
        layer.forward(x, out2, true);
        double s = 0;
        for (std::size_t i = 0; i < out2.size(); ++i) s += static_cast<double>(out2[i]) * R[i];
        return s;
    };
    std::size_t total = 0, bad = 0;
    double worst = 0;
    auto compare = [&](float analytic, double numeric) {
        ++total;
        const double err = std::fabs(analytic - numeric);
        const double scale = std::max(1.0, std::fabs(numeric) + std::fabs(analytic));
        const double rel = err / scale;
        worst = std::max(worst, rel);
        if (rel > 2e-2) ++bad;
    };
    const float eps = 1e-2f;
    std::vector<nn::ParamRef> params;
    layer.parameters(params);
    // copy analytic grads (forward/backward calls below would not touch them, but be safe)
    std::vector<std::vector<float>> grads;
    for (auto& p : params) grads.emplace_back(p.grad, p.grad + p.size);
    for (std::size_t k = 0; k < params.size(); ++k) {
        auto& p = params[k];
        const std::size_t stride = std::max<std::size_t>(1, p.size / 40);
        for (std::size_t i = 0; i < p.size; i += stride) {
            const float orig = p.value[i];
            p.value[i] = orig + eps;
            const double lp = lossf();
            p.value[i] = orig - eps;
            const double lm = lossf();
            p.value[i] = orig;
            compare(grads[k][i], (lp - lm) / (2 * eps));
        }
    }
    const std::size_t stride = std::max<std::size_t>(1, x.size() / 60);
    for (std::size_t i = 0; i < x.size(); i += stride) {
        const float orig = x[i];
        x[i] = orig + eps;
        const double lp = lossf();
        x[i] = orig - eps;
        const double lm = lossf();
        x[i] = orig;
        compare(gin[i], (lp - lm) / (2 * eps));
    }
    const double frac = total ? static_cast<double>(bad) / total : 0;
    CHECK_MSG(frac <= tolerated, "%s: %zu/%zu gradient entries off (worst rel err %.3g)", label, bad,
              total, worst);
}

// ------------------------------------------------------------------ tests

TEST(gemm_matches_naive) {
    nn::Rng rng(1);
    const std::size_t shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {17, 33, 9}, {64, 300, 257}, {129, 70, 513}, {2, 600, 40}};
    for (auto& s : shapes) {
        const std::size_t M = s[0], N = s[1], K = s[2];
        for (int ta = 0; ta < 2; ++ta)
            for (int tb = 0; tb < 2; ++tb) {
                nn::Matrix A = ta ? random_matrix(K, M, rng) : random_matrix(M, K, rng);
                nn::Matrix B = tb ? random_matrix(N, K, rng) : random_matrix(K, N, rng);
                nn::Matrix C = random_matrix(M, N, rng), ref = C;
                const float alpha = 0.7f, beta = 0.3f;
                for (std::size_t i = 0; i < M; ++i)
                    for (std::size_t j = 0; j < N; ++j) {
                        double acc = 0;
                        for (std::size_t k = 0; k < K; ++k)
                            acc += (ta ? A(k, i) : A(i, k)) * (tb ? B(j, k) : B(k, j));
                        ref(i, j) = static_cast<float>(alpha * acc + beta * ref(i, j));
                    }
                nn::kernels::gemm(ta, tb, M, N, K, alpha, A.data(), A.cols(), B.data(), B.cols(), beta,
                                  C.data(), N);
                double err = 0;
                for (std::size_t i = 0; i < C.size(); ++i) err = std::max(err, (double)std::fabs(C[i] - ref[i]));
                CHECK_MSG(err < 1e-3 * std::sqrt((double)K), "gemm %zux%zux%zu ta=%d tb=%d err=%g", M, N, K, ta, tb, err);
            }
    }
}

TEST(gpu_backends_match_cpu) {
    const std::size_t saved_threshold = nn::offload_threshold();
    const nn::Backend saved_backend = nn::current_backend();
    CHECK(nn::backend_from_string("OpenCL") == nn::Backend::OpenCL);
    for (nn::Backend b : nn::compiled_backends()) {
        if (b == nn::Backend::CPU) continue;
        if (!nn::set_backend(b)) {
            std::printf("  (%s compiled in, but no device available - skipped)\n", nn::to_string(b));
            continue;
        }
        std::printf("  testing %s\n", nn::device_name().c_str());
        nn::set_offload_threshold(0);  // force every product onto the GPU
        nn::Rng rng(21);
        const std::size_t shapes[][3] = {{1, 1, 1}, {5, 3, 7}, {33, 70, 17}, {130, 65, 200}};
        for (auto& s : shapes)
            for (int ta = 0; ta < 2; ++ta)
                for (int tb = 0; tb < 2; ++tb)
                    for (float beta : {0.0f, 0.5f}) {
                        const std::size_t M = s[0], N = s[1], K = s[2], pad = 3;  // ld > width
                        const std::size_t lda = (ta ? M : K) + pad, ldb = (tb ? K : N) + pad, ldc = N + pad;
                        nn::Matrix A = random_matrix(ta ? K : M, lda, rng), B = random_matrix(tb ? N : K, ldb, rng);
                        nn::Matrix C = random_matrix(M, ldc, rng), ref = C;
                        for (std::size_t i = 0; i < M; ++i)
                            for (std::size_t j = 0; j < N; ++j) {
                                double acc = 0;
                                for (std::size_t k = 0; k < K; ++k)
                                    acc += (ta ? A(k, i) : A(i, k)) * (tb ? B(j, k) : B(k, j));
                                ref(i, j) = static_cast<float>(1.5 * acc + beta * ref(i, j));
                            }
                        nn::kernels::gemm(ta, tb, M, N, K, 1.5f, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
                        double err = 0;
                        for (std::size_t i = 0; i < M; ++i)
                            for (std::size_t j = 0; j < ldc; ++j)
                                err = std::max(err, (double)std::fabs(C(i, j) - ref(i, j)));  // padding untouched too
                        CHECK_MSG(err < 2e-3 * std::sqrt((double)K), "%s gemm %zux%zux%zu ta=%d tb=%d beta=%g err=%g",
                                  nn::to_string(b), M, N, K, ta, tb, beta, err);
                    }
        // full training run on the GPU gives the same accuracy as the CPU
        nn::Matrix x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        nn::Matrix y = {{0}, {1}, {1}, {0}};
        nn::Network net(2, 4);
        net.add<nn::Dense>(8, nn::Activation::tanh());
        net.add<nn::Dense>(1, nn::Activation::sigmoid());
        net.compile(nn::Loss::BinaryCrossEntropy, std::make_unique<nn::Adam>(0.05f));
        nn::FitOptions o;
        o.epochs = 300;
        o.batch_size = 4;
        o.verbose = 0;
        net.fit(x, y, o);
        CHECK_MSG(net.evaluate_accuracy(x, y) == 1.0f, "%s: XOR not learned", nn::to_string(b));
        CHECK(nn::current_backend() == b);  // no silent fallback happened
    }
    nn::set_backend(saved_backend);
    nn::set_offload_threshold(saved_threshold);
    CHECK(nn::current_backend() == saved_backend);
}

TEST(fastmath_accuracy) {
    double e_exp = 0, e_log = 0, e_tanh = 0, e_sig = 0;
    for (float x = -20; x <= 20; x += 0.0137f) {
        e_exp = std::max(e_exp, (double)(std::fabs(nn::fastmath::exp(x) - std::exp(x)) / std::exp(x)));
        e_tanh = std::max(e_tanh, (double)std::fabs(nn::fastmath::tanh(x) - std::tanh(x)));
        e_sig = std::max(e_sig, (double)std::fabs(nn::fastmath::sigmoid(x) - 1 / (1 + std::exp(-x))));
    }
    for (float x = 1e-6f; x < 1e6f; x *= 1.37f)
        e_log = std::max(e_log, (double)std::fabs(nn::fastmath::log(x) - std::log(x)));
    CHECK_MSG(e_exp < 1e-6, "exp rel err %g", e_exp);
    CHECK_MSG(e_log < 1e-5, "log abs err %g", e_log);
    CHECK_MSG(e_tanh < 1e-6, "tanh abs err %g", e_tanh);
    CHECK_MSG(e_sig < 1e-6, "sigmoid abs err %g", e_sig);
}

TEST(activation_gradients) {
    using K = nn::ActivationKind;
    for (int k = 0; k < static_cast<int>(K::Count_); ++k) {
        const K kind = static_cast<K>(k);
        if (kind == K::Step) continue;  // straight-through estimator is not the true derivative
        nn::Rng rng(7);
        nn::Dense d(6, nn::Activation(kind));
        d.build(5, rng);
        const bool kinky = kind == K::ReLU || kind == K::LeakyReLU || kind == K::HardSigmoid ||
                           kind == K::HardTanh || kind == K::HardSwish || kind == K::ReLU6 ||
                           kind == K::ELU || kind == K::SELU;
        grad_check(d, 5, 4, nn::to_string(kind), kinky ? 0.05 : 0.0);
    }
}

TEST(activation_values) {
    using nn::Activation;
    auto f = [](Activation a, float x) {
        float y;
        nn::activation_forward(a, 1, 1, &x, &y);
        return y;
    };
    CHECK(std::fabs(f(Activation::relu(), -2) - 0) < 1e-7);
    CHECK(std::fabs(f(Activation::leaky_relu(0.1f), -2) + 0.2f) < 1e-6);
    CHECK(std::fabs(f(Activation::elu(), -1) - (std::exp(-1.0f) - 1)) < 1e-6);
    CHECK(std::fabs(f(Activation::gelu(), 1) - 0.841192f) < 1e-4);
    CHECK(std::fabs(f(Activation::swish(), 1) - 0.731059f) < 1e-5);
    CHECK(std::fabs(f(Activation::mish(), 1) - 0.865098f) < 1e-5);
    CHECK(std::fabs(f(Activation::softplus(), 0) - std::log(2.0f)) < 1e-6);
    CHECK(std::fabs(f(Activation::softsign(), 1) - 0.5f) < 1e-7);
    CHECK(std::fabs(f(Activation::hard_swish(), 1) - 2.0f / 3.0f) < 1e-6);
    CHECK(f(Activation::step(), 0.1f) == 1.0f && f(Activation::step(), -0.1f) == 0.0f);
    float z[3] = {1, 2, 3}, y[3];
    nn::activation_forward(Activation::softmax(), 1, 3, z, y);
    CHECK(std::fabs(y[0] + y[1] + y[2] - 1) < 1e-6);
    CHECK(std::fabs(y[2] - 0.665241f) < 1e-5);
}

TEST(layer_gradients) {
    nn::Rng rng(3);
    {
        nn::RBF l(5, 0.8f);
        l.build(3, rng);
        grad_check(l, 3, 4, "RBF");
    }
    {
        nn::LayerNorm l;
        l.build(7, rng);
        grad_check(l, 7, 3, "LayerNorm");
    }
    for (bool seq : {false, true}) {
        nn::SimpleRNN r(4, 5, seq);
        r.build(15, rng);
        grad_check(r, 15, 3, seq ? "SimpleRNN(seq)" : "SimpleRNN");
        nn::LSTM l(4, 5, seq);
        l.build(15, rng);
        grad_check(l, 15, 3, seq ? "LSTM(seq)" : "LSTM");
        nn::GRU g(4, 5, seq);
        g.build(15, rng);
        grad_check(g, 15, 3, seq ? "GRU(seq)" : "GRU");
    }
    {
        nn::Conv2D c({2, 6, 5}, 3, 3, 1, 1, nn::Activation::tanh());
        c.build(60, rng);
        grad_check(c, 60, 2, "Conv2D(pad)");
        nn::Conv2D c2({1, 7, 7}, 2, 3, 2, 0, nn::Activation::linear());
        c2.build(49, rng);
        grad_check(c2, 49, 3, "Conv2D(stride)");
    }
    {
        nn::MaxPool2D p({2, 4, 6}, 2);
        p.build(48, rng);
        grad_check(p, 48, 2, "MaxPool2D", 0.02);
    }
}

TEST(fused_output_gradients) {
    // Whole-network check incl. loss: softmax+CE and sigmoid+BCE use fused gradients.
    struct Case { nn::Loss loss; nn::Activation act; std::size_t out; };
    for (Case c : {Case{nn::Loss::CategoricalCrossEntropy, nn::Activation::softmax(), 4},
                   Case{nn::Loss::BinaryCrossEntropy, nn::Activation::sigmoid(), 3},
                   Case{nn::Loss::MeanSquaredError, nn::Activation::tanh(), 2},
                   Case{nn::Loss::Huber, nn::Activation::linear(), 2}}) {
        nn::Network net(5, 9);
        net.add<nn::Dense>(7, nn::Activation::tanh());
        net.add<nn::Dense>(c.out, c.act);
        net.compile(c.loss, std::make_unique<nn::SGD>(0.0f));
        nn::Rng rng(5);
        nn::Matrix x = random_matrix(6, 5, rng);
        nn::Matrix y(6, c.out);
        for (std::size_t r = 0; r < 6; ++r) {
            if (c.loss == nn::Loss::CategoricalCrossEntropy) y(r, rng.below(c.out)) = 1;
            else for (std::size_t j = 0; j < c.out; ++j) y(r, j) = rng.uniform(c.loss == nn::Loss::BinaryCrossEntropy ? 0.f : -1.f, 1.f);
        }
        net.train_batch(x, y);  // lr 0: computes gradients without changing weights
        std::vector<nn::ParamRef> ps;
        for (std::size_t i = 0; i < net.size(); ++i) net.layer(i).parameters(ps);
        std::size_t bad = 0, total = 0;
        for (auto& p : ps)
            for (std::size_t i = 0; i < p.size; i += 3) {
                const float o = p.value[i], eps = 5e-3f;
                p.value[i] = o + eps;
                const double lp = net.evaluate_loss(x, y);
                p.value[i] = o - eps;
                const double lm = net.evaluate_loss(x, y);
                p.value[i] = o;
                const double num = (lp - lm) / (2 * eps);
                ++total;
                if (std::fabs(num - p.grad[i]) > 2e-2 * std::max(1.0, std::fabs(num))) ++bad;
            }
        CHECK_MSG(bad == 0, "%s: %zu/%zu off", nn::to_string(c.loss), bad, total);
    }
}

TEST(dropout_behaviour) {
    nn::Rng rng(1);
    nn::Dropout d(0.25f);
    d.build(1000, rng);
    nn::Matrix x(4, 1000, 1.0f), y, g;
    d.forward(x, y, false);
    CHECK(y[17] == 1.0f);
    d.forward(x, y, true);
    std::size_t zeros = 0;
    double sum = 0;
    for (std::size_t i = 0; i < y.size(); ++i) { zeros += y[i] == 0; sum += y[i]; }
    CHECK(zeros > 800 && zeros < 1200);
    CHECK(std::fabs(sum / y.size() - 1.0) < 0.05);
    nn::Matrix go(4, 1000, 1.0f);
    d.backward(x, y, go, &g);
    for (std::size_t i = 0; i < y.size(); ++i) CHECK(g[i] == y[i]);
}

TEST(save_load_roundtrip) {
    nn::Network net(2 * 6 * 6, 11);
    net.add<nn::Conv2D>(nn::Shape3{2, 6, 6}, 3, 3, 1, 1, nn::Activation::relu());
    net.add<nn::MaxPool2D>(nn::Shape3{3, 6, 6}, 2);
    net.add<nn::LayerNorm>();
    net.add<nn::GRU>(5, 3, true);
    net.add<nn::LSTM>(4, 3, true);
    net.add<nn::SimpleRNN>(4, 3, false, nn::Activation::elu(0.5f));
    net.add<nn::RBF>(6);
    net.add<nn::Dropout>(0.3f);
    net.add<nn::Dense>(3, nn::Activation::softmax());
    net.compile(nn::Loss::CategoricalCrossEntropy, std::make_unique<nn::Adam>(1e-3f));
    nn::Rng rng(2);
    nn::Matrix x = random_matrix(5, 72, rng);
    nn::Matrix p1 = net.predict(x);
    std::stringstream ss;
    net.save(ss);
    nn::Network back = nn::Network::load(ss);
    nn::Matrix p2 = back.predict(x);
    CHECK(back.size() == net.size());
    CHECK(back.parameter_count() == net.parameter_count());
    double err = 0;
    for (std::size_t i = 0; i < p1.size(); ++i) err = std::max(err, (double)std::fabs(p1[i] - p2[i]));
    CHECK_MSG(err == 0, "prediction mismatch after load: %g", err);
    std::stringstream bad("garbage");
    bool threw = false;
    try { nn::Network::load(bad); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

TEST(learns_xor) {
    nn::Matrix x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    nn::Matrix y = {{0}, {1}, {1}, {0}};
    nn::Network net(2, 4);
    net.add<nn::Dense>(8, nn::Activation::tanh());
    net.add<nn::Dense>(1, nn::Activation::sigmoid());
    net.compile(nn::Loss::BinaryCrossEntropy, std::make_unique<nn::Adam>(0.05f));
    nn::FitOptions o;
    o.epochs = 500;
    o.batch_size = 4;
    o.verbose = 0;
    net.fit(x, y, o);
    CHECK(net.evaluate_accuracy(x, y) == 1.0f);
}

TEST(learns_seeds) {
    nn::Matrix raw = nn::data::load_csv(NN_DATA_DIR "/seeds_dataset.csv");
    CHECK(raw.rows() == 210 && raw.cols() == 8);
    auto [x, labels] = nn::data::split_xy(raw);
    std::size_t k = nn::data::encode_labels(labels);
    CHECK(k == 3);
    nn::Rng rng(10);
    auto s = nn::data::train_test_split(x, nn::data::one_hot(labels, k), 0.25f, rng);
    nn::data::StandardScaler sc;
    sc.fit_transform(s.x_train);
    sc.transform(s.x_test);
    nn::Network net(7, 3);
    net.add<nn::Dense>(16, nn::Activation::relu());
    net.add<nn::Dense>(3, nn::Activation::softmax());
    net.compile(nn::Loss::CategoricalCrossEntropy, std::make_unique<nn::Adam>(0.01f));
    nn::FitOptions o;
    o.epochs = 150;
    o.batch_size = 16;
    o.verbose = 0;
    net.fit(s.x_train, s.y_train, o);
    const float acc = net.evaluate_accuracy(s.x_test, s.y_test);
    CHECK_MSG(acc >= 0.88f, "seeds test accuracy %.3f", acc);
}

TEST(learns_sequence_sum_sign) {
    // classify whether the sum of a random sequence is positive (needs memory)
    const std::size_t N = 512, T = 8;
    nn::Rng rng(4);
    nn::Matrix x(N, T), y(N, 1);
    for (std::size_t i = 0; i < N; ++i) {
        float s = 0;
        for (std::size_t t = 0; t < T; ++t) s += (x(i, t) = rng.uniform(-1, 1));
        y(i, 0) = s > 0;
    }
    for (int which = 0; which < 3; ++which) {
        nn::Network net(T, 6);
        if (which == 0) net.add<nn::LSTM>(12, T);
        if (which == 1) net.add<nn::GRU>(12, T);
        if (which == 2) net.add<nn::SimpleRNN>(12, T);
        net.add<nn::Dense>(1, nn::Activation::sigmoid());
        net.compile(nn::Loss::BinaryCrossEntropy, std::make_unique<nn::Adam>(0.01f));
        nn::FitOptions o;
        o.epochs = 40;
        o.verbose = 0;
        net.fit(x, y, o);
        const float acc = net.evaluate_accuracy(x, y);
        CHECK_MSG(acc > 0.9f, "recurrent variant %d accuracy %.3f", which, acc);
    }
}

TEST(lr_schedules) {
    nn::Matrix x = {{0}, {1}};
    nn::Matrix y = {{0}, {1}};
    nn::Network net(1, 1);
    net.add<nn::Dense>(1);
    net.compile(nn::Loss::MeanSquaredError, std::make_unique<nn::SGD>(0.1f));
    nn::FitOptions o;
    o.epochs = 4;
    o.verbose = 0;
    o.schedule = nn::LRSchedule::step_decay(2, 0.5f);
    auto h = net.fit(x, y, o);
    CHECK(std::fabs(h.learning_rate[0] - 0.1f) < 1e-7 && std::fabs(h.learning_rate[2] - 0.05f) < 1e-7);
    CHECK(std::fabs(net.optimizer()->learning_rate() - 0.025f) < 1e-7);
}

TEST(spiking_lif_rate) {
    // constant suprathreshold current: analytic ISI = t_ref + tau ln(R I / (R I - (thr - rest)))
    nn::spiking::Network sn(0.01f);
    nn::spiking::LIFParams p;
    auto pop = sn.add_population(1, p);
    sn.set_bias_current(pop, 2.0f);  // RI = 20 mV > 15 mV
    sn.run(100000);                  // 1 s
    const double isi = p.t_ref + p.tau_m * std::log(20.0 / (20.0 - 15.0));
    const double expected = 1000.0 / isi;
    CHECK_MSG(std::fabs(sn.firing_rate(pop) - expected) < 0.03 * expected, "LIF rate %.2f vs %.2f",
              sn.firing_rate(pop), expected);
}

TEST(spiking_models_fire) {
    nn::spiking::Network sn(0.1f);
    auto izh = sn.add_population(4, nn::spiking::IzhikevichParams::regular_spiking());
    auto fs = sn.add_population(4, nn::spiking::IzhikevichParams::fast_spiking());
    auto adex = sn.add_population(4, nn::spiking::AdExParams{});
    auto hh = sn.add_population(4, nn::spiking::HodgkinHuxleyParams{});
    auto quiet = sn.add_population(4, nn::spiking::LIFParams{});
    sn.set_bias_current(izh, 10.0f);
    sn.set_bias_current(fs, 10.0f);
    sn.set_bias_current(adex, 800.0f);
    sn.set_bias_current(hh, 10.0f);
    sn.run(10000);  // 1 s
    CHECK_MSG(sn.firing_rate(izh) > 5, "izh %.1f", sn.firing_rate(izh));
    CHECK_MSG(sn.firing_rate(fs) > sn.firing_rate(izh), "fs %.1f", sn.firing_rate(fs));
    CHECK_MSG(sn.firing_rate(adex) > 5, "adex %.1f", sn.firing_rate(adex));
    CHECK_MSG(sn.firing_rate(hh) > 40 && sn.firing_rate(hh) < 120, "hh %.1f", sn.firing_rate(hh));
    CHECK(sn.firing_rate(quiet) == 0);
    for (int i = 0; i < 4; ++i) CHECK(std::isfinite(sn.voltage(hh)[i]));
}

TEST(spiking_propagation_and_stdp) {
    nn::spiking::Network sn(0.1f, 2.0f, 3);
    nn::spiking::LIFParams lp;
    auto pre = sn.add_population(1, lp, "pre");
    auto post = sn.add_population(1, lp, "post");
    sn.connect_one_to_one(pre, post, 40.0f);  // strong: every pre spike causes a post spike
    nn::spiking::STDPParams sp;
    sp.w_max = 100.0f;
    sn.enable_stdp(pre, post, sp);
    sn.set_bias_current(pre, 2.0f);
    sn.run(5000);
    CHECK(sn.spike_counts()[1] > 0);
    CHECK_MSG(sn.weights()[0] > 40.0f, "causal pairing should potentiate, w=%.3f", sn.weights()[0]);

    // random connectivity count is close to p * n^2
    nn::spiking::Network big(0.1f);
    auto a = big.add_population(1000, nn::spiking::IzhikevichParams{});
    std::size_t c = big.connect(a, a, 0.05f, 1.0f);
    CHECK_MSG(c > 47000 && c < 53000, "synapses %zu", c);
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("%s\n", nn::build_info());
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int ran = 0;
    for (auto& t : registry()) {
        if (filter && std::string(t.name).find(filter) == std::string::npos) continue;
        g_current = t.name;
        const int before = g_failed;
        try {
            t.fn();
        } catch (const std::exception& e) {
            ++g_failed;
            std::printf("  FAIL [%s] exception: %s\n", t.name, e.what());
        }
        std::printf("%s %s\n", g_failed == before ? "[ OK ]" : "[FAIL]", t.name);
        ++ran;
    }
    std::printf("\n%d tests, %d checks, %d failures\n", ran, g_checks, g_failed);
    return g_failed ? 1 : 0;
}
