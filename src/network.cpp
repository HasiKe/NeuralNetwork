#include "nn/network.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

#include "serialize.hpp"

namespace nn {

namespace {
constexpr std::uint32_t kMagic = 0x54454E4E;  // "NNET"
constexpr std::uint32_t kFormatVersion = 2;

bool is_classification(Loss l) {
    return l == Loss::CategoricalCrossEntropy || l == Loss::BinaryCrossEntropy;
}

// Number of correct predictions in a batch.
std::size_t count_correct(const Matrix& pred, const Matrix& target) {
    std::size_t correct = 0;
    const std::size_t rows = pred.rows(), pc = pred.cols(), tc = target.cols();
    for (std::size_t r = 0; r < rows; ++r) {
        const real* p = pred.row(r);
        const real* t = target.row(r);
        if (pc == 1) {
            correct += (p[0] >= 0.5f) == (t[0] >= 0.5f);
        } else if (tc == 1) {
            const std::size_t a = static_cast<std::size_t>(std::max_element(p, p + pc) - p);
            correct += a == static_cast<std::size_t>(t[0]);
        } else {
            const std::size_t a = static_cast<std::size_t>(std::max_element(p, p + pc) - p);
            const std::size_t b = static_cast<std::size_t>(std::max_element(t, t + tc) - t);
            correct += a == b;
        }
    }
    return correct;
}

void copy_rows(const Matrix& src, std::size_t first, std::size_t count, Matrix& dst) {
    dst.resize(count, src.cols());
    std::memcpy(dst.data(), src.row(first), count * src.cols() * sizeof(real));
}
}  // namespace

Network::Network(std::size_t input_dim, std::uint64_t seed) : m_input_dim(input_dim), m_rng(seed) {}
Network::Network(Network&&) noexcept = default;
Network& Network::operator=(Network&&) noexcept = default;
Network::~Network() = default;

Layer& Network::add(LayerPtr layer) {
    if (!layer) throw std::invalid_argument("nn: null layer");
    const std::size_t in = m_layers.empty() ? m_input_dim : m_layers.back()->output_dim();
    if (in == 0) throw std::invalid_argument("nn: network input dimension must be > 0");
    layer->build(in, m_rng);
    layer->seed(m_rng.next_u64());
    if (!m_layers.empty()) m_layers.back()->set_fused_output(false);
    m_layers.push_back(std::move(layer));
    m_acts.emplace_back();
    m_grads.emplace_back();
    m_params_dirty = true;
    if (m_opt) m_layers.back()->set_fused_output(m_layers.back()->can_fuse_output(static_cast<int>(m_loss)));
    return *m_layers.back();
}

void Network::compile(Loss loss, std::unique_ptr<Optimizer> opt) {
    m_loss = loss;
    m_opt = std::move(opt);
    if (!m_opt) throw std::invalid_argument("nn: optimizer required");
    for (auto& l : m_layers) l->set_fused_output(false);
    if (!m_layers.empty())
        m_layers.back()->set_fused_output(m_layers.back()->can_fuse_output(static_cast<int>(loss)));
    m_params_dirty = true;
}

std::size_t Network::output_dim() const {
    return m_layers.empty() ? m_input_dim : m_layers.back()->output_dim();
}

void Network::refresh_params() {
    if (!m_params_dirty) return;
    m_params.clear();
    for (auto& l : m_layers) l->parameters(m_params);
    if (m_opt) m_opt->reset();
    m_params_dirty = false;
}

const Matrix& Network::forward(const Matrix& x, bool training) {
    if (m_layers.empty()) throw std::logic_error("nn: network has no layers");
    if (x.cols() != m_input_dim) throw std::invalid_argument("nn: input width mismatch");
    m_last_input = &x;
    const Matrix* in = &x;
    for (std::size_t i = 0; i < m_layers.size(); ++i) {
        m_layers[i]->forward(*in, m_acts[i], training);
        in = &m_acts[i];
    }
    return m_acts.back();
}

real Network::backward(const Matrix& target) {
    if (!m_last_input) throw std::logic_error("nn: backward() without forward()");
    const std::size_t L = m_layers.size();
    const bool fused = m_layers.back()->can_fuse_output(static_cast<int>(m_loss)) && m_opt;
    const real loss = compute_loss(m_loss, m_acts.back(), target, &m_loss_grad, fused);
    const Matrix* grad = &m_loss_grad;
    for (std::size_t i = L; i-- > 0;) {
        const Matrix& in = i == 0 ? *m_last_input : m_acts[i - 1];
        Matrix* gin = i == 0 ? nullptr : &m_grads[i - 1];
        m_layers[i]->backward(in, m_acts[i], *grad, gin);
        grad = gin;
    }
    return loss;
}

real Network::train_batch(const Matrix& x, const Matrix& y) {
    if (!m_opt) throw std::logic_error("nn: call compile() before training");
    refresh_params();
    forward(x, true);
    const real loss = backward(y);
    m_opt->step(m_params);
    return loss;
}

History Network::fit(const Matrix& x, const Matrix& y, const FitOptions& opt) {
    if (opt.validation_split > 0 && opt.validation_split < 1) {
        std::vector<std::size_t> idx(x.rows());
        std::iota(idx.begin(), idx.end(), 0);
        m_rng.shuffle(idx.begin(), idx.end());
        const std::size_t nv = static_cast<std::size_t>(std::round(x.rows() * opt.validation_split));
        const std::size_t nt = x.rows() - nv;
        Matrix xt, yt, xv, yv;
        xt.gather_rows(x, idx.data(), nt);
        yt.gather_rows(y, idx.data(), nt);
        xv.gather_rows(x, idx.data() + nt, nv);
        yv.gather_rows(y, idx.data() + nt, nv);
        return fit(xt, yt, xv, yv, opt);
    }
    return fit(x, y, Matrix(), Matrix(), opt);
}

History Network::fit(const Matrix& x, const Matrix& y, const Matrix& xv, const Matrix& yv,
                     const FitOptions& opt) {
    if (!m_opt) throw std::logic_error("nn: call compile() before fit()");
    if (x.rows() != y.rows()) throw std::invalid_argument("nn: x/y row count mismatch");
    if (y.cols() != output_dim()) throw std::invalid_argument("nn: target width != output width");
    const std::size_t N = x.rows();
    if (N == 0) return {};
    const std::size_t bs = std::max<std::size_t>(1, std::min(opt.batch_size, N));
    const bool has_val = xv.rows() > 0;
    const bool cls = is_classification(m_loss);
    const real lr0 = m_opt->learning_rate();

    std::vector<std::size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    History h;
    real best = std::numeric_limits<real>::infinity();
    std::size_t since_best = 0, plateau = 0;
    real plateau_best = std::numeric_limits<real>::infinity();
    std::vector<std::vector<real>> best_params;

    for (std::size_t e = 0; e < opt.epochs; ++e) {
        const auto t0 = std::chrono::steady_clock::now();
        if (opt.shuffle) m_rng.shuffle(idx.begin(), idx.end());
        double loss_sum = 0;
        std::size_t correct = 0;
        for (std::size_t s = 0; s < N; s += bs) {
            const std::size_t n = std::min(bs, N - s);
            m_bx.gather_rows(x, idx.data() + s, n);
            m_by.gather_rows(y, idx.data() + s, n);
            loss_sum += static_cast<double>(train_batch(m_bx, m_by)) * n;
            if (cls) correct += count_correct(m_acts.back(), m_by);
        }
        const real loss = static_cast<real>(loss_sum / N);
        const real acc = cls ? static_cast<real>(correct) / N : 0;
        real vloss = std::numeric_limits<real>::quiet_NaN(), vacc = 0;
        if (has_val) {
            vloss = evaluate_loss(xv, yv);
            if (cls) vacc = evaluate_accuracy(xv, yv);
        }
        h.loss.push_back(loss);
        h.accuracy.push_back(acc);
        h.val_loss.push_back(vloss);
        h.val_accuracy.push_back(vacc);
        h.learning_rate.push_back(m_opt->learning_rate());

        if (opt.verbose > 0) {
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::ostringstream line;
            line << "epoch " << (e + 1) << "/" << opt.epochs << "  loss " << std::setprecision(5) << loss;
            if (cls) line << "  acc " << std::setprecision(4) << acc;
            if (has_val) {
                line << "  val_loss " << std::setprecision(5) << vloss;
                if (cls) line << "  val_acc " << std::setprecision(4) << vacc;
            }
            line << "  lr " << std::setprecision(3) << m_opt->learning_rate() << "  " << std::fixed
                 << std::setprecision(1) << ms << "ms";
            std::cout << line.str() << '\n';
        }

        const real monitored = has_val ? vloss : loss;

        // learning-rate schedule
        const LRSchedule& sc = opt.schedule;
        real lr = m_opt->learning_rate();
        switch (sc.kind) {
            case LRSchedule::Kind::Constant: break;
            case LRSchedule::Kind::StepDecay:
                if (sc.step_size && (e + 1) % sc.step_size == 0) lr *= sc.factor;
                break;
            case LRSchedule::Kind::Exponential: lr *= sc.factor; break;
            case LRSchedule::Kind::Cosine: {
                const double pi = 3.14159265358979323846;
                const double p = static_cast<double>(e + 1) / static_cast<double>(opt.epochs);
                lr = static_cast<real>(sc.min_lr + 0.5 * (lr0 - sc.min_lr) * (1 + std::cos(pi * p)));
                break;
            }
            case LRSchedule::Kind::ReduceOnPlateau:
                if (monitored < plateau_best * (1 - 1e-4f)) {
                    plateau_best = monitored;
                    plateau = 0;
                } else if (++plateau >= sc.patience) {
                    lr *= sc.factor;
                    plateau = 0;
                }
                break;
        }
        m_opt->set_learning_rate(std::max(lr, sc.kind == LRSchedule::Kind::Constant ? lr : sc.min_lr));

        if (opt.on_epoch_end && !opt.on_epoch_end(e, loss, vloss)) break;

        if (opt.early_stopping_patience > 0) {
            if (monitored < best) {
                best = monitored;
                since_best = 0;
                best_params.resize(m_params.size());
                for (std::size_t k = 0; k < m_params.size(); ++k)
                    best_params[k].assign(m_params[k].value, m_params[k].value + m_params[k].size);
            } else if (++since_best >= opt.early_stopping_patience) {
                if (opt.verbose > 0) std::cout << "early stopping, restoring best weights\n";
                for (std::size_t k = 0; k < best_params.size(); ++k)
                    std::copy(best_params[k].begin(), best_params[k].end(), m_params[k].value);
                break;
            }
        }
    }
    return h;
}

Matrix Network::predict(const Matrix& x, std::size_t batch_size) {
    Matrix out(x.rows(), output_dim());
    if (x.rows() == 0) return out;
    batch_size = std::max<std::size_t>(1, batch_size);
    if (x.rows() <= batch_size) {
        out = forward(x, false);
        return out;
    }
    for (std::size_t s = 0; s < x.rows(); s += batch_size) {
        const std::size_t n = std::min(batch_size, x.rows() - s);
        copy_rows(x, s, n, m_bx);
        const Matrix& y = forward(m_bx, false);
        std::memcpy(out.row(s), y.data(), y.size() * sizeof(real));
    }
    return out;
}

std::vector<std::size_t> Network::predict_classes(const Matrix& x, std::size_t batch_size) {
    Matrix p = predict(x, batch_size);
    if (p.cols() == 1) {
        std::vector<std::size_t> r(p.rows());
        for (std::size_t i = 0; i < p.rows(); ++i) r[i] = p[i] >= 0.5f ? 1 : 0;
        return r;
    }
    return argmax_rows(p);
}

real Network::evaluate_loss(const Matrix& x, const Matrix& y, std::size_t batch_size) {
    if (x.rows() == 0) return 0;
    Matrix p = predict(x, batch_size);
    return compute_loss(m_loss, p, y, nullptr, false);
}

real Network::evaluate_accuracy(const Matrix& x, const Matrix& y, std::size_t batch_size) {
    if (x.rows() == 0) return 0;
    Matrix p = predict(x, batch_size);
    return static_cast<real>(count_correct(p, y)) / static_cast<real>(x.rows());
}

std::size_t Network::parameter_count() {
    std::size_t n = 0;
    for (auto& l : m_layers) n += l->parameter_count();
    return n;
}

std::string Network::summary() {
    std::ostringstream s;
    s << std::left << std::setw(4) << "#" << std::setw(56) << "layer" << std::setw(10) << "output"
      << "params\n";
    s << std::string(80, '-') << '\n';
    s << std::setw(4) << "-" << std::setw(56) << "Input" << std::setw(10) << m_input_dim << "0\n";
    for (std::size_t i = 0; i < m_layers.size(); ++i)
        s << std::setw(4) << i << std::setw(56) << m_layers[i]->name() << std::setw(10)
          << m_layers[i]->output_dim() << m_layers[i]->parameter_count() << '\n';
    s << std::string(80, '-') << '\n' << "total parameters: " << parameter_count() << '\n';
    return s.str();
}

void Network::save(std::ostream& os) const {
    io::put<std::uint32_t>(os, kMagic);
    io::put<std::uint32_t>(os, kFormatVersion);
    io::put_size(os, m_input_dim);
    io::put<std::int32_t>(os, static_cast<std::int32_t>(m_loss));
    io::put_size(os, m_layers.size());
    for (const auto& l : m_layers) {
        io::put<std::uint32_t>(os, static_cast<std::uint32_t>(l->type()));
        l->save(os);
    }
}

bool Network::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    save(f);
    return static_cast<bool>(f);
}

Network Network::load(std::istream& is) {
    io::expect(io::get<std::uint32_t>(is) == kMagic, "not a model file (bad magic)");
    const auto ver = io::get<std::uint32_t>(is);
    io::expect(ver == kFormatVersion, "unsupported model format version");
    Network net(io::get_size(is));
    net.m_loss = static_cast<Loss>(io::get<std::int32_t>(is));
    const std::size_t n = io::get_size(is);
    io::expect(n < 100000, "corrupt layer count");
    std::size_t width = net.m_input_dim;
    for (std::size_t i = 0; i < n; ++i) {
        LayerPtr l = make_layer(static_cast<LayerType>(io::get<std::uint32_t>(is)));
        l->load(is);
        io::expect(l->input_dim() == width, "layer width mismatch in model file");
        width = l->output_dim();
        l->seed(net.m_rng.next_u64());
        net.m_layers.push_back(std::move(l));
        net.m_acts.emplace_back();
        net.m_grads.emplace_back();
    }
    return net;
}

Network Network::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("nn: cannot open '" + path + "'");
    return load(f);
}

}  // namespace nn
