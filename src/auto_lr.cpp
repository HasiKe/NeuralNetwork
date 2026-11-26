// Automatic learning rate: LR range test, warmup, reduce on plateau, rollback on
// divergence and early stopping with the best weights.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "nn/network.hpp"

namespace nn {

namespace {

using Snapshot = std::vector<std::vector<real>>;

void take(const std::vector<ParamRef>& params, Snapshot& s) {
    s.resize(params.size());
    for (std::size_t k = 0; k < params.size(); ++k)
        s[k].assign(params[k].value, params[k].value + params[k].size);
}

void put_back(const Snapshot& s, std::vector<ParamRef>& params) {
    for (std::size_t k = 0; k < s.size() && k < params.size(); ++k)
        std::copy(s[k].begin(), s[k].end(), params[k].value);
}

bool all_finite(const std::vector<ParamRef>& params) {
    for (const auto& p : params)
        for (std::size_t i = 0; i < p.size; ++i)
            if (!std::isfinite(p.value[i])) return false;
    return true;
}

std::string num(real v) {
    std::ostringstream s;
    s << std::setprecision(3) << v;
    return s.str();
}

}  // namespace

LRFindResult Network::find_learning_rate(const Matrix& x, const Matrix& y, std::size_t batch_size,
                                         const AutoLR& o) {
    if (!m_opt) throw std::logic_error("nn: call compile() before find_learning_rate()");
    if (x.rows() != y.rows() || x.rows() == 0)
        throw std::invalid_argument("nn: find_learning_rate: empty data or x/y row count mismatch");
    if (y.cols() != output_dim()) throw std::invalid_argument("nn: target width != output width");
    if (!(o.min_lr > 0) || !(o.max_lr > o.min_lr) || o.range_steps < 10 || !(o.range_divisor >= 1))
        throw std::invalid_argument("nn: AutoLR: invalid range test settings");
    std::unique_ptr<Optimizer> probe = m_opt->clone();
    if (!probe)
        throw std::logic_error("nn: the range test needs Optimizer::clone() (or set AutoLR::initial_lr)");
    probe->reset();
    refresh_params();

    Snapshot weights;
    take(m_params, weights);
    std::swap(m_opt, probe);  // `probe` now holds the untouched user optimizer
    struct Restore {
        std::unique_ptr<Optimizer>&opt, &user;
        Snapshot& weights;
        std::vector<ParamRef>& params;
        ~Restore() {
            put_back(weights, params);
            std::swap(opt, user);
        }
    } restore{m_opt, probe, weights, m_params};

    const std::size_t N = x.rows();
    const std::size_t bs = std::max<std::size_t>(1, std::min(batch_size, N));
    std::vector<std::size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    Rng rng(0x5EED);  // own generator: the network's stream stays untouched
    rng.shuffle(idx.begin(), idx.end());

    LRFindResult r;
    const double growth = std::log(static_cast<double>(o.max_lr) / o.min_lr);
    constexpr double beta = 0.9;
    double avg = 0, best = std::numeric_limits<double>::infinity();
    std::size_t pos = 0;
    for (std::size_t i = 0; i < o.range_steps; ++i) {
        if (pos + bs > N) {
            rng.shuffle(idx.begin(), idx.end());
            pos = 0;
        }
        m_bx.gather_rows(x, idx.data() + pos, bs);
        m_by.gather_rows(y, idx.data() + pos, bs);
        pos += bs;
        const real lr = static_cast<real>(
            o.min_lr * std::exp(growth * static_cast<double>(i) / static_cast<double>(o.range_steps - 1)));
        m_opt->set_learning_rate(lr);
        const double loss = train_batch(m_bx, m_by);
        if (!std::isfinite(loss)) {
            r.diverged = true;
            break;
        }
        avg = beta * avg + (1 - beta) * loss;
        const double smooth = avg / (1 - std::pow(beta, static_cast<double>(i + 1)));
        r.lrs.push_back(lr);
        r.losses.push_back(static_cast<real>(smooth));
        // the first steps are noisy: only trust the minimum once the average has settled
        if (i >= 5 && smooth < best) {
            best = smooth;
            r.min_loss_lr = lr;
        }
        if (i >= 10 && smooth > 4 * best) {
            r.diverged = true;
            break;
        }
    }
    if (r.min_loss_lr <= 0) r.min_loss_lr = r.lrs.empty() ? o.min_lr : r.lrs.front();
    r.lr = std::max(o.min_lr, r.min_loss_lr / o.range_divisor);
    return r;
}

History Network::fit_auto(const Matrix& x_in, const Matrix& y_in, const Matrix& xv_in,
                          const Matrix& yv_in, const FitOptions& opt) {
    const AutoLR& a = opt.auto_lr;
    const LRSchedule::Kind kind = opt.schedule.kind;
    if (kind != LRSchedule::Kind::Constant && kind != LRSchedule::Kind::ReduceOnPlateau)
        throw std::invalid_argument("nn: auto_lr only combines with constant or reduce_on_plateau schedules");
    if (!(a.factor > 0 && a.factor < 1)) throw std::invalid_argument("nn: AutoLR: factor must be in (0, 1)");
    if (!(a.validation_split >= 0 && a.validation_split < 1))
        throw std::invalid_argument("nn: AutoLR: validation_split must be in [0, 1)");

    History h;
    AutoLRReport& rep = h.auto_lr;
    rep.used = true;

    // ---- monitoring data
    const Matrix *px = &x_in, *py = &y_in, *pxv = &xv_in, *pyv = &yv_in;
    Matrix xt, yt, xh, yh;
    if (xv_in.rows() == 0 && a.validation_split > 0 && x_in.rows() >= 40) {
        std::vector<std::size_t> all(x_in.rows());
        std::iota(all.begin(), all.end(), 0);
        m_rng.shuffle(all.begin(), all.end());
        const std::size_t nv = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::round(x_in.rows() * a.validation_split)));
        const std::size_t nt = x_in.rows() - nv;
        xt.gather_rows(x_in, all.data(), nt);
        yt.gather_rows(y_in, all.data(), nt);
        xh.gather_rows(x_in, all.data() + nt, nv);
        yh.gather_rows(y_in, all.data() + nt, nv);
        px = &xt;
        py = &yt;
        pxv = &xh;
        pyv = &yh;
    }
    const Matrix &x = *px, &y = *py, &xv = *pxv, &yv = *pyv;
    const bool has_val = xv.rows() > 0;
    rep.validation_rows = xv.rows();

    const std::size_t N = x.rows();
    const std::size_t bs = std::max<std::size_t>(1, std::min(opt.batch_size, N));
    const std::size_t steps_per_epoch = (N + bs - 1) / bs;
    const bool cls = m_loss == Loss::CategoricalCrossEntropy || m_loss == Loss::BinaryCrossEntropy;

    // ---- schedule parameters
    real factor = a.factor;
    std::size_t patience = a.patience ? a.patience : std::min<std::size_t>(20, std::max<std::size_t>(5, opt.epochs / 10));
    if (kind == LRSchedule::Kind::ReduceOnPlateau) {
        patience = std::max<std::size_t>(1, opt.schedule.patience);
        factor = opt.schedule.factor;
    }
    std::size_t stop_patience = a.stop_patience ? a.stop_patience : 4 * patience;
    if (opt.early_stopping_patience) stop_patience = opt.early_stopping_patience;
    rep.patience = patience;
    rep.stop_patience = stop_patience;

    // ---- start value
    refresh_params();
    if (a.initial_lr > 0) {
        rep.start_lr = a.initial_lr;
    } else {
        rep.range = find_learning_rate(x, y, bs, a);
        rep.start_lr = rep.range.lr;
    }
    real floor_lr = std::max(rep.start_lr * a.floor_ratio, std::numeric_limits<real>::min());
    if (kind == LRSchedule::Kind::ReduceOnPlateau) floor_lr = opt.schedule.min_lr;

    const std::size_t total_steps = opt.epochs * steps_per_epoch;
    std::size_t warmup = static_cast<std::size_t>(std::llround(std::max<real>(0, a.warmup_fraction) * total_steps));
    warmup = std::min(warmup, 5 * steps_per_epoch);
    rep.warmup_steps = warmup;

    // ---- best state so far (initially: the untrained network)
    auto monitored_loss = [&]() { return has_val ? evaluate_loss(xv, yv) : evaluate_loss(x, y); };
    const real initial_loss = monitored_loss();
    Snapshot best_weights;
    take(m_params, best_weights);
    real best = std::isfinite(initial_loss) ? initial_loss : std::numeric_limits<real>::infinity();
    rep.best_loss = initial_loss;

    if (opt.verbose > 0) {
        std::cout << "auto_lr: start lr " << num(rep.start_lr);
        if (a.initial_lr <= 0)
            std::cout << " (range test: lowest loss at " << num(rep.range.min_loss_lr) << ", " << rep.range.lrs.size()
                      << " steps)";
        std::cout << ", warmup " << warmup << " steps, factor " << num(factor) << " after " << patience
                  << " epochs, stop after " << stop_patience << ", monitoring "
                  << (has_val ? "validation loss (" + std::to_string(xv.rows()) + " rows)" : std::string("training loss"))
                  << '\n';
    }

    real base_lr = rep.start_lr;
    std::size_t warmup_left = warmup, warmup_len = warmup;
    m_opt->reset();
    std::vector<std::size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    detail::Plateau plateau{std::numeric_limits<real>::infinity()};
    std::size_t since_best = 0;

    for (std::size_t e = 0; e < opt.epochs; ++e) {
        if (opt.shuffle) m_rng.shuffle(idx.begin(), idx.end());
        double loss_sum = 0;
        std::size_t seen = 0, correct = 0;
        bool broken = false;
        for (std::size_t s = 0; s < N && !broken; s += bs) {
            const std::size_t n = std::min(bs, N - s);
            m_bx.gather_rows(x, idx.data() + s, n);
            m_by.gather_rows(y, idx.data() + s, n);
            real lr = base_lr;
            if (warmup_left > 0) {
                lr = base_lr * static_cast<real>(warmup_len - warmup_left + 1) / static_cast<real>(warmup_len);
                --warmup_left;
            }
            m_opt->set_learning_rate(lr);
            const real l = train_batch(m_bx, m_by);
            if (!std::isfinite(l)) {
                broken = true;
                break;
            }
            loss_sum += static_cast<double>(l) * n;
            seen += n;
            if (cls) {
                const Matrix& p = m_acts.back();
                for (std::size_t r = 0; r < p.rows(); ++r) {
                    if (p.cols() == 1) {
                        correct += (p(r, 0) >= 0.5f) == (m_by(r, 0) >= 0.5f);
                    } else {
                        const std::size_t i = static_cast<std::size_t>(std::max_element(p.row(r), p.row(r) + p.cols()) - p.row(r));
                        const std::size_t j = static_cast<std::size_t>(std::max_element(m_by.row(r), m_by.row(r) + m_by.cols()) - m_by.row(r));
                        correct += i == j;
                    }
                }
            }
        }
        const real loss = broken || seen == 0 ? std::numeric_limits<real>::quiet_NaN()
                                              : static_cast<real>(loss_sum / seen);
        const bool finite_weights = !broken && all_finite(m_params);
        real vloss = std::numeric_limits<real>::quiet_NaN(), vacc = 0;
        if (has_val && finite_weights) {
            vloss = evaluate_loss(xv, yv);
            if (cls) vacc = evaluate_accuracy(xv, yv);
        }
        // Without validation data the epoch average would mix old and new weights.
        const real monitored = !finite_weights ? std::numeric_limits<real>::quiet_NaN()
                                               : (has_val ? vloss : evaluate_loss(x, y));
        h.loss.push_back(loss);
        h.accuracy.push_back(cls && seen ? static_cast<real>(correct) / seen : 0);
        h.val_loss.push_back(vloss);
        h.val_accuracy.push_back(vacc);
        h.learning_rate.push_back(m_opt->learning_rate());
        if (opt.verbose > 0) {
            std::ostringstream line;
            line << "epoch " << (e + 1) << "/" << opt.epochs << "  loss " << std::setprecision(5) << loss;
            if (has_val) line << "  val_loss " << vloss;
            if (cls && has_val) line << "  val_acc " << std::setprecision(4) << vacc;
            line << "  lr " << std::setprecision(3) << m_opt->learning_rate();
            std::cout << line.str() << '\n';
        }

        // "Diverged" is relative to the best value, with a margin tied to the initial loss so
        // that normal jitter around a very small loss does not count.
        const real margin = std::isfinite(initial_loss) ? 0.1f * std::fabs(initial_loss) : 0;
        const bool diverged = !std::isfinite(monitored) ||
                              (std::isfinite(best) && monitored > std::max(best * a.divergence_factor, best + margin));
        if (diverged) {
            put_back(best_weights, m_params);
            m_opt->reset();
            ++rep.rollbacks;
            // harsher than a plateau step: the learning rate was clearly too high
            base_lr *= std::min<real>(factor, 0.3f);
            floor_lr = std::min(floor_lr, base_lr);
            warmup_len = warmup_left = std::max<std::size_t>(1, std::min(warmup, steps_per_epoch));
            plateau.wait = 0;
            if (opt.verbose > 0)
                std::cout << "auto_lr: divergence, restored best weights, lr -> " << num(base_lr) << '\n';
            if (rep.rollbacks > a.max_rollbacks) {
                rep.stopped_early = true;
                if (opt.verbose > 0) std::cout << "auto_lr: giving up after " << rep.rollbacks << " rollbacks\n";
                break;
            }
        } else {
            if (monitored < best) {
                best = monitored;
                rep.best_loss = monitored;
                rep.best_epoch = e + 1;
                take(m_params, best_weights);
                since_best = 0;
            } else {
                ++since_best;
            }
            if (warmup_left == 0 && plateau.update(monitored, patience) && base_lr > floor_lr) {
                base_lr = std::max(floor_lr, base_lr * factor);
                ++rep.reductions;
                if (opt.verbose > 0) std::cout << "auto_lr: plateau, lr -> " << num(base_lr) << '\n';
            }
        }

        if (opt.on_epoch_end && !opt.on_epoch_end(e, loss, vloss)) break;
        if (!diverged && since_best >= stop_patience && e + 1 < opt.epochs) {
            rep.stopped_early = true;
            if (opt.verbose > 0) std::cout << "auto_lr: early stopping\n";
            break;
        }
    }

    put_back(best_weights, m_params);
    rep.final_lr = base_lr;
    m_opt->set_learning_rate(base_lr);
    if (opt.verbose > 0)
        std::cout << "auto_lr: kept weights of epoch " << rep.best_epoch << " (loss " << num(rep.best_loss)
                  << "), final lr " << num(base_lr) << ", " << rep.reductions << " reductions, " << rep.rollbacks
                  << " rollbacks\n";
    return h;
}

}  // namespace nn
