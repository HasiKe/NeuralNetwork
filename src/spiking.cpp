#include "nn/spiking.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "nn/fastmath.hpp"

namespace nn::spiking {

namespace fm = fastmath;

namespace {
constexpr real kHHSubstep = 0.025f;  // ms, stability limit for explicit HH integration

// x / (1 - exp(-x / k)) with the removable singularity at x = 0 handled.
inline real vtrap(real x, real k) {
    return std::fabs(x / k) < 1e-6f ? k * (1.0f + x / (2.0f * k)) : x / (1.0f - std::exp(-x / k));
}
}  // namespace

Network::Network(real dt, real tau_syn, std::uint64_t seed)
    : m_dt(dt), m_tau_syn(tau_syn), m_syn_decay(std::exp(-dt / tau_syn)), m_rng(seed) {
    if (dt <= 0 || tau_syn <= 0) throw std::invalid_argument("spiking: dt and tau_syn must be > 0");
}

PopId Network::add_pop(std::size_t n, Model m, const std::string& name) {
    if (m_finalized) throw std::logic_error("spiking: cannot add populations after the first step");
    if (n == 0) throw std::invalid_argument("spiking: population size must be > 0");
    Population p;
    p.name = name.empty() ? "pop" + std::to_string(m_pops.size()) : name;
    p.model = m;
    p.offset = static_cast<std::uint32_t>(m_n);
    p.n = static_cast<std::uint32_t>(n);
    m_pops.push_back(p);
    m_n += n;
    for (auto* v : {&m_v, &m_u, &m_h, &m_n_gate, &m_ref, &m_isyn, &m_iext}) v->resize(m_n, 0.0f);
    m_fired.resize(m_n, 0);
    m_counts.resize(m_n, 0);
    return static_cast<PopId>(m_pops.size() - 1);
}

PopId Network::add_population(std::size_t n, const LIFParams& p, const std::string& name) {
    PopId id = add_pop(n, Model::LIF, name);
    m_pops[id].lif = p;
    reset_state();
    return id;
}
PopId Network::add_population(std::size_t n, const IzhikevichParams& p, const std::string& name) {
    PopId id = add_pop(n, Model::Izhikevich, name);
    m_pops[id].izh = p;
    reset_state();
    return id;
}
PopId Network::add_population(std::size_t n, const AdExParams& p, const std::string& name) {
    PopId id = add_pop(n, Model::AdEx, name);
    m_pops[id].adex = p;
    reset_state();
    return id;
}
PopId Network::add_population(std::size_t n, const HodgkinHuxleyParams& p, const std::string& name) {
    PopId id = add_pop(n, Model::HodgkinHuxley, name);
    m_pops[id].hh = p;
    reset_state();
    return id;
}

void Network::reset_state() {
    for (const auto& p : m_pops) {
        for (std::uint32_t i = p.offset; i < p.offset + p.n; ++i) {
            m_ref[i] = 0;
            m_isyn[i] = 0;
            m_iext[i] = 0;
            m_fired[i] = 0;
            m_counts[i] = 0;
            switch (p.model) {
                case Model::LIF: m_v[i] = p.lif.v_rest; break;
                case Model::Izhikevich:
                    m_v[i] = p.izh.c;
                    m_u[i] = p.izh.b * p.izh.c;
                    break;
                case Model::AdEx:
                    m_v[i] = p.adex.e_l;
                    m_u[i] = 0;
                    break;
                case Model::HodgkinHuxley:
                    m_v[i] = -65.0f;
                    m_u[i] = 0.0529f;     // m
                    m_h[i] = 0.5961f;     // h
                    m_n_gate[i] = 0.3177f;  // n
                    break;
            }
        }
    }
    std::fill(m_xpre.begin(), m_xpre.end(), 0.0f);
    std::fill(m_xpost.begin(), m_xpost.end(), 0.0f);
    m_spiked.clear();
    m_step = 0;
}

std::size_t Network::connect(PopId pre, PopId post, real prob, real weight, real stddev, bool autapses) {
    if (m_finalized) throw std::logic_error("spiking: cannot connect after the first step");
    const Population& a = m_pops.at(pre);
    const Population& b = m_pops.at(post);
    if (prob <= 0) return 0;
    if (prob >= 1) {
        std::size_t c = 0;
        for (std::uint32_t i = 0; i < a.n; ++i)
            for (std::uint32_t j = 0; j < b.n; ++j) {
                if (!autapses && pre == post && i == j) continue;
                m_coo.push_back({a.offset + i, b.offset + j, weight + (stddev > 0 ? stddev * m_rng.normal() : 0.0f)});
                ++c;
            }
        return c;
    }
    // geometric skipping: O(#synapses) instead of O(n_pre * n_post) random draws
    const double log_q = std::log(1.0 - static_cast<double>(prob));
    const std::uint64_t total = static_cast<std::uint64_t>(a.n) * b.n;
    std::size_t c = 0;
    std::uint64_t k = 0;
    for (;;) {
        const double u = 1.0 - static_cast<double>(m_rng.uniform());  // (0, 1]
        k += static_cast<std::uint64_t>(std::floor(std::log(u) / log_q));
        if (k >= total) break;
        const std::uint32_t i = static_cast<std::uint32_t>(k / b.n), j = static_cast<std::uint32_t>(k % b.n);
        ++k;
        if (!autapses && pre == post && i == j) continue;
        m_coo.push_back({a.offset + i, b.offset + j, weight + (stddev > 0 ? stddev * m_rng.normal() : 0.0f)});
        ++c;
    }
    return c;
}

std::size_t Network::connect_all_to_all(PopId pre, PopId post, real weight, bool autapses) {
    return connect(pre, post, 1.0f, weight, 0, autapses);
}

std::size_t Network::connect_one_to_one(PopId pre, PopId post, real weight) {
    if (m_finalized) throw std::logic_error("spiking: cannot connect after the first step");
    const Population& a = m_pops.at(pre);
    const Population& b = m_pops.at(post);
    if (a.n != b.n) throw std::invalid_argument("spiking: one_to_one needs equal population sizes");
    for (std::uint32_t i = 0; i < a.n; ++i) m_coo.push_back({a.offset + i, b.offset + i, weight});
    return a.n;
}

void Network::enable_stdp(PopId pre, PopId post, const STDPParams& p) {
    if (m_finalized) throw std::logic_error("spiking: enable_stdp must be called before the first step");
    (void)m_pops.at(pre);  // range check, throws std::out_of_range
    (void)m_pops.at(post);
    if (m_rules.size() >= 32000) throw std::length_error("spiking: too many plasticity rules");
    m_rules.push_back({pre, post, p});
}

void Network::set_bias_current(PopId pop, real current) { m_pops.at(pop).bias = current; }

real* Network::input_current(PopId pop) { return m_iext.data() + m_pops.at(pop).offset; }

void Network::add_poisson_input(PopId pop, real rate_hz, real weight) {
    m_pops.at(pop).poisson_rate = rate_hz;
    m_pops.at(pop).poisson_weight = weight;
}

void Network::finalize() {
    if (m_finalized) return;
    const std::size_t S = m_coo.size();
    // counting sort by pre -> CSR
    m_row.assign(m_n + 1, 0);
    for (const auto& e : m_coo) ++m_row[e.pre + 1];
    for (std::size_t i = 0; i < m_n; ++i) m_row[i + 1] += m_row[i];
    m_post.resize(S);
    m_w.resize(S);
    m_rule.assign(S, -1);
    std::vector<std::uint32_t> fill(m_row.begin(), m_row.end() - 1);
    std::vector<std::uint32_t> pre_of(S);
    std::vector<PopId> pop_of(m_n);
    for (PopId p = 0; p < m_pops.size(); ++p)
        std::fill(pop_of.begin() + m_pops[p].offset, pop_of.begin() + m_pops[p].offset + m_pops[p].n, p);
    for (const auto& e : m_coo) {
        const std::uint32_t s = fill[e.pre]++;
        m_post[s] = e.post;
        m_w[s] = e.w;
        pre_of[s] = e.pre;
        for (std::size_t r = 0; r < m_rules.size(); ++r)
            if (m_rules[r].pre == pop_of[e.pre] && m_rules[r].post == pop_of[e.post]) {
                m_rule[s] = static_cast<std::int16_t>(r);
                break;
            }
    }
    m_coo.clear();
    m_coo.shrink_to_fit();

    if (!m_rules.empty()) {
        // reverse index (incoming synapses per post neuron) for potentiation
        m_rev_row.assign(m_n + 1, 0);
        for (std::size_t s = 0; s < S; ++s)
            if (m_rule[s] >= 0) ++m_rev_row[m_post[s] + 1];
        for (std::size_t i = 0; i < m_n; ++i) m_rev_row[i + 1] += m_rev_row[i];
        m_rev_syn.resize(m_rev_row[m_n]);
        std::vector<std::uint32_t> f2(m_rev_row.begin(), m_rev_row.end() - 1);
        for (std::size_t s = 0; s < S; ++s)
            if (m_rule[s] >= 0) m_rev_syn[f2[m_post[s]]++] = static_cast<std::uint32_t>(s);
        m_pre_of = std::move(pre_of);
        m_xpre.assign(m_n, 0.0f);
        m_xpost.assign(m_n, 0.0f);
        m_decay_plus = std::exp(-m_dt / m_rules[0].p.tau_plus);
        m_decay_minus = std::exp(-m_dt / m_rules[0].p.tau_minus);
    }
    m_finalized = true;
}

void Network::update_population(Population& p) {
    const std::size_t o = p.offset, n = p.n;
    real* NN_RESTRICT v = m_v.data() + o;
    real* NN_RESTRICT u = m_u.data() + o;
    real* NN_RESTRICT ref = m_ref.data() + o;
    const real* NN_RESTRICT isyn = m_isyn.data() + o;
    real* NN_RESTRICT iext = m_iext.data() + o;
    std::uint8_t* NN_RESTRICT fired = m_fired.data() + o;
    const real dt = m_dt, bias = p.bias;

    switch (p.model) {
        case Model::LIF: {
            const LIFParams& q = p.lif;
            const real k = dt / q.tau_m;
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real I = bias + iext[i] + isyn[i];
                const bool refractory = ref[i] > 0;
                real vn = v[i] + k * (q.v_rest - v[i] + q.r_m * I);
                vn = refractory ? q.v_reset : vn;
                const bool spike = vn >= q.v_thresh;
                v[i] = spike ? q.v_reset : vn;
                ref[i] = spike ? q.t_ref : (refractory ? ref[i] - dt : 0.0f);
                fired[i] = spike;
            }
            break;
        }
        case Model::Izhikevich: {
            const IzhikevichParams& q = p.izh;
            const real h = 0.5f * dt;  // two half steps for v (numerical stability)
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real I = bias + iext[i] + isyn[i];
                real vv = v[i], uu = u[i];
                vv += h * (0.04f * vv * vv + 5.0f * vv + 140.0f - uu + I);
                vv += h * (0.04f * vv * vv + 5.0f * vv + 140.0f - uu + I);
                uu += dt * q.a * (q.b * vv - uu);
                const bool spike = vv >= 30.0f;
                v[i] = spike ? q.c : vv;
                u[i] = spike ? uu + q.d : uu;
                fired[i] = spike;
            }
            break;
        }
        case Model::AdEx: {
            const AdExParams& q = p.adex;
            const real invC = 1.0f / q.C, invdT = 1.0f / q.delta_t, invtw = 1.0f / q.tau_w;
            NN_SIMD for (std::size_t i = 0; i < n; ++i) {
                const real I = bias + iext[i] + isyn[i];
                const real vv = v[i], w = u[i];
                const real ex = q.g_l * q.delta_t * fm::exp((vv - q.v_t) * invdT);
                real vn = vv + dt * invC * (-q.g_l * (vv - q.e_l) + ex - w + I);
                const real wn = w + dt * invtw * (q.a * (vv - q.e_l) - w);
                const bool spike = vn >= q.v_peak;
                v[i] = spike ? q.v_reset : vn;
                u[i] = spike ? wn + q.b : wn;
                fired[i] = spike;
            }
            break;
        }
        case Model::HodgkinHuxley: {
            const HodgkinHuxleyParams& q = p.hh;
            real* NN_RESTRICT hg = m_h.data() + o;
            real* NN_RESTRICT ng = m_n_gate.data() + o;
            const int sub = std::max(1, static_cast<int>(std::ceil(dt / kHHSubstep)));
            const real ds = dt / static_cast<real>(sub);
            for (std::size_t i = 0; i < n; ++i) {
                const real I = bias + iext[i] + isyn[i];
                real vv = v[i], m = u[i], hh = hg[i], nn = ng[i];
                bool spike = false;
                for (int s = 0; s < sub; ++s) {
                    const real an = 0.01f * vtrap(vv + 55.0f, 10.0f);
                    const real bn = 0.125f * std::exp(-(vv + 65.0f) / 80.0f);
                    const real am = 0.1f * vtrap(vv + 40.0f, 10.0f);
                    const real bm = 4.0f * std::exp(-(vv + 65.0f) / 18.0f);
                    const real ah = 0.07f * std::exp(-(vv + 65.0f) / 20.0f);
                    const real bh = 1.0f / (1.0f + std::exp(-(vv + 35.0f) / 10.0f));
                    // Rush-Larsen exponential integration of the gates
                    auto gate = [ds](real g, real a, real b) {
                        const real tot = a + b, inf = a / tot;
                        return inf + (g - inf) * std::exp(-ds * tot);
                    };
                    m = gate(m, am, bm);
                    hh = gate(hh, ah, bh);
                    nn = gate(nn, an, bn);
                    const real n2 = nn * nn;
                    const real ina = q.g_na * m * m * m * hh * (vv - q.e_na);
                    const real ik = q.g_k * n2 * n2 * (vv - q.e_k);
                    const real il = q.g_l * (vv - q.e_l);
                    const real vprev = vv;
                    vv += ds * (I - ina - ik - il) / q.C;
                    spike = spike || (vprev < q.v_spike && vv >= q.v_spike);
                }
                v[i] = vv;
                u[i] = m;
                hg[i] = hh;
                ng[i] = nn;
                fired[i] = spike;
            }
            break;
        }
    }
    std::fill(iext, iext + n, 0.0f);
}

void Network::step() {
    finalize();
    // synaptic current decay
    {
        real* NN_RESTRICT is = m_isyn.data();
        const real d = m_syn_decay;
        const std::size_t n = m_n;
        NN_SIMD for (std::size_t i = 0; i < n; ++i) is[i] *= d;
    }
    // Poisson drive
    for (auto& p : m_pops) {
        if (p.poisson_rate <= 0) continue;
        const real prob = p.poisson_rate * m_dt * 1e-3f;
        const std::uint32_t thresh = static_cast<std::uint32_t>(std::min(1.0f, prob) * 16777216.0f);
        for (std::uint32_t i = p.offset; i < p.offset + p.n; ++i)
            if (static_cast<std::uint32_t>(m_rng.next_u64() >> 40) < thresh) m_isyn[i] += p.poisson_weight;
    }
    for (auto& p : m_pops) update_population(p);

    m_spiked.clear();
    for (std::uint32_t i = 0; i < m_n; ++i)
        if (m_fired[i]) m_spiked.push_back(i);

    // plasticity
    if (!m_rules.empty()) {
        const std::size_t n = m_n;
        real* NN_RESTRICT xp = m_xpre.data();
        real* NN_RESTRICT xq = m_xpost.data();
        const real dp = m_decay_plus, dm = m_decay_minus;
        NN_SIMD for (std::size_t i = 0; i < n; ++i) {
            xp[i] *= dp;
            xq[i] *= dm;
        }
        for (std::uint32_t i : m_spiked) {
            // depression: pre spike after post activity
            for (std::uint32_t s = m_row[i]; s < m_row[i + 1]; ++s) {
                const int r = m_rule[s];
                if (r < 0) continue;
                const STDPParams& q = m_rules[static_cast<std::size_t>(r)].p;
                m_w[s] = std::max(q.w_min, m_w[s] - q.a_minus * m_xpost[m_post[s]]);
            }
            // potentiation: post spike after pre activity
            for (std::uint32_t k = m_rev_row[i]; k < m_rev_row[i + 1]; ++k) {
                const std::uint32_t s = m_rev_syn[k];
                const STDPParams& q = m_rules[static_cast<std::size_t>(m_rule[s])].p;
                m_w[s] = std::min(q.w_max, m_w[s] + q.a_plus * m_xpre[m_pre_of[s]]);
            }
        }
        for (std::uint32_t i : m_spiked) {
            m_xpre[i] += 1.0f;
            m_xpost[i] += 1.0f;
        }
    }

    // event-driven propagation (arrives next step)
    real* NN_RESTRICT is = m_isyn.data();
    for (std::uint32_t i : m_spiked) {
        ++m_counts[i];
        if (m_record) m_log.emplace_back(m_step, i);
        const std::uint32_t* NN_RESTRICT post = m_post.data();
        const real* NN_RESTRICT w = m_w.data();
        for (std::uint32_t s = m_row[i]; s < m_row[i + 1]; ++s) is[post[s]] += w[s];
    }
    ++m_step;
}

real Network::firing_rate(PopId pop) const {
    const Population& p = m_pops.at(pop);
    if (m_step == 0) return 0;
    std::uint64_t total = 0;
    for (std::uint32_t i = p.offset; i < p.offset + p.n; ++i) total += m_counts[i];
    const double seconds = static_cast<double>(m_step) * m_dt * 1e-3;
    return static_cast<real>(static_cast<double>(total) / (p.n * seconds));
}

}  // namespace nn::spiking
