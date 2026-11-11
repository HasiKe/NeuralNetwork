#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "config.hpp"
#include "random.hpp"

/// Biologically inspired spiking neurons, simulated with a fixed time step.
///
/// Units: time in ms, voltage in mV. Input currents are in each model's
/// native unit (LIF: nA with membrane resistance in MOhm, Izhikevich: model
/// units, AdEx: pA, Hodgkin-Huxley: uA/cm^2). Synaptic weights use the same unit
/// as the target population's input current.
///
/// Storage is structure-of-arrays per population (vectorisable updates) and
/// synapses are kept in CSR form, so spike propagation is event driven:
/// cost scales with the number of spikes, not the number of synapses.
namespace nn::spiking {

enum class Model : std::uint8_t { LIF, Izhikevich, AdEx, HodgkinHuxley };

/// Leaky integrate-and-fire.
struct LIFParams {
    real tau_m = 20.0f;     ///< membrane time constant [ms]
    real v_rest = -65.0f;   ///< resting potential [mV]
    real v_reset = -65.0f;  ///< reset potential [mV]
    real v_thresh = -50.0f; ///< spike threshold [mV]
    real r_m = 10.0f;       ///< membrane resistance [MOhm]
    real t_ref = 2.0f;      ///< absolute refractory period [ms]
};

/// Izhikevich (2003) simple model; presets reproduce the classic firing patterns.
struct IzhikevichParams {
    real a = 0.02f, b = 0.2f, c = -65.0f, d = 8.0f;
    static IzhikevichParams regular_spiking() { return {0.02f, 0.2f, -65.0f, 8.0f}; }
    static IzhikevichParams intrinsically_bursting() { return {0.02f, 0.2f, -55.0f, 4.0f}; }
    static IzhikevichParams chattering() { return {0.02f, 0.2f, -50.0f, 2.0f}; }
    static IzhikevichParams fast_spiking() { return {0.1f, 0.2f, -65.0f, 2.0f}; }
    static IzhikevichParams low_threshold_spiking() { return {0.02f, 0.25f, -65.0f, 2.0f}; }
    static IzhikevichParams thalamo_cortical() { return {0.02f, 0.25f, -65.0f, 0.05f}; }
    static IzhikevichParams resonator() { return {0.1f, 0.26f, -65.0f, 2.0f}; }
};

/// Adaptive exponential integrate-and-fire (Brette & Gerstner 2005).
struct AdExParams {
    real C = 281.0f;        ///< capacitance [pF]
    real g_l = 30.0f;       ///< leak conductance [nS]
    real e_l = -70.6f;      ///< leak reversal [mV]
    real v_t = -50.4f;      ///< threshold slope factor midpoint [mV]
    real delta_t = 2.0f;    ///< slope factor [mV]
    real tau_w = 144.0f;    ///< adaptation time constant [ms]
    real a = 4.0f;          ///< sub-threshold adaptation [nS]
    real b = 80.5f;         ///< spike-triggered adaptation [pA]
    real v_reset = -70.6f;  ///< [mV]
    real v_peak = 20.0f;    ///< spike cut-off [mV]
};

/// Hodgkin-Huxley (1952) squid axon model with Na, K and leak channels.
struct HodgkinHuxleyParams {
    real C = 1.0f;          ///< [uF/cm^2]
    real g_na = 120.0f, g_k = 36.0f, g_l = 0.3f;  ///< [mS/cm^2]
    real e_na = 50.0f, e_k = -77.0f, e_l = -54.387f;  ///< [mV]
    real v_spike = 0.0f;    ///< upward crossing counted as a spike [mV]
};

/// Pair-based spike-timing dependent plasticity with exponential traces.
struct STDPParams {
    real a_plus = 0.01f;     ///< potentiation step (pre before post)
    real a_minus = 0.012f;   ///< depression step (post before pre)
    real tau_plus = 20.0f;   ///< [ms]
    real tau_minus = 20.0f;  ///< [ms]
    real w_min = 0.0f, w_max = 1.0f;
};

using PopId = std::uint32_t;

class Network {
public:
    /// dt: integration step [ms]. tau_syn: decay of exponential synaptic currents [ms].
    explicit Network(real dt = 0.1f, real tau_syn = 5.0f, std::uint64_t seed = 1);

    PopId add_population(std::size_t n, const LIFParams& p, const std::string& name = "");
    PopId add_population(std::size_t n, const IzhikevichParams& p, const std::string& name = "");
    PopId add_population(std::size_t n, const AdExParams& p, const std::string& name = "");
    PopId add_population(std::size_t n, const HodgkinHuxleyParams& p, const std::string& name = "");

    /// Random connectivity: each (pre, post) pair connected with `probability`.
    /// Weight drawn from N(weight, weight_stddev). Returns synapses created.
    std::size_t connect(PopId pre, PopId post, real probability, real weight,
                        real weight_stddev = 0, bool allow_autapses = false);
    std::size_t connect_all_to_all(PopId pre, PopId post, real weight, bool allow_autapses = false);
    std::size_t connect_one_to_one(PopId pre, PopId post, real weight);
    /// Make synapses from pre to post plastic (must be called before the first step).
    void enable_stdp(PopId pre, PopId post, const STDPParams& p);

    /// Constant external current per neuron of a population (persistent until changed).
    void set_bias_current(PopId pop, real current);
    /// Per-neuron external current for the next step only (adds to bias).
    real* input_current(PopId pop);
    /// Independent Poisson spike trains driving every neuron of `pop`.
    void add_poisson_input(PopId pop, real rate_hz, real weight);

    /// Build synapse tables. Called automatically by the first step().
    void finalize();
    void step();
    void run(std::size_t steps) { for (std::size_t i = 0; i < steps; ++i) step(); }
    void reset_state();

    /// Global neuron ids that spiked in the most recent step.
    const std::vector<std::uint32_t>& spikes() const { return m_spiked; }
    void record_spikes(bool on) { m_record = on; }
    /// Recorded (step, global id) pairs.
    const std::vector<std::pair<std::uint64_t, std::uint32_t>>& spike_log() const { return m_log; }
    void clear_spike_log() { m_log.clear(); }
    /// Number of spikes per neuron since construction / reset_state().
    const std::vector<std::uint32_t>& spike_counts() const { return m_counts; }
    /// Mean firing rate of a population over the simulated time [Hz].
    real firing_rate(PopId pop) const;

    std::size_t neuron_count() const { return m_n; }
    std::size_t synapse_count() const { return m_post.size() + m_coo.size(); }
    std::size_t population_size(PopId pop) const { return m_pops.at(pop).n; }
    std::size_t population_count() const { return m_pops.size(); }
    const std::string& population_name(PopId pop) const { return m_pops.at(pop).name; }
    Model population_model(PopId pop) const { return m_pops.at(pop).model; }
    std::uint32_t first_neuron(PopId pop) const { return m_pops.at(pop).offset; }
    const real* voltage(PopId pop) const { return m_v.data() + m_pops.at(pop).offset; }
    real time() const { return static_cast<real>(m_step) * m_dt; }
    real dt() const { return m_dt; }
    /// Weight of every synapse (CSR order, after finalize()).
    std::vector<real>& weights() { return m_w; }
    /// CSR synapse table (after finalize()): synapses of neuron i are
    /// [synapse_offsets()[i], synapse_offsets()[i+1]) with targets synapse_targets().
    const std::vector<std::uint32_t>& synapse_offsets() const { return m_row; }
    const std::vector<std::uint32_t>& synapse_targets() const { return m_post; }

private:
    struct Population {
        std::string name;
        Model model;
        std::uint32_t offset, n;
        LIFParams lif;
        IzhikevichParams izh;
        AdExParams adex;
        HodgkinHuxleyParams hh;
        real bias = 0;
        real poisson_rate = 0, poisson_weight = 0;
    };
    struct Edge { std::uint32_t pre, post; real w; };
    struct PlasticRule { PopId pre, post; STDPParams p; };

    PopId add_pop(std::size_t n, Model m, const std::string& name);
    void update_population(Population& p);

    real m_dt, m_tau_syn, m_syn_decay;
    Rng m_rng;
    std::vector<Population> m_pops;
    std::size_t m_n = 0;
    std::uint64_t m_step = 0;
    bool m_finalized = false, m_record = false;

    // neuron state (global arrays, populations are contiguous slices)
    std::vector<real> m_v, m_u, m_h, m_n_gate, m_ref, m_isyn, m_iext;
    std::vector<std::uint8_t> m_fired;
    std::vector<std::uint32_t> m_counts, m_spiked;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> m_log;

    // synapses
    std::vector<Edge> m_coo;
    std::vector<std::uint32_t> m_row, m_post;  // CSR by pre
    std::vector<real> m_w;
    std::vector<std::int16_t> m_rule;          // plastic rule index or -1
    std::vector<std::uint32_t> m_rev_row, m_rev_syn;  // incoming synapse index by post
    std::vector<std::uint32_t> m_pre_of;              // pre neuron of each synapse (STDP only)
    std::vector<PlasticRule> m_rules;
    std::vector<real> m_xpre, m_xpost;         // STDP traces
    real m_decay_plus = 1, m_decay_minus = 1;
};

}  // namespace nn::spiking
