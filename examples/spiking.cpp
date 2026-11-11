// Spiking neurons: a cortical-like network of Izhikevich neurons plus single
// LIF / AdEx / Hodgkin-Huxley cells driven by constant currents.
#include <cstdio>
#include <string>

#include <nn/nn.hpp>

namespace sp = nn::spiking;

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::stoul(argv[1]) : 10000;

    sp::Network net(0.5f, 5.0f, 42);
    auto exc = net.add_population(n * 4 / 5, sp::IzhikevichParams::regular_spiking(), "exc");
    auto inh = net.add_population(n / 5, sp::IzhikevichParams::fast_spiking(), "inh");
    const float p = std::min(1.0f, 100.0f / static_cast<float>(n));  // ~100 synapses per neuron
    net.connect(exc, exc, p, 0.5f, 0.1f);
    net.connect(exc, inh, p, 0.5f, 0.1f);
    net.connect(inh, exc, p, -1.0f, 0.2f);
    net.connect(inh, inh, p, -1.0f, 0.2f);
    net.add_poisson_input(exc, 2000, 1.5f);
    net.add_poisson_input(inh, 2000, 1.5f);

    // isolated single-cell populations to compare neuron models
    auto lif = net.add_population(1, sp::LIFParams{}, "LIF");
    auto adex = net.add_population(1, sp::AdExParams{}, "AdEx");
    auto hh = net.add_population(1, sp::HodgkinHuxleyParams{}, "HH");
    auto burst = net.add_population(1, sp::IzhikevichParams::chattering(), "Izh-CH");
    net.set_bias_current(lif, 2.0f);     // nA
    net.set_bias_current(adex, 1000.0f); // pA
    net.set_bias_current(hh, 10.0f);     // uA/cm^2
    net.set_bias_current(burst, 10.0f);

    net.finalize();
    std::printf("%zu neurons, %zu synapses\n", net.neuron_count(), net.synapse_count());

    // crude population activity trace: spikes per 10 ms bin
    const std::size_t steps_per_bin = 20, bins = 100;  // 1 s
    std::printf("excitatory activity (spikes / 10 ms):\n");
    for (std::size_t b = 0; b < bins; ++b) {
        std::size_t count = 0;
        for (std::size_t s = 0; s < steps_per_bin; ++s) {
            net.step();
            for (auto id : net.spikes()) count += id < net.population_size(exc);
        }
        if (b % 5 == 0) {
            const int bar = static_cast<int>(60.0 * count / (0.02 * net.population_size(exc) + 1));
            std::printf("%4zu ms |%s %zu\n", b * 10, std::string(std::min(bar, 60), '#').c_str(), count);
        }
    }
    for (auto [id, name] : {std::pair{exc, "exc"}, {inh, "inh"}, {lif, "LIF"}, {adex, "AdEx"},
                            {hh, "Hodgkin-Huxley"}, {burst, "Izhikevich CH"}})
        std::printf("%-15s %6.1f Hz\n", name, net.firing_rate(id));
    return 0;
}
