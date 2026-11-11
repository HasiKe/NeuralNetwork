#pragma once

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "matrix.hpp"

/// Recording of networks for the 3D viewer (viz/viewer.html).
///
///   nn::viz::NetworkRecorder rec(net);
///   opts.on_epoch_end = [&](size_t e, float loss, float) {
///       rec.snapshot(x, &y, "epoch " + std::to_string(e + 1), loss);
///       return true;
///   };
///   net.fit(x, y, opts);
///   rec.save("model.viz.json");   // open in viz/viewer.html
///
/// The file contains the layer structure, the strongest connections, and the
/// activation of every neuron for a few samples at each snapshot, so the
/// viewer can replay how signals flow and how training changes the network.
namespace nn {
class Network;
namespace spiking { class Network; }
}  // namespace nn

namespace nn::viz {

struct RecordOptions {
    std::size_t samples = 8;                ///< input rows captured per snapshot
    std::size_t max_neurons_per_layer = 4096;  ///< larger layers are subsampled evenly
    std::size_t max_edges_per_layer = 800;  ///< strongest |w| connections kept per Dense layer
};

class NetworkRecorder {
public:
    explicit NetworkRecorder(Network& net, RecordOptions opt = {});

    /// Run a forward pass on the first `samples` rows of x and store the
    /// activations of every layer plus the current connection weights.
    /// `y` (targets, optional) is stored for display.
    void snapshot(const Matrix& x, const Matrix* y, const std::string& label,
                  float loss = std::numeric_limits<float>::quiet_NaN(),
                  float accuracy = std::numeric_limits<float>::quiet_NaN());

    std::size_t snapshot_count() const { return m_snapshots.size(); }
    std::string to_json() const;
    bool save(const std::string& path) const;

private:
    struct Layer {
        std::string name, type;
        std::vector<std::size_t> shape;  // [n] | [C,H,W] | [T,F]
        std::vector<std::uint32_t> shown;  // displayed neuron indices
    };
    Network& m_net;
    RecordOptions m_opt;
    std::vector<Layer> m_layers;  // index 0 = input
    std::vector<std::string> m_snapshots;  // pre-serialised JSON objects
};

/// Records spikes of a spiking::Network step by step.
class SpikeRecorder {
public:
    /// Call after building (and finalize()) the network.
    explicit SpikeRecorder(spiking::Network& net, std::size_t max_neurons = 3000,
                           std::size_t max_synapses = 4000, std::uint64_t seed = 1);
    /// Store the spikes of the most recent step (call after every net.step()).
    void record_step();
    std::string to_json() const;
    bool save(const std::string& path) const;

private:
    spiking::Network& m_net;
    std::vector<std::int32_t> m_slot;     // global neuron id -> displayed index or -1
    std::vector<std::uint32_t> m_shown;   // displayed index -> global id
    std::vector<std::uint32_t> m_pop_of;  // displayed index -> population
    std::vector<std::uint32_t> m_edges;   // pairs (src, dst) of displayed indices
    std::vector<float> m_edge_w;
    std::vector<std::vector<std::uint32_t>> m_frames;  // displayed indices spiking per step
};

}  // namespace nn::viz
