#include "nn/viz.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <queue>
#include <sstream>

#include "nn/layers/conv.hpp"
#include "nn/layers/dense.hpp"
#include "nn/layers/recurrent.hpp"
#include "nn/network.hpp"
#include "nn/spiking.hpp"

namespace nn::viz {

namespace {

// Minimal JSON helpers (compact numbers, NaN -> null).
void num(std::string& out, double v) {
    if (!std::isfinite(v)) {
        out += "null";
        return;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4g", v);
    out += buf;
}
void str(std::string& out, const std::string& s) {
    out += '"';
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) continue;
        out += c;
    }
    out += '"';
}
template <class T>
void int_array(std::string& out, const std::vector<T>& v) {
    out += '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ',';
        out += std::to_string(v[i]);
    }
    out += ']';
}

const char* type_name(LayerType t) {
    switch (t) {
        case LayerType::Dense: return "dense";
        case LayerType::Dropout: return "dropout";
        case LayerType::RBF: return "rbf";
        case LayerType::SimpleRNN: return "rnn";
        case LayerType::LSTM: return "lstm";
        case LayerType::GRU: return "gru";
        case LayerType::Conv2D: return "conv2d";
        case LayerType::MaxPool2D: return "maxpool2d";
        case LayerType::LayerNorm: return "layernorm";
    }
    return "layer";
}

std::vector<std::uint32_t> pick(std::size_t n, std::size_t cap) {
    std::vector<std::uint32_t> v;
    const std::size_t m = std::min(n, cap);
    v.reserve(m);
    for (std::size_t k = 0; k < m; ++k) v.push_back(static_cast<std::uint32_t>(k * n / m));
    return v;
}

bool write_file(const std::string& path, const std::string& s) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << s;
    return static_cast<bool>(f);
}

}  // namespace

// ------------------------------------------------------------------ NetworkRecorder

NetworkRecorder::NetworkRecorder(Network& net, RecordOptions opt) : m_net(net), m_opt(opt) {
    if (net.size() == 0) throw std::invalid_argument("viz: network has no layers");
    Layer in;
    in.name = "Input";
    in.type = "input";
    nn::Layer& first = net.layer(0);
    if (auto* c = dynamic_cast<Conv2D*>(&first)) {
        const Shape3 s = c->input_shape();
        in.shape = {s.channels, s.height, s.width};
    } else if (auto* p = dynamic_cast<MaxPool2D*>(&first)) {
        const Shape3 s = p->input_shape();
        in.shape = {s.channels, s.height, s.width};
    } else if (auto* r = dynamic_cast<RecurrentBase*>(&first)) {
        in.shape = {r->timesteps(), r->features()};
    } else {
        in.shape = {net.input_dim()};
    }
    in.shown = pick(net.input_dim(), m_opt.max_neurons_per_layer);
    m_layers.push_back(in);

    for (std::size_t i = 0; i < net.size(); ++i) {
        nn::Layer& l = net.layer(i);
        Layer v;
        v.name = l.name();
        v.type = type_name(l.type());
        if (auto* c = dynamic_cast<Conv2D*>(&l)) {
            const Shape3 s = c->output_shape();
            v.shape = {s.channels, s.height, s.width};
        } else if (auto* p = dynamic_cast<MaxPool2D*>(&l)) {
            const Shape3 s = p->output_shape();
            v.shape = {s.channels, s.height, s.width};
        } else if (auto* r = dynamic_cast<RecurrentBase*>(&l); r && r->return_sequences()) {
            v.shape = {r->timesteps(), r->units()};
        } else {
            v.shape = {l.output_dim()};
        }
        v.shown = pick(l.output_dim(), m_opt.max_neurons_per_layer);
        m_layers.push_back(v);
    }
}

void NetworkRecorder::snapshot(const Matrix& x, const Matrix* y, const std::string& label, float loss,
                               float accuracy) {
    const std::size_t rows = std::min(m_opt.samples, x.rows());
    if (rows == 0) return;
    Matrix xs(rows, x.cols());
    std::copy(x.data(), x.data() + rows * x.cols(), xs.data());
    m_net.forward(xs, false);

    std::string o = "{\"label\":";
    str(o, label);
    o += ",\"loss\":";
    num(o, loss);
    o += ",\"accuracy\":";
    num(o, accuracy);

    // strongest connections of every Dense layer (between displayed neurons)
    o += ",\"edges\":[";
    for (std::size_t i = 0; i < m_net.size(); ++i) {
        if (i) o += ',';
        o += '[';
        auto* d = dynamic_cast<Dense*>(&m_net.layer(i));
        if (d) {
            const Layer& src = m_layers[i];
            const Layer& dst = m_layers[i + 1];
            const Matrix& W = d->weights();  // [in x out]
            struct E { float a; std::uint32_t s, t; float w; };
            auto cmp = [](const E& p, const E& q) { return p.a > q.a; };  // min-heap on |w|
            std::priority_queue<E, std::vector<E>, decltype(cmp)> heap(cmp);
            for (std::uint32_t si = 0; si < src.shown.size(); ++si)
                for (std::uint32_t ti = 0; ti < dst.shown.size(); ++ti) {
                    const float w = W(src.shown[si], dst.shown[ti]);
                    const float a = std::fabs(w);
                    if (heap.size() < m_opt.max_edges_per_layer) heap.push({a, si, ti, w});
                    else if (a > heap.top().a) { heap.pop(); heap.push({a, si, ti, w}); }
                }
            bool first = true;
            while (!heap.empty()) {
                const E e = heap.top();
                heap.pop();
                if (!first) o += ',';
                first = false;
                o += std::to_string(e.s) + ',' + std::to_string(e.t) + ',';
                num(o, e.w);
            }
        }
        o += ']';
    }
    o += "],\"samples\":[";
    for (std::size_t r = 0; r < rows; ++r) {
        if (r) o += ',';
        o += "{\"target\":";
        if (y && r < y->rows()) {
            o += '[';
            for (std::size_t c = 0; c < y->cols(); ++c) {
                if (c) o += ',';
                num(o, (*y)(r, c));
            }
            o += ']';
        } else {
            o += "null";
        }
        o += ",\"act\":[";
        for (std::size_t li = 0; li < m_layers.size(); ++li) {
            if (li) o += ',';
            const real* a = li == 0 ? xs.row(r) : m_net.layer_output(li - 1).row(r);
            o += '[';
            const auto& shown = m_layers[li].shown;
            for (std::size_t k = 0; k < shown.size(); ++k) {
                if (k) o += ',';
                num(o, a[shown[k]]);
            }
            o += ']';
        }
        o += "]}";
    }
    o += "]}";
    m_snapshots.push_back(std::move(o));
}

std::string NetworkRecorder::to_json() const {
    std::string o = "{\"format\":\"nn-viz\",\"version\":1,\"kind\":\"network\",\"layers\":[";
    for (std::size_t i = 0; i < m_layers.size(); ++i) {
        const Layer& l = m_layers[i];
        if (i) o += ',';
        o += "{\"name\":";
        str(o, l.name);
        o += ",\"type\":";
        str(o, l.type);
        o += ",\"shape\":";
        int_array(o, l.shape);
        o += ",\"shown\":";
        std::size_t total = 1;
        for (auto s : l.shape) total *= s;
        if (l.shown.size() == total) o += "null";  // all neurons
        else int_array(o, l.shown);
        o += '}';
    }
    o += "],\"snapshots\":[";
    for (std::size_t i = 0; i < m_snapshots.size(); ++i) {
        if (i) o += ',';
        o += m_snapshots[i];
    }
    o += "]}";
    return o;
}

bool NetworkRecorder::save(const std::string& path) const { return write_file(path, to_json()); }

// ------------------------------------------------------------------ SpikeRecorder

SpikeRecorder::SpikeRecorder(spiking::Network& net, std::size_t max_neurons, std::size_t max_synapses,
                             std::uint64_t seed)
    : m_net(net) {
    net.finalize();
    const std::size_t total = net.neuron_count();
    m_slot.assign(total, -1);
    // proportional, evenly spaced subsample of every population
    for (spiking::PopId p = 0; p < net.population_count(); ++p) {
        const std::size_t n = net.population_size(p), off = net.first_neuron(p);
        std::size_t take = total <= max_neurons ? n : std::max<std::size_t>(1, n * max_neurons / total);
        take = std::min(take, n);
        for (std::size_t k = 0; k < take; ++k) {
            const std::uint32_t id = static_cast<std::uint32_t>(off + k * n / take);
            m_slot[id] = static_cast<std::int32_t>(m_shown.size());
            m_shown.push_back(id);
            m_pop_of.push_back(p);
        }
    }
    // reservoir sample of synapses between displayed neurons
    Rng rng(seed);
    const auto& row = net.synapse_offsets();
    const auto& post = net.synapse_targets();
    const auto& w = net.weights();
    std::size_t seen = 0;
    for (std::uint32_t s : m_shown) {
        for (std::uint32_t k = row[s]; k < row[s + 1]; ++k) {
            if (m_slot[post[k]] < 0) continue;
            ++seen;
            const std::uint32_t a = static_cast<std::uint32_t>(m_slot[s]);
            const std::uint32_t b = static_cast<std::uint32_t>(m_slot[post[k]]);
            if (m_edge_w.size() < max_synapses) {
                m_edges.push_back(a);
                m_edges.push_back(b);
                m_edge_w.push_back(w[k]);
            } else {
                const std::size_t j = rng.below(seen);
                if (j < max_synapses) {
                    m_edges[2 * j] = a;
                    m_edges[2 * j + 1] = b;
                    m_edge_w[j] = w[k];
                }
            }
        }
    }
}

void SpikeRecorder::record_step() {
    std::vector<std::uint32_t> f;
    for (std::uint32_t id : m_net.spikes())
        if (m_slot[id] >= 0) f.push_back(static_cast<std::uint32_t>(m_slot[id]));
    m_frames.push_back(std::move(f));
}

std::string SpikeRecorder::to_json() const {
    static const char* models[] = {"LIF", "Izhikevich", "AdEx", "HodgkinHuxley"};
    std::string o = "{\"format\":\"nn-viz\",\"version\":1,\"kind\":\"spiking\",\"dt\":";
    num(o, m_net.dt());
    o += ",\"populations\":[";
    for (spiking::PopId p = 0; p < m_net.population_count(); ++p) {
        if (p) o += ',';
        o += "{\"name\":";
        str(o, m_net.population_name(p));
        o += ",\"model\":";
        str(o, models[static_cast<int>(m_net.population_model(p))]);
        o += ",\"size\":" + std::to_string(m_net.population_size(p)) + '}';
    }
    o += "],\"pop\":";
    int_array(o, m_pop_of);
    o += ",\"edges\":";
    int_array(o, m_edges);
    o += ",\"weights\":[";
    for (std::size_t i = 0; i < m_edge_w.size(); ++i) {
        if (i) o += ',';
        num(o, m_edge_w[i]);
    }
    o += "],\"frames\":[";
    for (std::size_t i = 0; i < m_frames.size(); ++i) {
        if (i) o += ',';
        int_array(o, m_frames[i]);
    }
    o += "]}";
    return o;
}

bool SpikeRecorder::save(const std::string& path) const { return write_file(path, to_json()); }

}  // namespace nn::viz
