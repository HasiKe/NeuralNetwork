#include "nn/layer.hpp"

#include <stdexcept>

#include "nn/layers/conv.hpp"
#include "nn/layers/dense.hpp"
#include "nn/layers/dropout.hpp"
#include "nn/layers/layernorm.hpp"
#include "nn/layers/rbf.hpp"
#include "nn/layers/recurrent.hpp"

namespace nn {

std::size_t Layer::parameter_count() {
    std::vector<ParamRef> p;
    parameters(p);
    std::size_t n = 0;
    for (const auto& r : p) n += r.size;
    return n;
}

LayerPtr make_layer(LayerType t) {
    switch (t) {
        case LayerType::Dense: return std::make_unique<Dense>();
        case LayerType::Dropout: return std::make_unique<Dropout>();
        case LayerType::RBF: return std::make_unique<RBF>();
        case LayerType::SimpleRNN: return std::make_unique<SimpleRNN>();
        case LayerType::LSTM: return std::make_unique<LSTM>();
        case LayerType::GRU: return std::make_unique<GRU>();
        case LayerType::Conv2D: return std::make_unique<Conv2D>();
        case LayerType::MaxPool2D: return std::make_unique<MaxPool2D>();
        case LayerType::LayerNorm: return std::make_unique<LayerNorm>();
    }
    throw std::runtime_error("nn: unknown layer type in model file");
}

}  // namespace nn
