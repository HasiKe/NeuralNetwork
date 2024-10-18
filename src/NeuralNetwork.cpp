#include "NeuralNetwork.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <random>

// Network Implementation

Network::Network() : m_nLayers(0) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
}

Network::~Network() = default;

void Network::initializeNetwork(int nInputs, int nHidden, int nOutputs) {
    addLayer(nHidden, nInputs + 1);
    addLayer(nOutputs, nHidden + 1);
}

void Network::addLayer(int nNeurons, int nWeights) {
    m_layers.emplace_back(nNeurons, nWeights);
    ++m_nLayers;
}

std::vector<float> Network::forwardPropagate(const std::vector<float>& inputs) {
    std::vector<float> layerInputs = inputs;
    for (auto& layer : m_layers) {
        std::vector<float> layerOutputs;
        layerOutputs.reserve(layer.getNeurons().size());
        for (auto& neuron : layer.getNeurons()) {
            neuron.activate(layerInputs);
            neuron.transfer();
            layerOutputs.push_back(neuron.getOutput());
        }
        layerInputs = std::move(layerOutputs);
    }
    return layerInputs;
}

void Network::backwardPropagateError(const std::vector<float>& expected) {
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        Layer& layer = *it;
        std::vector<float> errors;
        if (it == m_layers.rbegin()) {
            for (size_t i = 0; i < layer.getNeurons().size(); ++i) {
                errors.push_back(expected[i] - layer.getNeurons()[i].getOutput());
            }
        } else {
            const Layer& nextLayer = *(it - 1);
            for (const auto& neuron : layer.getNeurons()) {
                float error = 0.0f;
                for (size_t j = 0; j < nextLayer.getNeurons().size(); ++j) {
                    error += nextLayer.getNeurons()[j].getWeights()[neuron.getIndex()] * nextLayer.getNeurons()[j].getDelta();
                }
                errors.push_back(error);
            }
        }
        for (size_t i = 0; i < layer.getNeurons().size(); ++i) {
            layer.getNeurons()[i].setDelta(errors[i] * layer.getNeurons()[i].transferDerivative());
        }
    }
}

void Network::updateWeights(const std::vector<float>& inputs, float learningRate) {
    std::vector<float> layerInputs = inputs;
    for (auto& layer : m_layers) {
        for (auto& neuron : layer.getNeurons()) {
            for (size_t i = 0; i < layerInputs.size(); ++i) {
                neuron.updateWeight(i, learningRate * neuron.getDelta() * layerInputs[i]);
            }
            neuron.updateWeight(layerInputs.size(), learningRate * neuron.getDelta()); // Update bias
        }
        layerInputs.clear();
        layerInputs.reserve(layer.getNeurons().size());
        for (const auto& neuron : layer.getNeurons()) {
            layerInputs.push_back(neuron.getOutput());
        }
    }
}

void Network::train(const std::vector<std::vector<float>>& trainingData, float learningRate, size_t nEpochs, size_t nOutputs) {
    for (size_t epoch = 0; epoch < nEpochs; ++epoch) {
        float sumError = 0.0f;
        for (const auto& row : trainingData) {
            std::vector<float> outputs = forwardPropagate(std::vector<float>(row.begin(), row.end() - 1));
            std::vector<float> expected(nOutputs, 0.0f);
            expected[static_cast<int>(row.back())] = 1.0f;
            for (size_t i = 0; i < nOutputs; ++i) {
                sumError += std::pow(expected[i] - outputs[i], 2);
            }
            backwardPropagateError(expected);
            updateWeights(row, learningRate);
        }
        std::cout << "[>] epoch=" << epoch << ", learning_rate=" << learningRate << ", error=" << sumError << std::endl;
    }
}

int Network::predict(const std::vector<float>& input) {
    const auto outputs = forwardPropagate(input);
    return static_cast<int>(std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end())));
}

void Network::displayHuman() const {
    std::cout << "[Network] (Layers: " << m_nLayers << ")" << std::endl << "{" << std::endl;
    for (size_t l = 0; l < m_layers.size(); ++l) {
        std::cout << "\t (Layer " << l << "): {";
        const auto& neurons = m_layers[l].getNeurons();
        for (size_t i = 0; i < neurons.size(); ++i) {
            const auto& neuron = neurons[i];
            std::cout << "<(Neuron " << i << "): [ weights={";
            const auto& weights = neuron.getWeights();
            for (size_t w = 0; w < weights.size(); ++w) {
                std::cout << weights[w];
                if (w < weights.size() - 1) std::cout << ", ";
            }
            std::cout << "}, output=" << neuron.getOutput() 
                      << ", activation=" << neuron.getActivation() 
                      << ", delta=" << neuron.getDelta() << "]>";
            if (i < neurons.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << (l < m_layers.size() - 1 ? ", " : "") << std::endl;
    }
    std::cout << "}" << std::endl;
}

bool Network::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file '" << filename << "' for saving the network." << std::endl;
        return false;
    }

    file << m_nLayers << std::endl;
    for (const auto& layer : m_layers) {
        saveLayer(layer, file);
    }

    return true;
}

bool Network::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file '" << filename << "' for loading the network." << std::endl;
        return false;
    }

    file >> m_nLayers;
    m_layers.clear();
    m_layers.reserve(m_nLayers);

    for (size_t i = 0; i < m_nLayers; ++i) {
        Layer layer;
        loadLayer(layer, file);
        m_layers.push_back(std::move(layer));
    }

    return true;
}

void Network::saveLayer(const Layer& layer, std::ofstream& file) const {
    const auto& neurons = layer.getNeurons();
    file << neurons.size() << std::endl;

    for (const auto& neuron : neurons) {
        const auto& weights = neuron.getWeights();
        file << weights.size() << std::endl;
        for (float weight : weights) {
            file << weight << " ";
        }
        file << std::endl;
    }
}

void Network::loadLayer(Layer& layer, std::ifstream& file) {
    size_t nNeurons;
    file >> nNeurons;

    std::vector<Neuron> neurons;
    neurons.reserve(nNeurons);

    for (size_t i = 0; i < nNeurons; ++i) {
        Neuron neuron;
        loadWeights(neuron, file);
        neurons.push_back(std::move(neuron));
    }

    layer.setNeurons(std::move(neurons));
}

void Network::loadWeights(Neuron& neuron, std::ifstream& file) {
    size_t nWeights;
    file >> nWeights;

    std::vector<float> weights(nWeights);
    for (size_t i = 0; i < nWeights; ++i) {
        file >> weights[i];
    }

    neuron.setWeights(std::move(weights));
}
