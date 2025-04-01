#pragma once

#include <iostream>
#include <vector>
#include "Neuron.h"

class Layer {
public:
    Layer();
    Layer(int n_neurons, int n_weights, ActivationType activation_type = ActivationType::SIGMOID);
    ~Layer();

    // Return mutable reference to the neurons
    std::vector<Neuron>& get_neurons() { return m_neurons; }

    // Return const reference to the neurons (const version)
    const std::vector<Neuron>& get_neurons() const { return m_neurons; }

    // Set the neurons (used during loading)
    void set_neurons(const std::vector<Neuron>& neurons) { m_neurons = neurons; }
    
    // Set activation type for all neurons in this layer
    void set_activation_type(ActivationType type);
    
    // Get current activation type
    ActivationType get_activation_type() const;

private:
    void initNeurons(int n_neurons, int n_weights, ActivationType activation_type);

    std::vector<Neuron> m_neurons;
    ActivationType m_activation_type;
};

