#include "Layer.h"

#include <iostream>
#include <vector>
#include <algorithm>

/* LAYER */

/*
* Default Constructor
*/
Layer::Layer() : m_activation_type(ActivationType::SIGMOID) { }

/*
* Layer Constructor
*/
Layer::Layer(int n_neurons, int n_weights, ActivationType activation_type) 
    : m_activation_type(activation_type) {
    this->initNeurons(n_neurons, n_weights, activation_type);
}

/*
* Layer Destructor
*/
Layer::~Layer() { }

/*
* Initialize neurons with specified activation type
*/
void Layer::initNeurons(int n_neurons, int n_weights, ActivationType activation_type) {
    m_neurons.reserve(n_neurons);  // Pre-allocate for better performance
    for (int n = 0; n < n_neurons; n++) {
        m_neurons.push_back(Neuron(n_weights, activation_type));
    }
}

/*
* Set activation type for all neurons in this layer
*/
void Layer::set_activation_type(ActivationType type) {
    m_activation_type = type;
    for (auto& neuron : m_neurons) {
        neuron.set_activation_type(type);
    }
}

/*
* Get the activation type of this layer
*/
ActivationType Layer::get_activation_type() const {
    // If there are neurons, return the type of the first one (they should all be the same)
    if (!m_neurons.empty()) {
        return m_neurons[0].get_activation_type();
    }
    return m_activation_type;
}
