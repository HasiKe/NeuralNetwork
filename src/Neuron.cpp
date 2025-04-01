#include "Neuron.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>

/*
* Default Constructor
*/
Neuron::Neuron() : 
    m_nWeights(0), 
    m_activation(0.0f), 
    m_output(0.0f), 
    m_delta(0.0f),
    m_activation_type(ActivationType::SIGMOID) {
    setupActivationFunctions();
}

/*
* Neuron Constructor
*/
Neuron::Neuron(int n_weights, ActivationType activation_type) : 
    m_nWeights(n_weights),
    m_activation(0.0f),
    m_output(0.0f),
    m_delta(0.0f),
    m_activation_type(activation_type) {
    this->initWeights(n_weights);
    setupActivationFunctions();
}

/*
* Neuron Destructor
*/
Neuron::~Neuron() {
    // Nothing to clean up
}

/*
* Initialize weights using a better random number generator
*/
void Neuron::initWeights(int n_weights) {
    // Verwende Xavier/Glorot-Initialisierung für bessere Konvergenz
    std::random_device rd;
    std::mt19937 gen(rd());
    // Skaliere die Gewichte basierend auf der Anzahl der Ein- und Ausgänge
    // Die Standardabweichung sollte sqrt(2 / (fan_in + fan_out)) sein
    float scale = std::sqrt(2.0f / (n_weights * 2.0f));
    std::normal_distribution<float> dis(0.0f, scale);
    
    m_weights.reserve(n_weights);  // Pre-allocate memory for better performance
    for (int w = 0; w < n_weights; w++) {
        m_weights.push_back(dis(gen));
    }
}

/*
* Set up activation function and its derivative based on activation type
*/
void Neuron::setupActivationFunctions() {
    switch (m_activation_type) {
        case ActivationType::SIGMOID:
            m_activation_function = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
            m_derivative_function = [this](float) { return m_output * (1.0f - m_output); };
            break;
            
        case ActivationType::RELU:
            m_activation_function = [](float x) { return std::max(0.0f, x); };
            m_derivative_function = [](float x) { return x > 0.0f ? 1.0f : 0.0f; };
            break;
            
        case ActivationType::LEAKY_RELU:
            m_activation_function = [](float x) { return x > 0.0f ? x : 0.01f * x; };
            m_derivative_function = [](float x) { return x > 0.0f ? 1.0f : 0.01f; };
            break;
            
        case ActivationType::TANH:
            m_activation_function = [](float x) { return std::tanh(x); };
            m_derivative_function = [this](float) { return 1.0f - m_output * m_output; };
            break;
            
        case ActivationType::ELU:
            m_activation_function = [](float x) { return x > 0.0f ? x : 0.01f * (std::exp(x) - 1.0f); };
            m_derivative_function = [](float x) { return x > 0.0f ? 1.0f : 0.01f * std::exp(x); };
            break;
            
        default:
            // Default to sigmoid
            m_activation_function = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
            m_derivative_function = [this](float) { return m_output * (1.0f - m_output); };
            break;
    }
}

/*
* Set activation type and update functions
*/
void Neuron::set_activation_type(ActivationType type) {
    m_activation_type = type;
    setupActivationFunctions();
}

/*
* Return string representation of activation function
*/
std::string Neuron::get_activation_name() const {
    switch (m_activation_type) {
        case ActivationType::SIGMOID: return "SIGMOID";
        case ActivationType::RELU: return "RELU";
        case ActivationType::LEAKY_RELU: return "LEAKY_RELU";
        case ActivationType::TANH: return "TANH";
        case ActivationType::ELU: return "ELU";
        default: return "UNKNOWN";
    }
}

/*
* Activate neuron with optimized code
*/
void Neuron::activate(const std::vector<float>& inputs) {
    // The last weight is assumed to be the bias
    m_activation = m_weights[m_nWeights - 1];

    // Accumulate weighted inputs with better performance
    for (size_t i = 0; i < m_nWeights - 1; i++) {
        m_activation += m_weights[i] * inputs[i];
    }
}

/*
* Apply activation function
*/
void Neuron::transfer() {
    m_output = m_activation_function(m_activation);
}

/*
* Calculate derivative of activation function
*/
float Neuron::transfer_derivative() const {
    return m_derivative_function(m_activation);
}
