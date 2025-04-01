#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <memory>

enum class ActivationType {
    SIGMOID,
    RELU,
    LEAKY_RELU,
    TANH,
    ELU
};

class Neuron {
public:
    Neuron();
    Neuron(int n_weights, ActivationType activation_type = ActivationType::SIGMOID);
    ~Neuron();

    void activate(const std::vector<float>& inputs);
    void transfer();
    float transfer_derivative() const;

    // Return mutable reference to the neuron weights
    std::vector<float>& get_weights() { return m_weights; }

    // Return const reference to the neuron weights (const version)
    const std::vector<float>& get_weights() const { return m_weights; }

    float get_output() const { return m_output; }
    float get_activation() const { return m_activation; }
    float get_delta() const { return m_delta; }
    ActivationType get_activation_type() const { return m_activation_type; }

    void set_delta(float delta) { m_delta = delta; }
    void set_activation_type(ActivationType type);

    // Set the weights (used during loading)
    void set_weights(const std::vector<float>& weights) { m_weights = weights; m_nWeights = weights.size(); }

    // Get string representation of activation function
    std::string get_activation_name() const;

private:
    size_t m_nWeights;
    std::vector<float> m_weights;
    float m_activation;
    float m_output;
    float m_delta;
    ActivationType m_activation_type;

    // Function pointers for activation and derivative
    std::function<float(float)> m_activation_function;
    std::function<float(float)> m_derivative_function;

private:
    void initWeights(int n_weights);
    void setupActivationFunctions();
};
