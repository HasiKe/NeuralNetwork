#include "algorythm/train/train.h"
#include <cmath>
#include <iostream>
#include <algorithm>


// Implementierung der Adam-Trainingsmethode
void train_adam(Network* network, const std::vector<std::vector<float>>& trainings_data, float l_rate, size_t n_epoch, size_t n_outputs) {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    // Initialisiere m und v für jedes Neuron
    for (auto& layer : network->get_layers()) {
        for (auto& neuron : layer.get_neurons()) {
            neuron.m.resize(neuron.get_weights().size(), 0.0f);
            neuron.v.resize(neuron.get_weights().size(), 0.0f);
            neuron.t = 0;
        }
    }

    for (size_t e = 0; e < n_epoch; e++) {
        float sum_error = 0.0f;

        for (const auto& row : trainings_data) {
            std::vector<float> outputs = network->forward_propagate(row);
            std::vector<float> expected(n_outputs, 0.0f);
            expected[static_cast<int>(row.back())] = 1.0f;
            for (size_t x = 0; x < n_outputs; x++) {
                sum_error += std::pow((expected[x] - outputs[x]), 2);
            }
            backward_propagate_error(network, expected);
            update_weights_adam(network, row, l_rate, beta1, beta2, epsilon);
        }
        std::cout << "[Adam] epoch=" << e << ", l_rate=" << l_rate << ", error=" << sum_error << std::endl;
    }
}

void update_weights_adam(Network* network, const std::vector<float>& inputs, float l_rate, float beta1, float beta2, float epsilon) {
    std::vector<float> new_inputs = inputs;
    for (size_t i = 0; i < network->get_layers().size(); i++) {
        if (i != 0) {
            new_inputs.clear();
            for (const auto& neuron : network->get_layers()[i - 1].get_neurons()) {
                new_inputs.push_back(neuron.get_output());
            }
        }
        for (auto& neuron : network->get_layers()[i].get_neurons()) {
            neuron.t += 1;
            for (size_t j = 0; j < new_inputs.size(); j++) {
                float gradient = neuron.get_delta() * new_inputs[j];

                // Update biased first moment estimate
                neuron.m[j] = beta1 * neuron.m[j] + (1 - beta1) * gradient;
                // Update biased second raw moment estimate
                neuron.v[j] = beta2 * neuron.v[j] + (1 - beta2) * gradient * gradient;

                // Compute bias-corrected first and second moment estimates
                float m_hat = neuron.m[j] / (1 - std::pow(beta1, neuron.t));
                float v_hat = neuron.v[j] / (1 - std::pow(beta2, neuron.t));

                // Update weights
                neuron.get_weights()[j] += l_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
            // Bias
            size_t bias_index = neuron.get_weights().size() - 1;
            float gradient = neuron.get_delta();

            neuron.m[bias_index] = beta1 * neuron.m[bias_index] + (1 - beta1) * gradient;
            neuron.v[bias_index] = beta2 * neuron.v[bias_index] + (1 - beta2) * gradient * gradient;

            float m_hat = neuron.m[bias_index] / (1 - std::pow(beta1, neuron.t));
            float v_hat = neuron.v[bias_index] / (1 - std::pow(beta2, neuron.t));

            neuron.get_weights()[bias_index] += l_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
}