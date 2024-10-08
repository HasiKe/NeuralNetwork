#include "algorythm/train/train.h"
#include <cmath>
#include <iostream>
#include <algorithm>


// Implementierung der Momentum-Trainingsmethode
void train_momentum(Network* network, const std::vector<std::vector<float>>& trainings_data, float l_rate, size_t n_epoch, size_t n_outputs) {
    float momentum = 0.9f; // Momentum-Faktor

    // Initialisiere die Velocity für jedes Neuron
    for (auto& layer : network->get_layers()) {
        for (auto& neuron : layer.get_neurons()) {
            neuron.velocity.resize(neuron.get_weights().size(), 0.0f);
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
            update_weights_momentum(network, row, l_rate, momentum);
        }
        std::cout << "[Momentum] epoch=" << e << ", l_rate=" << l_rate << ", error=" << sum_error << std::endl;
    }
}

void update_weights_momentum(Network* network, const std::vector<float>& inputs, float l_rate, float momentum) {
    std::vector<float> new_inputs = inputs;
    for (size_t i = 0; i < network->get_layers().size(); i++) {
        if (i != 0) {
            new_inputs.clear();
            for (const auto& neuron : network->get_layers()[i - 1].get_neurons()) {
                new_inputs.push_back(neuron.get_output());
            }
        }
        for (auto& neuron : network->get_layers()[i].get_neurons()) {
            for (size_t j = 0; j < new_inputs.size(); j++) {
                float delta_weight = l_rate * neuron.get_delta() * new_inputs[j];
                neuron.velocity[j] = momentum * neuron.velocity[j] + delta_weight;
                neuron.get_weights()[j] += neuron.velocity[j];
            }
            // Bias
            float delta_weight = l_rate * neuron.get_delta();
            size_t bias_index = neuron.get_weights().size() - 1;
            neuron.velocity[bias_index] = momentum * neuron.velocity[bias_index] + delta_weight;
            neuron.get_weights()[bias_index] += neuron.velocity[bias_index];
        }
    }
}