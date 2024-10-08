#include "algorythm/train/train.h"
#include <cmath>
#include <iostream>
#include <algorithm>

// Implementierung der SGD-Trainingsmethode
void train_sgd(Network* network, const std::vector<std::vector<float>>& trainings_data, float l_rate, size_t n_epoch, size_t n_outputs) {
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
            update_weights_sgd(network, row, l_rate);
        }
        std::cout << "[SGD] epoch=" << e << ", l_rate=" << l_rate << ", error=" << sum_error << std::endl;
    }
}

void update_weights_sgd(Network* network, const std::vector<float>& inputs, float l_rate) {
    std::vector<float> new_inputs = inputs;

    // Durchlaufe jede Schicht des Netzwerks
    for (size_t i = 0; i < network->get_layers().size(); ++i) {
        Layer& layer = network->get_layers()[i];
        std::vector<float> inputs_for_layer = new_inputs;

        if (i != 0) {
            // Eingaben für die aktuelle Schicht sind die Ausgaben der vorherigen Schicht
            inputs_for_layer.clear();
            const std::vector<Neuron>& prev_layer_neurons = network->get_layers()[i - 1].get_neurons();
            for (const Neuron& neuron : prev_layer_neurons) {
                inputs_for_layer.push_back(neuron.get_output());
            }
        }

        // Aktualisiere die Gewichte jedes Neurons in der Schicht
        for (Neuron& neuron : layer.get_neurons()) {
            for (size_t j = 0; j < inputs_for_layer.size(); ++j) {
                // Delta-Gewicht berechnen
                float delta_weight = l_rate * neuron.get_delta() * inputs_for_layer[j];
                // Gewicht anpassen
                neuron.get_weights()[j] += delta_weight;
            }
            // Bias-Gewicht aktualisieren
            size_t bias_index = neuron.get_weights().size() - 1;
            float delta_bias = l_rate * neuron.get_delta();
            neuron.get_weights()[bias_index] += delta_bias;
        }
    }
}