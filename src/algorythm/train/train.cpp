#include "NeuralNetwork.h"
#include "train.h"
#include <cmath>
#include <iostream>
#include <algorithm>

// Implementierung der train-Funktion
void train(Network* network, const std::vector<std::vector<float>>& trainings_data, float l_rate, size_t n_epoch, size_t n_outputs, TrainingAlgorithm algorithm) {
    switch (algorithm) {
        case TrainingAlgorithm::SGD:
            train_sgd(network, trainings_data, l_rate, n_epoch, n_outputs);
            break;
        case TrainingAlgorithm::Momentum:
            train_momentum(network, trainings_data, l_rate, n_epoch, n_outputs);
            break;
        case TrainingAlgorithm::Adam:
            train_adam(network, trainings_data, l_rate, n_epoch, n_outputs);
            break;
        default:
            std::cerr << "Unbekannter Trainingsalgorithmus" << std::endl;
            break;
    }
}

// Implementierung der Hilfsfunktionen
void Network::backward_propagate_error(std::vector<float> expected) {
    // Reverse traverse the layers
    for (size_t i = m_nLayers; i-- > 0;) {
        std::vector<Neuron>& layer_neurons = m_layers[i].get_neurons();

        for (size_t n = 0; n < layer_neurons.size(); n++) {
            float error = 0.0f;
            if (i == m_nLayers - 1) {
                error = expected[n] - layer_neurons[n].get_output();
            } else {
                for (auto& neu : m_layers[i + 1].get_neurons()) {
                    error += (neu.get_weights()[n] * neu.get_delta());
                }
            }
            layer_neurons[n].set_delta(error * layer_neurons[n].transfer_derivative());
        }
    }
}

