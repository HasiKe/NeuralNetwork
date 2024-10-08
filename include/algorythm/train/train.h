#pragma once

#include "NeuralNetwork.h"

// Enum zur Auswahl des Trainingsalgorithmus
enum class TrainingAlgorithm {
    SGD,
    Momentum,
    Adam
};

// Deklaration der train-Funktion
void train(Network* network, const std::vector<std::vector<float>>& trainings_data, float l_rate, size_t n_epoch, size_t n_outputs, TrainingAlgorithm algorithm);

// Deklarationen der einzelnen Trainingsfunktionen
void train_momentum(Network* network, const std::vector<std::vector<float>>& trainings_data, float l_rate, size_t n_epoch, size_t n_outputs);
void train_adam(Network* network, const std::vector<std::vector<float>>& trainings_data, float l_rate, size_t n_epoch, size_t n_outputs);
void train_sgd(Network* network, const std::vector<std::vector<float>>& trainings_data, float l_rate, size_t n_epoch, size_t n_outputs);

// Weitere Hilfsfunktionen
void backward_propagate_error(Network* network, const std::vector<float>& expected);
void update_weights_momentum(Network* network, const std::vector<float>& inputs, float l_rate, float momentum);
void update_weights_adam(Network* network, const std::vector<float>& inputs, float l_rate, float beta1, float beta2, float epsilon);
void update_weights_sgd(Network* network, const std::vector<float>& inputs, float l_rate);