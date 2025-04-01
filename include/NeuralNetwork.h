#pragma once

#include <iostream>
#include <vector>
#include <string>
#include "Layer.h"

class Network {
public:
    Network();
    ~Network();

    // Initialize a neural network with specified parameters
    void initialize_network(int n_inputs, int n_hidden, int n_outputs, 
                           ActivationType hidden_activation = ActivationType::SIGMOID,
                           ActivationType output_activation = ActivationType::SIGMOID);

    // Add a new layer with specified activation type
    void add_layer(int n_neurons, int n_weights, ActivationType activation_type = ActivationType::SIGMOID);
    
    // Set activation type for a specific layer
    void set_activation_type(size_t layer_index, ActivationType type);
    
    // Forward propagation
    std::vector<float> forward_propagate(const std::vector<float>& inputs);
    
    // Backward propagation
    void backward_propagate_error(const std::vector<float>& expected);
    
    // Update weights with learning rate
    void update_weights(const std::vector<float>& inputs, float l_rate);

    // Train the network
    void train(const std::vector<std::vector<float>>& trainings_data, float l_rate, 
               size_t n_epoch, size_t n_outputs, bool show_progress = true);
               
    // Make a prediction
    int predict(const std::vector<float>& input);

    // Display network information
    void display_human();

    // Save and load functionality
    bool save(const std::string& filename);
    bool load(const std::string& filename);

private:
    size_t m_nLayers;
    std::vector<Layer> m_layers;

    // Helper methods for saving and loading
    void save_layer(const Layer& layer, std::ofstream& file);
    void load_layer(Layer& layer, std::ifstream& file);
    void load_weights(Neuron& neuron, std::ifstream& file);
    
    // Helper method to validate file streams
    bool validate_file_stream(const std::ios& stream, const std::string& filename, bool is_loading);
};

