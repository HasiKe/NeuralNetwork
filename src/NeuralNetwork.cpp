#include "NeuralNetwork.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <iomanip> // for std::setprecision
#include <random>
#include <numeric> // for std::iota

// Helper function to convert ActivationType to string
std::string get_activation_name(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: return "SIGMOID";
        case ActivationType::RELU: return "RELU";
        case ActivationType::LEAKY_RELU: return "LEAKY_RELU";
        case ActivationType::TANH: return "TANH";
        case ActivationType::ELU: return "ELU";
        default: return "UNKNOWN";
    }
}

/* NETWORK */

/*
* Network Constructor
*/
Network::Network() : m_nLayers(0) {
    // Initialize with modern C++ random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
}

/*
* Network Destructor
*/
Network::~Network() {
    // Nothing to clean up
}

/*
* Initialize a network with specified activation functions
*/
void Network::initialize_network(int n_inputs, int n_hidden, int n_outputs, 
                                ActivationType hidden_activation, 
                                ActivationType output_activation) {
    // Add a hidden layer (n_hidden neurons connected to all inputs)
    this->add_layer(n_hidden, n_inputs + 1, hidden_activation);

    // Add an output layer (one neuron per output connected to previous layer's neurons)
    this->add_layer(n_outputs, n_hidden + 1, output_activation);
}

/*
* Add another layer to the network with specified activation type
*/
void Network::add_layer(int n_neurons, int n_weights, ActivationType activation_type) {
    m_layers.push_back(Layer(n_neurons, n_weights, activation_type));
    m_nLayers++;
}

/*
* Set activation type for a specific layer
*/
void Network::set_activation_type(size_t layer_index, ActivationType type) {
    if (layer_index < m_nLayers) {
        m_layers[layer_index].set_activation_type(type);
    } else {
        std::cerr << "Error: Layer index out of bounds. Cannot set activation type." << std::endl;
    }
}

/*
* Forward propagate an input (optimized)
*/
std::vector<float> Network::forward_propagate(const std::vector<float>& inputs) {
    std::vector<float> current_inputs = inputs;
    std::vector<float> new_inputs;
    
    for (size_t i = 0; i < m_nLayers; i++) {
        new_inputs.clear();
        new_inputs.reserve(m_layers[i].get_neurons().size());  // Pre-allocate for better performance

        // Reference the layer neurons directly
        std::vector<Neuron>& layer_neurons = m_layers[i].get_neurons();
        for (auto& neuron : layer_neurons) {
            neuron.activate(current_inputs);
            neuron.transfer();
            new_inputs.push_back(neuron.get_output());
        }
        current_inputs = new_inputs;
    }
    return current_inputs;
}

/*
* Backward propagate error (optimized with improved gradient calculation)
*/
void Network::backward_propagate_error(const std::vector<float>& expected) {
    // Process output layer error (Verbesserte Fehlerberechnung)
    size_t output_layer_idx = m_nLayers - 1;
    auto& output_neurons = m_layers[output_layer_idx].get_neurons();
    
    // Berechne Fehler für jedes Output-Neuron
    for (size_t n = 0; n < output_neurons.size(); n++) {
        // Berechne Fehler (Zielwert - tatsächlicher Output)
        float error = expected[n] - output_neurons[n].get_output();
        
        // Berechne Delta mit der Ableitung der Aktivierungsfunktion
        float derivative = output_neurons[n].transfer_derivative();
        output_neurons[n].set_delta(error * derivative);
    }
    
    // Propagiere Fehler rückwärts durch Hidden Layers
    for (size_t i = output_layer_idx; i-- > 0;) {
        auto& current_neurons = m_layers[i].get_neurons();
        auto& next_neurons = m_layers[i + 1].get_neurons();
        
        // Für jedes Neuron in der aktuellen Schicht
        for (size_t n = 0; n < current_neurons.size(); n++) {
            float error_sum = 0.0f;
            
            // Summiere gewichtete Deltas aus der nächsten Schicht
            for (const auto& next_neuron : next_neurons) {
                // Gewichtung mit dem Gewicht, das dieses Neuron mit dem nächsten verbindet
                error_sum += next_neuron.get_weights()[n] * next_neuron.get_delta();
            }
            
            // Berechne Delta für das aktuelle Neuron
            float derivative = current_neurons[n].transfer_derivative();
            current_neurons[n].set_delta(error_sum * derivative);
        }
    }
}

/*
* Update weights with momentum and adaptive learning rate
*/
void Network::update_weights(const std::vector<float>& inputs, float l_rate) {
    // Momentum-Faktor für stabileres Training
    const float momentum = 0.9f;
    
    // Speichere frühere Gewichtsänderungen für Momentum (statische Variable)
    static std::vector<std::vector<std::vector<float>>> prev_deltas;
    
    // Initialisiere Gewichtsänderungen bei der ersten Ausführung
    if (prev_deltas.empty()) {
        prev_deltas.resize(m_nLayers);
        for (size_t i = 0; i < m_nLayers; i++) {
            prev_deltas[i].resize(m_layers[i].get_neurons().size());
            for (size_t j = 0; j < m_layers[i].get_neurons().size(); j++) {
                prev_deltas[i][j].resize(m_layers[i].get_neurons()[j].get_weights().size(), 0.0f);
            }
        }
    }
    
    // Caching der Eingaben für jede Schicht
    std::vector<float> layer_inputs;
    
    // Für jede Schicht im Netzwerk
    for (size_t i = 0; i < m_nLayers; i++) {
        // Bestimme die Eingaben für diese Schicht
        if (i == 0) {
            // Für die erste Schicht, verwende die Netzwerkeingaben (ohne das erwartete Ergebnis)
            layer_inputs.assign(inputs.begin(), inputs.end() - 1);
        } else {
            // Für folgende Schichten, verwende die Ausgaben der vorherigen Schicht
            const auto& prev_neurons = m_layers[i - 1].get_neurons();
            layer_inputs.resize(prev_neurons.size());
            
            for (size_t j = 0; j < prev_neurons.size(); j++) {
                layer_inputs[j] = prev_neurons[j].get_output();
            }
        }
        
        // Aktualisiere Gewichte für jedes Neuron in der aktuellen Schicht
        auto& neurons = m_layers[i].get_neurons();
        
        // Parallelisierbare Schleife für bessere Performance
        for (size_t n = 0; n < neurons.size(); n++) {
            auto& neuron = neurons[n];
            auto& weights = neuron.get_weights();
            float delta = neuron.get_delta();
            
            // Verstärkter Lernfaktor für den Output-Layer
            float layer_factor = (i == m_nLayers - 1) ? 2.0f : 1.0f;
            
            // Adaptive Lernrate basierend auf der Schicht und Neuronpoisition
            float effective_rate = l_rate * layer_factor;
            
            // Aktualisiere Gewichte und füge Momentum hinzu
            for (size_t j = 0; j < layer_inputs.size(); j++) {
                // Berechne Gewichtsänderung mit Momentum
                float delta_weight = effective_rate * delta * layer_inputs[j];
                
                // Momentum: Füge einen Anteil der letzten Änderung hinzu
                float updated_delta = delta_weight + momentum * prev_deltas[i][n][j];
                
                // Aktualisiere Gewicht
                weights[j] += updated_delta;
                
                // Speichere die Änderung für die nächste Iteration
                prev_deltas[i][n][j] = updated_delta;
            }
            
            // Aktualisiere Bias (letztes Gewicht) auch mit Momentum
            float bias_delta = effective_rate * delta;
            float updated_bias_delta = bias_delta + momentum * prev_deltas[i][n][layer_inputs.size()];
            weights.back() += updated_bias_delta;
            prev_deltas[i][n][layer_inputs.size()] = updated_bias_delta;
        }
    }
}

/*
* Train the network with adaptive learning rate and early stopping
*/
void Network::train(const std::vector<std::vector<float>>& trainings_data, 
                    float l_rate, size_t n_epoch, size_t n_outputs, bool show_progress) {
    const size_t report_interval = std::max(size_t(1), n_epoch / 20);  // Report progress every 5%
    
    float best_error = std::numeric_limits<float>::max();
    size_t no_improvement_count = 0;
    const size_t patience = 50;  // Early stopping patience
    float min_improvement = 0.001f;  // Minimum improvement to reset patience counter
    
    // Adaptive learning rate parameters
    float initial_l_rate = l_rate;
    float min_l_rate = 0.001f;
    float l_rate_decay = 0.9f;
    
    // Für Fehlertracking
    std::vector<float> error_history;
    error_history.reserve(n_epoch);
    
    for (size_t e = 0; e < n_epoch; e++) {
        float sum_error = 0.0f;
        size_t total_samples = trainings_data.size();

        // Zufällige Reihenfolge für jede Epoche (Stochastic Gradient Descent)
        std::vector<size_t> indices(total_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Training auf dem gemischten Datensatz
        for (size_t idx : indices) {
            const auto& row = trainings_data[idx];
            
            // Forward propagate
            std::vector<float> outputs = this->forward_propagate(row);
            
            // Convert expected class to one-hot encoding
            std::vector<float> expected(n_outputs, 0.0f);
            expected[static_cast<int>(row.back())] = 1.0f;
            
            // Quadratischen Fehler berechnen (einfacher zum Debuggen)
            for (size_t x = 0; x < n_outputs; x++) {
                sum_error += 0.5f * std::pow(expected[x] - outputs[x], 2);
            }
            
            // Backpropagate error and update weights
            this->backward_propagate_error(expected);
            this->update_weights(row, l_rate);
        }
        
        // Normalisiere Fehler über Anzahl der Samples
        sum_error /= total_samples;
        error_history.push_back(sum_error);
        
        // Early stopping Logik
        if (sum_error < best_error - min_improvement) {
            best_error = sum_error;
            no_improvement_count = 0;
        } else {
            no_improvement_count++;
            
            // Wenn keine Verbesserung für 'patience' Epochen, reduziere die Lernrate
            if (no_improvement_count >= patience) {
                l_rate *= l_rate_decay;
                l_rate = std::max(l_rate, min_l_rate);
                no_improvement_count = 0;
                
                if (show_progress) {
                    std::cout << "[>] Reducing learning rate to " << l_rate << std::endl;
                }
            }
        }
        
        // Report progress periodically
        if (show_progress && (e % report_interval == 0 || e == n_epoch - 1)) {
            std::cout << "[>] epoch=" << e << "/" << n_epoch 
                      << ", l_rate=" << std::fixed << std::setprecision(6) << l_rate 
                      << ", error=" << std::setprecision(10) << sum_error << std::endl;
        }
        
        // Early stopping, wenn der Fehler sehr klein ist
        if (sum_error < 0.001f) {
            if (show_progress) {
                std::cout << "[>] Early stopping at epoch " << e << " with error " << sum_error << std::endl;
            }
            break;
        }
    }
    
    // Am Ende des Trainings, setze die Lernrate für zukünftige Trainings zurück
    l_rate = initial_l_rate;
}

/*
* Make a prediction
*/
int Network::predict(const std::vector<float>& input) {
    std::vector<float> outputs = this->forward_propagate(input);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

/*
* Display the network with improved formatting
*/
void Network::display_human() {
    std::cout << "===== Neural Network (" << m_nLayers << " layers) =====" << std::endl;

    for (size_t l = 0; l < m_layers.size(); l++) {
        const auto& layer = m_layers[l];
        // Konvertiere Enums zu String, bevor die Ausgabe erfolgt
        std::string layer_activation_name = get_activation_name(layer.get_activation_type());
        std::cout << "Layer " << l << " [" << layer_activation_name << "]:" << std::endl;
        
        for (size_t i = 0; i < layer.get_neurons().size(); i++) {
            const auto& neuron = layer.get_neurons()[i];
            std::cout << "  Neuron " << i << " [" << neuron.get_activation_name() << "]:" << std::endl;
            
            // Display weights
            std::cout << "    Weights: ";
            const auto& weights = neuron.get_weights();
            for (size_t w = 0; w < weights.size(); ++w) {
                std::cout << std::fixed << std::setprecision(4) << weights[w];
                if (w < weights.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
            
            // Display other properties
            std::cout << "    Output: " << std::fixed << std::setprecision(6) << neuron.get_output() << std::endl;
            std::cout << "    Activation: " << neuron.get_activation() << std::endl;
            std::cout << "    Delta: " << neuron.get_delta() << std::endl;
        }
        std::cout << std::endl;
    }
}

/*
* Helper method to validate file streams
*/
bool Network::validate_file_stream(const std::ios& stream, const std::string& filename, bool is_loading) {
    if (!stream.good()) {
        std::cerr << "Error: Could not " << (is_loading ? "load from" : "save to") 
                  << " file '" << filename << "'." << std::endl;
        return false;
    }
    return true;
}

/*
* Save the network to a file with activation type information
*/
bool Network::save(const std::string& filename) {
    std::ofstream file(filename);
    if (!validate_file_stream(file, filename, false)) {
        return false;
    }

    // Save the number of layers
    file << m_nLayers << std::endl;

    // Save each layer with its activation type
    for (const auto& layer : m_layers) {
        // Save the layer's activation type
        file << static_cast<int>(layer.get_activation_type()) << std::endl;
        save_layer(layer, file);
    }

    file.close();
    return file.good();
}

/*
* Load the network from a file with activation type information
*/
bool Network::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!validate_file_stream(file, filename, true)) {
        return false;
    }

    try {
        // Load the number of layers
        size_t n_layers = 0;
        file >> n_layers;
        if (!file.good()) throw std::runtime_error("Failed to read layer count");
        
        m_nLayers = n_layers;

        // Clear existing layers
        m_layers.clear();

        // Load each layer with its activation type
        for (size_t i = 0; i < m_nLayers; ++i) {
            // Read activation type
            int activation_type_int;
            file >> activation_type_int;
            if (!file.good()) throw std::runtime_error("Failed to read activation type");
            
            ActivationType activation_type = static_cast<ActivationType>(activation_type_int);
            
            // Create layer and load neurons
            Layer layer;
            load_layer(layer, file);
            
            // Set the activation type for all neurons
            layer.set_activation_type(activation_type);
            
            m_layers.push_back(layer);
        }

        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading network: " << e.what() << std::endl;
        file.close();
        return false;
    }
}

/*
* Save a layer to a file
*/
void Network::save_layer(const Layer& layer, std::ofstream& file) {
    const auto& neurons = layer.get_neurons();
    
    // Save the number of neurons
    file << neurons.size() << std::endl;

    // Save each neuron
    for (const auto& neuron : neurons) {
        const auto& weights = neuron.get_weights();

        // Save the number of weights
        file << weights.size() << std::endl;

        // Save each weight with high precision
        file << std::fixed << std::setprecision(10);
        for (const float& weight : weights) {
            file << weight << " ";
        }
        file << std::endl;
    }
}

/*
* Load a layer from a file
*/
void Network::load_layer(Layer& layer, std::ifstream& file) {
    // Load the number of neurons
    size_t n_neurons = 0;
    file >> n_neurons;
    if (!file.good()) throw std::runtime_error("Failed to read neuron count");

    // Initialize the neurons
    std::vector<Neuron> neurons;
    neurons.reserve(n_neurons);  // Pre-allocate for better performance
    
    for (size_t i = 0; i < n_neurons; ++i) {
        Neuron neuron;
        load_weights(neuron, file);
        neurons.push_back(neuron);
    }
    layer.set_neurons(neurons);
}

/*
* Load weights for a neuron from a file
*/
void Network::load_weights(Neuron& neuron, std::ifstream& file) {
    // Load the number of weights
    size_t n_weights = 0;
    file >> n_weights;
    if (!file.good()) throw std::runtime_error("Failed to read weight count");

    // Load each weight
    std::vector<float> weights(n_weights);
    for (size_t i = 0; i < n_weights; ++i) {
        if (!(file >> weights[i])) {
            throw std::runtime_error("Failed to read weight value");
        }
    }

    // Set the weights of the neuron
    neuron.set_weights(weights);
}