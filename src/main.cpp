#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <regex>
#include <iterator>
#include <map>
#include <numeric>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "NeuralNetwork.h"

// Function declarations
std::vector<std::vector<float>> load_csv_data(const std::string& filename);
float evaluate_network(Network* network, const std::vector<std::vector<float>>& dataset);
void print_usage();
ActivationType get_activation_type_from_name(const std::string& name);
std::string get_activation_name(ActivationType type);

int main(int argc, char* argv[]) {
    // Parse command line arguments
    ActivationType hidden_activation = ActivationType::TANH; // Default to TANH
    ActivationType output_activation = ActivationType::SIGMOID; // Default to Sigmoid
    int n_hidden = 15;
    float l_rate = 0.01f; // Kleinere Lernrate für stabileres Training
    int n_epoch = 5000;   // Mehr Epochen
    std::string dataset_path = "data/seeds_dataset.csv";
    
    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--hidden" && i + 1 < argc) {
            hidden_activation = get_activation_type_from_name(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_activation = get_activation_type_from_name(argv[++i]);
        } else if (arg == "--neurons" && i + 1 < argc) {
            n_hidden = std::stoi(argv[++i]);
        } else if (arg == "--rate" && i + 1 < argc) {
            l_rate = std::stof(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            n_epoch = std::stoi(argv[++i]);
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_path = argv[++i];
        }
    }

    std::cout << "=== Neural Network from Scratch in C++ ===" << std::endl;
    std::cout << "Hidden layer activation: " << get_activation_name(hidden_activation) << std::endl;
    std::cout << "Output layer activation: " << get_activation_name(output_activation) << std::endl;
    std::cout << "Hidden neurons: " << n_hidden << std::endl;
    std::cout << "Learning rate: " << l_rate << std::endl;
    std::cout << "Epochs: " << n_epoch << std::endl;
    std::cout << "Dataset: " << dataset_path << std::endl << std::endl;

    // Load and preprocess dataset
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> csv_data;
    
    try {
        csv_data = load_csv_data(dataset_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Loaded dataset with " << csv_data.size() << " samples and " 
              << csv_data[0].size() << " features (including output)" << std::endl;

    // Normalize the class labels
    std::map<int, int> lookup = {};
    int index = 0;
    for (auto& vec : csv_data) {
        std::pair<std::map<int, int>::iterator, bool> ret;
        ret = lookup.insert(std::pair<int, int>(static_cast<int>(vec.back()), index));
        vec.back() = static_cast<float>(ret.first->second);
        if (ret.second) {
            index++;
        }
    }

    // Determine the number of output classes
    std::set<float> results;
    for (const auto& r : csv_data) {
        results.insert(r.back());
    }
    int n_outputs = results.size();
    std::cout << "Detected " << n_outputs << " output classes" << std::endl;

    // Initialize network
    Network* network = new Network();
    std::string network_filename = "saved_network.txt";

    // Check if a saved network exists
    std::ifstream infile(network_filename);
    if (infile.good()) {
        infile.close();
        std::cout << "Network found. Loading..." << std::endl;
        if (network->load(network_filename)) {
            std::cout << "Network loaded successfully." << std::endl;
        } else {
            std::cerr << "Failed to load network. Initializing new network." << std::endl;
            network->initialize_network(csv_data[0].size() - 1, n_hidden, n_outputs, 
                                       hidden_activation, output_activation);
        }
    } else {
        std::cout << "Network not found. Initializing new network." << std::endl;
        network->initialize_network(csv_data[0].size() - 1, n_hidden, n_outputs, 
                                   hidden_activation, output_activation);
    }

    // Display the initial network structure
    std::cout << "\nInitial Network Structure:" << std::endl;
    network->display_human();

    // Measure performance before training
    std::cout << "\nEvaluating network before training..." << std::endl;
    float accuracy_before = evaluate_network(network, csv_data);
    std::cout << "Initial accuracy: " << std::fixed << std::setprecision(2) << accuracy_before << "%" << std::endl;

    // Train network
    std::cout << "\nTraining network..." << std::endl;
    auto training_start = std::chrono::high_resolution_clock::now();
    network->train(csv_data, l_rate, n_epoch, n_outputs, true);
    auto training_end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> training_duration = training_end - training_start;
    std::cout << "Training completed in " << training_duration.count() << " seconds" << std::endl;

    // Measure performance after training
    std::cout << "\nEvaluating network after training..." << std::endl;
    float accuracy_after = evaluate_network(network, csv_data);
    std::cout << "Final accuracy: " << std::fixed << std::setprecision(2) << accuracy_after << "%" << std::endl;

    // View improvement report
    float improvement = accuracy_after - accuracy_before;
    std::cout << "\nPerformance Improvement: " << std::showpos << improvement << "%" << std::noshowpos << std::endl;

    // Save network if improved
    if (accuracy_after > accuracy_before) {
        std::cout << "Network improved. Saving to " << network_filename << "..." << std::endl;
        if (network->save(network_filename)) {
            std::cout << "Network saved successfully." << std::endl;
        } else {
            std::cerr << "Error: Network could not be saved." << std::endl;
        }
    } else if (accuracy_after < accuracy_before) {
        std::cout << "Warning: Network performance worsened. Not saving." << std::endl;
    } else {
        std::cout << "No change in network performance. Not saving." << std::endl;
    }

    // Calculate elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nTotal execution time: " << elapsed.count() << " seconds" << std::endl;

    delete network;
    return 0;
}

// Print usage information
void print_usage() {
    std::cout << "Usage: NN [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help, -h             Show this help message" << std::endl;
    std::cout << "  --hidden TYPE          Hidden layer activation type (SIGMOID, RELU, LEAKY_RELU, TANH, ELU)" << std::endl;
    std::cout << "  --output TYPE          Output layer activation type (SIGMOID, RELU, LEAKY_RELU, TANH, ELU)" << std::endl;
    std::cout << "  --neurons N            Number of neurons in hidden layer (default: 5)" << std::endl;
    std::cout << "  --rate RATE            Learning rate (default: 0.2)" << std::endl;
    std::cout << "  --epochs N             Number of training epochs (default: 500)" << std::endl;
    std::cout << "  --dataset PATH         Path to dataset CSV file (default: ../data/seeds_dataset.csv)" << std::endl;
}

// Convert string to activation type
ActivationType get_activation_type_from_name(const std::string& name) {
    if (name == "SIGMOID") return ActivationType::SIGMOID;
    if (name == "RELU") return ActivationType::RELU;
    if (name == "LEAKY_RELU") return ActivationType::LEAKY_RELU;
    if (name == "TANH") return ActivationType::TANH;
    if (name == "ELU") return ActivationType::ELU;
    
    std::cerr << "Warning: Unknown activation type '" << name << "'. Using SIGMOID." << std::endl;
    return ActivationType::SIGMOID;
}

// Extern-Deklaration der in NeuralNetwork.cpp definierten Funktion
extern std::string get_activation_name(ActivationType type);

// Definition of CSV data loading and normalization function with improved error handling
std::vector<std::vector<float>> load_csv_data(const std::string& filename) {
    std::ifstream csv_file(filename);
    if (!csv_file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    const std::regex comma(",");
    std::vector<std::vector<float>> data;
    std::string line;
    std::vector<float> mins;
    std::vector<float> maxs;
    bool first = true;

    while (csv_file && std::getline(csv_file, line)) {
        try {
            // Skip empty lines
            if (line.empty()) continue;
            
            // Split line by comma
            std::vector<std::string> srow{ std::sregex_token_iterator(line.begin(), line.end(), comma, -1), 
                                           std::sregex_token_iterator() };
            
            // Convert string vector to float vector
            std::vector<float> row(srow.size());
            std::transform(srow.begin(), srow.end(), row.begin(), 
                [](const std::string& val) { return std::stof(val); });

            // Capture min and max values for normalization
            if (first) {
                mins = row;
                maxs = row;
                first = false;
            } else {
                for (size_t t = 0; t < row.size(); t++) {
                    if (row[t] > maxs[t]) {
                        maxs[t] = row[t];
                    } else if (row[t] < mins[t]) {
                        mins[t] = row[t];
                    }
                }
            }

            data.push_back(row);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse line: " << line << std::endl;
            std::cerr << "Error: " << e.what() << std::endl;
            // Continue processing other lines
        }
    }

    if (data.empty()) {
        throw std::runtime_error("No valid data found in file: " + filename);
    }

    // Normalize values with robust error checking
    for (auto& vec : data) {
        // Ignore last column (output)
        for (size_t i = 0; i < vec.size() - 1; i++) {
            if (std::abs(maxs[i] - mins[i]) < 1e-10) {
                // Avoid division by zero for constant features
                vec[i] = 0.5f;
            } else {
                vec[i] = (vec[i] - mins[i]) / (maxs[i] - mins[i]);
            }
        }
    }

    csv_file.close();
    return data;
}

// Definition of the function for evaluating the network
float evaluate_network(Network* network, const std::vector<std::vector<float>>& dataset) {
    if (!network || dataset.empty()) {
        return 0.0f;
    }
    
    int correct = 0;
    int total = dataset.size();

    for (const auto& row : dataset) {
        std::vector<float> input = row;
        int expected = static_cast<int>(input.back());
        input.pop_back(); // Removes the expected output from the inputs

        int predicted = network->predict(input);

        if (predicted == expected) {
            correct++;
        }
    }

    float accuracy = static_cast<float>(correct) / total * 100.0f;
    return accuracy;
}