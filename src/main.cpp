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
#include <algorithm>
#include <memory>
#include "NeuralNetwork.h"

// Function declarations
std::vector<std::vector<float>> loadCsvData(const std::string& filename);
float evaluateNetwork(const Network& network, const std::vector<std::vector<float>>& dataset);

int main(int argc, char* argv[]) {
    std::cout << "Neural Network from scratch" << std::endl;

    const std::string datasetFilename = "../data/seeds_dataset.csv";
    const std::string networkFilename = "saved_network.txt";
    const int nHidden = 5;
    const float learningRate = 0.2f;
    const int nEpochs = 500;

    auto csvData = loadCsvData(datasetFilename);

    // Normalize the last column values
    std::map<int, int> lookup;
    int index = 0;
    for (auto& row : csvData) {
        int lastValue = static_cast<int>(row.back());
        auto [it, inserted] = lookup.try_emplace(lastValue, index);
        row.back() = static_cast<float>(it->second);
        if (inserted) {
            ++index;
        }
    }

    int nOutputs = static_cast<int>(std::set<float>(csvData.back().begin(), csvData.back().end()).size());

    // Initialize network
    std::unique_ptr<Network> network = std::make_unique<Network>();

    if (std::ifstream(networkFilename).good()) {
        std::cout << "Network found. Loading..." << std::endl;
        if (network->load(networkFilename)) {
            std::cout << "Success." << std::endl;
        } else {
            std::cerr << "Failed. Initializing new Network." << std::endl;
            network->initialize_network(csvData[0].size() - 1, nHidden, nOutputs);
        }
    } else {
        std::cout << "Network not found. Initializing new Network." << std::endl;
        network->initialize_network(csvData[0].size() - 1, nHidden, nOutputs);
    }

    // Measure performance before training
    float accuracyBefore = evaluateNetwork(*network, csvData);

    // Train network
    network->train(csvData, learningRate, nEpochs, nOutputs);

    // Measure performance after training
    float accuracyAfter = evaluateNetwork(*network, csvData);

    // View improvement report
    std::cout << "Previous accuracy: " << accuracyBefore << "%" << std::endl;
    std::cout << "Current accuracy: " << accuracyAfter << "%" << std::endl;

    if (accuracyAfter > accuracyBefore) {
        std::cout << "Network improved." << std::endl;
    } else if (accuracyAfter < accuracyBefore) {
        std::cout << "Network worsened." << std::endl;
    } else {
        std::cout << "No change in accuracy." << std::endl;
    }

    // Ask to save network
    std::cout << "Save Network? (y/n): ";
    char saveResponse;
    std::cin >> saveResponse;

    if (saveResponse == 'y' || saveResponse == 'Y') {
        if (network->save(networkFilename)) {
            std::cout << "Saved successfully." << std::endl;
        } else {
            std::cerr << "Error. Network could not be saved." << std::endl;
        }
    } else {
        std::cout << "Changes discarded." << std::endl;
    }

    return 0;
}

// Definition of CSV data loading and normalization function
std::vector<std::vector<float>> loadCsvData(const std::string& filename) {
    const std::regex comma(",");
    std::ifstream csvFile(filename);
    std::vector<std::vector<float>> data;
    std::string line;
    std::vector<float> mins, maxs;

    while (std::getline(csvFile, line)) {
        std::vector<std::string> srow{ 
            std::sregex_token_iterator(line.begin(), line.end(), comma, -1), 
            std::sregex_token_iterator() 
        };
        std::vector<float> row(srow.size());
        std::transform(srow.begin(), srow.end(), row.begin(), 
                       [](const std::string& val) { return std::stof(val); });

        if (data.empty()) {
            mins = maxs = row;
        } else {
            for (size_t i = 0; i < row.size(); ++i) {
                mins[i] = std::min(mins[i], row[i]);
                maxs[i] = std::max(maxs[i], row[i]);
            }
        }

        data.push_back(row);
    }

    // Normalize values
    for (auto& vec : data) {
        // Ignore last column (output).
        for (size_t i = 0; i < vec.size() - 1; ++i) {
            vec[i] = (vec[i] - mins[i]) / (maxs[i] - mins[i]);
        }
    }

    return data;
}

// Definition of the function for evaluating the network
float evaluateNetwork(const Network& network, const std::vector<std::vector<float>>& dataset) {
    int correct = 0;
    int total = dataset.size();

    for (const auto& row : dataset) {
        std::vector<float> input(row.begin(), row.end() - 1);
        int expected = static_cast<int>(row.back());
        int predicted = network.predict(input);

        if (predicted == expected) {
            ++correct;
        }
    }

    return static_cast<float>(correct) / total * 100.0f;
}
