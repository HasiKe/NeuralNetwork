# 🧠 Neural Network from Scratch in C++

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/HasiKe/NeuralNetwork/blob/main/README.md)
[![de](https://img.shields.io/badge/lang-de-red.svg)](https://github.com/HasiKe/NeuralNetwork/blob/main/docs/de/README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-11-blue.svg)](https://isocpp.org/)

> A simple neural network implementation in C++ without external machine learning libraries.

## 📚 About the Project

This project aims to demystify neural networks by implementing one from scratch in C++. It serves as an educational resource, providing insights into:

- Forward and backpropagation
- Weight adjustments
- Model saving and loading

## ✨ Features

- 🏗️ **Standalone Neural Network**: No external libraries required
- 💾 **Save and Load Functions**: Preserve trained models for future use
- 📊 **Performance Evaluation**: Compare accuracy before and after training
- 🤝 **User Interaction**: Informative prompts and save options

## 🚀 Getting Started

### Prerequisites

- C++ Compiler (C++11 or later)
- CMake
- Standard C++ libraries

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HasiKe/NeuralNetwork.git
   cd NeuralNetwork
   ```

2. Create and navigate to the build directory:
   ```bash
   mkdir build && cd build
   ```

3. Configure with CMake:
   ```bash
   cmake ..
   ```

4. Build the project:
   ```bash
   make
   ```

## 🖥️ Usage

Run the program:

```bash
./NN
```

The program will:

1. 📥 Load the Seeds dataset
2. 🔍 Check for existing network or initialize a new one
3. 📊 Evaluate initial performance
4. 🏋️ Train the network
5. 📈 Re-evaluate performance
6. 💾 Offer to save the trained network

## 🗂️ Project Structure

```
NeuralNetwork/
│
├── src/            # Source code
├── include/        # Header files
├── data/           # Dataset files
└── build/          # Build directory (created during build process)
```

## ⚙️ Adjusting Hyperparameters

Modify these values in `main.cpp`:

```cpp
int nHidden = 5;   // Number of neurons in hidden layer
float learningRate = 0.2f; // Learning rate
int nEpochs = 500;  // Number of training epochs
```

## 📊 Preparing Data

Ensure `seeds_dataset.csv` is in the `data/` directory. Adjust the path in `main.cpp` if needed:

```cpp
csv_data = load_csv_data("data/seeds_dataset.csv");
```

## 📝 TODO

- [ ] Implement automatic learning rate adjustment
- [ ] Add support for different types of neurons

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ by HasiKe</p>
