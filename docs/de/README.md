# nn — CPU-effiziente Neuronale-Netze-Bibliothek in C++17

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/HasiKe/NeuralNetwork/blob/main/README.md)
[![de](https://img.shields.io/badge/lang-de-red.svg)](https://github.com/HasiKe/NeuralNetwork/blob/main/docs/de/README.md)

Eine abhängigkeitsfreie Bibliothek für neuronale Netze, von Grund auf
geschrieben. Ursprünglich ein Lernprojekt (ein Perzeptron mit einer
versteckten Schicht für den Seeds-Datensatz), jetzt eine wiederverwendbare
Bibliothek mit vielen Neuronentypen, Mini-Batch-Training und
cache-optimierten, SIMD-vektorisierten, mehrkernfähigen CPU-Kernels.

## Funktionen

**Neuronentypen**

| Familie | Typen |
|---|---|
| Aktivierungsfunktionen (`nn::Activation`) | Linear, Sigmoid, Tanh, ReLU, LeakyReLU, ELU, SELU, GELU, Swish/SiLU, Mish, Softplus, Softsign, HardSigmoid, HardTanh, HardSwish, ReLU6, Gauß, Sinus, Step (Perzeptron), Softmax |
| Feed-Forward-Schichten | `Dense` (voll verbunden), `RBF` (radiale Basisfunktionen), `Dropout`, `LayerNorm` |
| Rekurrente Zellen | `SimpleRNN` (Elman), `LSTM`, `GRU` — stapelbar über `return_sequences` |
| Faltung | `Conv2D`, `MaxPool2D` |
| Spikende Neuronen (`nn::spiking`) | Leaky Integrate-and-Fire, Izhikevich (7 Presets), Adaptive Exponential IF (AdEx), Hodgkin-Huxley; exponentielle Synapsen, Poisson-Input, STDP-Plastizität |

**Training**: MSE, MAE, Huber, kategorische / binäre Kreuzentropie (mit
Softmax- / Sigmoid-Ausgang fusioniert), SGD (+Momentum, Nesterov),
Adam/AdamW, RMSProp, Adagrad, Gradient Clipping, Weight Decay,
Lernraten-Schedules (Step, exponentiell, Cosine, *Reduce-on-Plateau* für
automatische Lernrate), Validierungssplit, Early Stopping mit
Wiederherstellung der besten Gewichte.

**Hilfsfunktionen**: schneller CSV-Loader, Label-Encoding, One-Hot,
Train/Test-Split, Min-Max- und Standard-Skalierung, binäres Speichern/Laden.

## Performance-Design

* **Zusammenhängende Matrizen statt Neuron-Objekten.** Jede Schicht speichert
  ihre Gewichte als eine ausgerichtete `[in x out]`-Matrix; ein ganzer
  Mini-Batch läuft mit einer GEMM durch die Schicht statt einem Skalarprodukt
  pro Neuron.
* **Cache- und registerblockierte GEMM** (4x16-Mikrokernel, L2-große Panels),
  über Ausgabekacheln mit OpenMP parallelisiert. Optional CBLAS
  (`-DNN_USE_BLAS=ON`, z. B. OpenBLAS/MKL).
* **Vektorisierte Transzendente** (`nn/fastmath.hpp`): verzweigungsfreie
  exp/log/tanh/sigmoid, die der Compiler ohne `-ffast-math` zu AVX/NEON-Code
  macht; Eingaben werden so begrenzt, dass keine subnormalen Zahlen entstehen.
* **Keine Allokationen in der Trainingsschleife**: Matrizen wachsen nur, alle
  Arbeitspuffer werden zwischen Batches wiederverwendet.
* **Rekurrente Schichten** berechnen die Eingangsprojektion aller Zeitschritte
  in einer großen GEMM; nur `h_{t-1} W_h` ist sequentiell.
* **Spiking-Simulation** mit Structure-of-Arrays-Zustand (vektorisierte
  Updates) und CSR-Synapsen: Kosten skalieren mit der Zahl der Spikes, nicht
  der Synapsen. Zufallsverbindungen per geometrischem Überspringen
  (O(Synapsen) statt O(n²)).

Gemessen auf einem Intel N97 (4 Kerne, AVX2), `./build/nn_bench`:

| Benchmark | Ergebnis |
|---|---|
| GEMM 256³ / 1024³ | ~80 / ~50-65 GFLOP/s |
| MLP 784-512-256-10 Training (Batch 128) | ~17.000 Samples/s |
| MLP Inferenz | ~60.000 Samples/s |
| LSTM (T=32, H=128) Training | ~2.100 Sequenzen/s |
| 10k Izhikevich-Neuronen, 1 Mio. Synapsen | ~7x schneller als Echtzeit |

## Bauen

Voraussetzungen: C++17-Compiler (GCC ≥ 8, Clang ≥ 7, MSVC 2019), CMake ≥ 3.14.
OpenMP wird genutzt, falls vorhanden.

**Klassisches Programm (Standard).** Ohne Flags baut das Projekt wie
früher: das ursprüngliche Netz mit einer versteckten Schicht
(`src/main.cpp`, `Neuron` / `Layer` / `Network`) als `bin/NN`.

```bash
mkdir build && cd build
cmake ..
make
cd ../bin && ./NN
```

**Bibliothek.** Die neue `nn`-Bibliothek samt Tests, Beispielen und
Benchmarks wird mit `NN_BUILD_LIBRARY` aktiviert:

```bash
cmake -S . -B build -DNN_BUILD_LIBRARY=ON   # standardmäßig Release
cmake --build build -j
ctest --test-dir build                      # oder ./build/nn_tests
./build/nn_bench
```

CMake-Optionen:

| Option | Standard | Bedeutung |
|---|---|---|
| `NN_BUILD_LIBRARY` | OFF (Top-Level), ON (add_subdirectory) | `nn`-Bibliothek und Extras bauen |
| `NN_BUILD_LEGACY` | ON (Top-Level) | klassisches Programm `bin/NN` bauen |
| `NN_USE_OPENMP` | ON | mehrkernfähige Kernels |
| `NN_USE_BLAS` | OFF | GEMM an externes CBLAS delegieren |
| `NN_NATIVE_ARCH` | ON (Top-Level) | `-march=native`; für portable Binaries ausschalten |
| `NN_BUILD_EXAMPLES` / `NN_BUILD_TESTS` / `NN_BUILD_BENCHMARKS` | ON (Top-Level) | Extras |
| `BUILD_SHARED_LIBS` | OFF | `libnn.so` statt `libnn.a` |

### Einbinden in eigene Projekte

Installieren und `find_package` nutzen:

```bash
cmake -S . -B build -DNN_BUILD_LIBRARY=ON && cmake --build build -j
cmake --install build --prefix /opt/nn
```
```cmake
find_package(nn 2 REQUIRED)            # mit CMAKE_PREFIX_PATH=/opt/nn
target_link_libraries(myapp PRIVATE nn::nn)
```

oder als Unterverzeichnis:

```cmake
add_subdirectory(external/NeuralNetwork)   # Beispiele/Tests automatisch aus
target_link_libraries(myapp PRIVATE nn::nn)
```

## Schnellstart

```cpp
#include <nn/nn.hpp>

nn::Matrix raw = nn::data::load_csv("data/seeds_dataset.csv");
auto [x, labels] = nn::data::split_xy(raw);          // letzte Spalte = Label
std::size_t k = nn::data::encode_labels(labels);
nn::Matrix y = nn::data::one_hot(labels, k);
nn::data::StandardScaler().fit_transform(x);

nn::Network net(x.cols());
net.add<nn::Dense>(32, nn::Activation::gelu());
net.add<nn::Dropout>(0.1f);
net.add<nn::Dense>(k, nn::Activation::softmax());
net.compile(nn::Loss::CategoricalCrossEntropy, std::make_unique<nn::Adam>(1e-2f));

nn::FitOptions opt;
opt.epochs = 100;
opt.batch_size = 16;
opt.validation_split = 0.2f;
opt.schedule = nn::LRSchedule::reduce_on_plateau();  // automatische Lernrate
net.fit(x, y, opt);

auto classes = net.predict_classes(x);
net.save("model.nnet");
nn::Network again = nn::Network::load("model.nnet");
```

### Datenlayout

Jede Schicht verarbeitet Matrizen der Form `[Batch x Merkmale]`. Strukturierte
Schichten interpretieren eine Zeile als flachen Tensor, Reshape-Schichten sind
daher unnötig:

* **Sequenzen** (RNN/LSTM/GRU): zeitlich geordnet, `[x_0 | x_1 | ... | x_{T-1}]`,
  `Merkmale = input_dim / timesteps`.
* **Bilder** (Conv2D/MaxPool2D): Kanal-Höhe-Breite (`CHW`).

```cpp
nn::Network seq(T * F);
seq.add<nn::LSTM>(64, T, /*return_sequences=*/true);
seq.add<nn::GRU>(32, T);
seq.add<nn::Dense>(1);

nn::Network cnn(28 * 28);
auto& c1 = cnn.add<nn::Conv2D>(nn::Shape3{1, 28, 28}, 16, 3, 1, 1, nn::Activation::relu());
auto& p1 = cnn.add<nn::MaxPool2D>(c1.output_shape(), 2);
cnn.add<nn::Dense>(10, nn::Activation::softmax());
```

### Spikende Netze

```cpp
namespace sp = nn::spiking;
sp::Network net(/*dt ms*/ 0.5f);
auto exc = net.add_population(8000, sp::IzhikevichParams::regular_spiking());
auto inh = net.add_population(2000, sp::IzhikevichParams::fast_spiking());
net.connect(exc, exc, 0.01f, 0.5f);
net.connect(inh, exc, 0.01f, -1.0f);
net.add_poisson_input(exc, 2000, 1.5f);
net.enable_stdp(exc, exc, sp::STDPParams{});
net.run(2000);                         // 1 s
float hz = net.firing_rate(exc);
```

## Beispiele

| Programm | Zeigt |
|---|---|
| `example_seeds` | Nachfolger der ursprünglichen Demo: Laden/Trainieren/Bewerten/Speichern auf dem Seeds-Datensatz |
| `example_xor` | dasselbe Problem mit 14 verschiedenen Neuronentypen und einem RBF-Netz |
| `example_sequence` | Zeitreihenprognose mit gestapelten SimpleRNN / GRU / LSTM |
| `example_mnist_like` | kleines CNN auf synthetischen 16x16-Formbildern |
| `example_spiking` | Izhikevich-Netz mit 10k Neuronen plus LIF- / AdEx- / HH-Zellen |

## Projektstruktur

```
include/nn/        öffentliche Header (nn.hpp = Sammelheader)
  layers/          Dense, Dropout, RBF, LayerNorm, rekurrent, Faltung
src/               Implementierung (Bibliothek) + klassisches Programm (main.cpp, Neuron/Layer/NeuralNetwork.cpp)
tests/             eigenständige Testsuite (Gradientenprüfung für jede Schicht und Aktivierung)
benchmarks/        nn_bench
examples/          Beispielprogramme
data/              Seeds-Datensatz
docs/              Hintergrund zu Netzarchitekturen
```

## Hinweise

* `Network` verwendet interne Puffer wieder und ist nicht thread-sicher; für
  parallele Inferenz eine Instanz pro Thread nutzen (Kopie über
  `save`/`load`). Die Kernels selbst sind mehrkernfähig.
* `nn::set_num_threads(n)` begrenzt die Threadzahl (ebenso `OMP_NUM_THREADS`).
* Modelldateien sind Little-Endian-Binärdateien (`.nnet`, Formatversion 2).
  Dateien der alten textbasierten Version 1 können nicht geladen werden.

## Lizenz

Siehe [LICENSE.txt](../../LICENSE.txt).
