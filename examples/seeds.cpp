// Seeds dataset classification (successor of the original NN demo).
//
//   example_seeds [path/to/seeds_dataset.csv] [model.nnet]
//
// Loads an existing model if present, trains, reports accuracy before/after and
// saves the model when it improved.
#include <cstdio>
#include <fstream>
#include <iostream>

#include <nn/nn.hpp>

int main(int argc, char** argv) {
    const std::string csv = argc > 1 ? argv[1] : "data/seeds_dataset.csv";
    const std::string model_path = argc > 2 ? argv[2] : "seeds.nnet";
    std::cout << nn::build_info() << "\n";

    nn::Matrix raw = nn::data::load_csv(csv);
    auto [x, labels] = nn::data::split_xy(raw);  // last column = class
    const std::size_t classes = nn::data::encode_labels(labels);
    nn::Rng rng(2024);
    auto split = nn::data::train_test_split(x, nn::data::one_hot(labels, classes), 0.2f, rng);

    nn::data::StandardScaler scaler;
    scaler.fit_transform(split.x_train);
    scaler.transform(split.x_test);

    nn::Network net(x.cols());
    bool loaded = false;
    if (std::ifstream(model_path).good()) {
        try {
            net = nn::Network::load(model_path);
            loaded = net.input_dim() == x.cols() && net.output_dim() == classes;
        } catch (const std::exception& e) {
            std::cerr << "could not load " << model_path << ": " << e.what() << "\n";
        }
    }
    if (!loaded) {
        net = nn::Network(x.cols(), 7);
        net.add<nn::Dense>(32, nn::Activation::gelu());
        net.add<nn::Dense>(16, nn::Activation::gelu());
        net.add<nn::Dense>(classes, nn::Activation::softmax());
        std::cout << "new network\n";
    } else {
        std::cout << "loaded " << model_path << "\n";
    }
    net.compile(nn::Loss::CategoricalCrossEntropy, std::make_unique<nn::Adam>(0.01f));
    std::cout << net.summary();

    const float before = net.evaluate_accuracy(split.x_test, split.y_test);

    nn::FitOptions opt;
    opt.epochs = 150;
    opt.batch_size = 16;
    opt.verbose = 0;
    opt.schedule = nn::LRSchedule::reduce_on_plateau(10, 0.5f);  // automatic learning rate
    opt.on_epoch_end = [](std::size_t e, float loss, float vloss) {
        (void)vloss;
        if ((e + 1) % 25 == 0) std::printf("epoch %3zu  loss %.4f\n", e + 1, loss);
        return true;
    };
    net.fit(split.x_train, split.y_train, opt);

    const float after = net.evaluate_accuracy(split.x_test, split.y_test);
    std::printf("test accuracy: before %.1f%%  after %.1f%%\n", before * 100, after * 100);
    if (after >= before) {
        net.save(model_path);
        std::cout << "saved to " << model_path << "\n";
    }
    return 0;
}
