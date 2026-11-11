// XOR with different neuron types: shows that every activation can be plugged in.
#include <cstdio>

#include <nn/nn.hpp>

int main() {
    nn::Matrix x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    nn::Matrix y = {{0}, {1}, {1}, {0}};

    using K = nn::ActivationKind;
    for (K k : {K::Sigmoid, K::Tanh, K::ReLU, K::LeakyReLU, K::ELU, K::SELU, K::GELU, K::Swish,
                K::Mish, K::Softplus, K::Softsign, K::HardSwish, K::Gaussian, K::Sine}) {
        nn::Network net(2, 1);
        net.add<nn::Dense>(8, nn::Activation(k));
        net.add<nn::Dense>(1, nn::Activation::sigmoid());
        net.compile(nn::Loss::BinaryCrossEntropy, std::make_unique<nn::Adam>(0.05f));
        nn::FitOptions o;
        o.epochs = 400;
        o.batch_size = 4;
        o.verbose = 0;
        auto h = net.fit(x, y, o);
        nn::Matrix p = net.predict(x);
        std::printf("%-12s loss %.4f  ->  %.2f %.2f %.2f %.2f\n", nn::to_string(k), h.loss.back(), p[0],
                    p[1], p[2], p[3]);
    }

    // RBF network: radial neurons solve XOR with a single hidden layer, too.
    nn::Network rbf(2, 3);
    rbf.add<nn::RBF>(4, 2.0f);
    rbf.add<nn::Dense>(1, nn::Activation::sigmoid());
    rbf.compile(nn::Loss::BinaryCrossEntropy, std::make_unique<nn::Adam>(0.05f));
    nn::FitOptions o;
    o.epochs = 400;
    o.batch_size = 4;
    o.verbose = 0;
    rbf.fit(x, y, o);
    nn::Matrix p = rbf.predict(x);
    std::printf("%-12s          ->  %.2f %.2f %.2f %.2f\n", "RBF", p[0], p[1], p[2], p[3]);
    return 0;
}
