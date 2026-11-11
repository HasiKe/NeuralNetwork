// Convolutional neurons on synthetic 16x16 images: classify which shape
// (horizontal bar, vertical bar, diagonal, box) appears at a random position.
#include <cstdio>

#include <nn/nn.hpp>

static void draw(nn::Matrix& x, std::size_t row, int cls, nn::Rng& rng) {
    const int S = 16;
    float* img = x.row(row);
    for (int i = 0; i < S * S; ++i) img[i] = 0.1f * rng.uniform();
    const int ox = static_cast<int>(rng.below(8)), oy = static_cast<int>(rng.below(8));
    for (int k = 0; k < 8; ++k) {
        switch (cls) {
            case 0: img[(oy + 4) * S + ox + k] = 1; break;            // horizontal
            case 1: img[(oy + k) * S + ox + 4] = 1; break;            // vertical
            case 2: img[(oy + k) * S + ox + k] = 1; break;            // diagonal
            default:                                                  // box outline
                img[oy * S + ox + k] = img[(oy + 7) * S + ox + k] = 1;
                img[(oy + k) * S + ox] = img[(oy + k) * S + ox + 7] = 1;
        }
    }
}

int main() {
    const std::size_t N = 4000;
    nn::Rng rng(3);
    nn::Matrix x(N, 256), labels(N, 1);
    for (std::size_t i = 0; i < N; ++i) {
        const int c = static_cast<int>(rng.below(4));
        draw(x, i, c, rng);
        labels(i, 0) = static_cast<float>(c);
    }
    auto split = nn::data::train_test_split(x, nn::data::one_hot(labels, 4), 0.2f, rng);

    nn::Network net(256, 1);
    auto& c1 = net.add<nn::Conv2D>(nn::Shape3{1, 16, 16}, 8, 3, 1, 1, nn::Activation::relu());
    auto& p1 = net.add<nn::MaxPool2D>(c1.output_shape(), 2);
    auto& c2 = net.add<nn::Conv2D>(p1.output_shape(), 16, 3, 1, 1, nn::Activation::relu());
    net.add<nn::MaxPool2D>(c2.output_shape(), 2);
    net.add<nn::Dense>(32, nn::Activation::relu());
    net.add<nn::Dense>(4, nn::Activation::softmax());
    net.compile(nn::Loss::CategoricalCrossEntropy, std::make_unique<nn::Adam>(2e-3f));
    std::printf("%s", net.summary().c_str());

    nn::FitOptions o;
    o.epochs = 8;
    o.batch_size = 64;
    net.fit(split.x_train, split.y_train, o);
    std::printf("test accuracy %.1f%%\n", 100 * net.evaluate_accuracy(split.x_test, split.y_test));
    return 0;
}
