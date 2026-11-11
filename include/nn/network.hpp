#pragma once

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

namespace nn {

struct FitOptions {
    std::size_t epochs = 10;
    std::size_t batch_size = 32;
    bool shuffle = true;
    /// 0 = silent, 1 = one line per epoch
    int verbose = 1;
    /// Fraction of the training data held out for validation (0 = none).
    real validation_split = 0;
    /// Stop when validation (or training) loss has not improved for this many epochs (0 = off).
    std::size_t early_stopping_patience = 0;
    LRSchedule schedule;
    /// Called after every epoch; return false to stop training.
    std::function<bool(std::size_t epoch, real loss, real val_loss)> on_epoch_end;
};

struct History {
    std::vector<real> loss, val_loss, accuracy, val_accuracy, learning_rate;
};

/// Sequential feed-forward container.
///
///   nn::Network net(784);
///   net.add<nn::Dense>(128, nn::Activation::relu());
///   net.add<nn::Dense>(10, nn::Activation::softmax());
///   net.compile(nn::Loss::CategoricalCrossEntropy, std::make_unique<nn::Adam>(1e-3f));
///   net.fit(X, Y, opts);
///
/// Not thread-safe: forward passes reuse internal buffers. Use one Network
/// per thread (copy via save/load) for concurrent inference.
class Network {
public:
    explicit Network(std::size_t input_dim = 0, std::uint64_t seed = 42);
    Network(Network&&) noexcept;
    Network& operator=(Network&&) noexcept;
    ~Network();

    /// Construct and append a layer: net.add<Dense>(64, Activation::relu()).
    template <class L, class... Args>
    L& add(Args&&... args) {
        static_assert(std::is_base_of<Layer, L>::value, "L must derive from nn::Layer");
        auto p = std::make_unique<L>(std::forward<Args>(args)...);
        L& ref = *p;
        add(std::move(p));
        return ref;
    }
    Layer& add(LayerPtr layer);

    void compile(Loss loss, std::unique_ptr<Optimizer> opt);

    /// Forward pass on a batch (any number of rows). Returns network output.
    const Matrix& forward(const Matrix& x, bool training = false);
    /// Back-propagate from the loss for the last forward(x, true). Returns batch loss.
    real backward(const Matrix& target);
    /// One optimisation step on a batch. Returns the loss before the update.
    real train_batch(const Matrix& x, const Matrix& y);

    History fit(const Matrix& x, const Matrix& y, const FitOptions& opt = {});
    History fit(const Matrix& x, const Matrix& y, const Matrix& x_val, const Matrix& y_val,
                const FitOptions& opt = {});

    /// Inference in chunks of `batch_size` rows (keeps memory bounded).
    Matrix predict(const Matrix& x, std::size_t batch_size = 256);
    std::vector<std::size_t> predict_classes(const Matrix& x, std::size_t batch_size = 256);

    real evaluate_loss(const Matrix& x, const Matrix& y, std::size_t batch_size = 256);
    /// Classification accuracy in [0,1]; targets either one-hot or a single class-index column.
    real evaluate_accuracy(const Matrix& x, const Matrix& y, std::size_t batch_size = 256);

    bool save(const std::string& path) const;
    void save(std::ostream& os) const;
    static Network load(const std::string& path);
    static Network load(std::istream& is);

    std::size_t input_dim() const { return m_input_dim; }
    std::size_t output_dim() const;
    std::size_t size() const { return m_layers.size(); }
    Layer& layer(std::size_t i) { return *m_layers.at(i); }
    std::size_t parameter_count();
    std::string summary();
    Optimizer* optimizer() { return m_opt.get(); }
    Loss loss() const { return m_loss; }
    Rng& rng() { return m_rng; }

private:
    void refresh_params();
    std::size_t m_input_dim;
    Rng m_rng;
    std::vector<LayerPtr> m_layers;
    std::vector<Matrix> m_acts;   // m_acts[i] = output of layer i
    std::vector<Matrix> m_grads;  // gradient wrt m_acts[i]
    const Matrix* m_last_input = nullptr;
    Matrix m_loss_grad;
    Loss m_loss = Loss::MeanSquaredError;
    std::unique_ptr<Optimizer> m_opt;
    std::vector<ParamRef> m_params;
    bool m_params_dirty = true;
    Matrix m_bx, m_by;  // batch buffers
};

}  // namespace nn
