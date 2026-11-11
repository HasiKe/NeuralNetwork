#pragma once

#include "../activation.hpp"
#include "../layer.hpp"

namespace nn {

/// Common base for recurrent layers.
///
/// Input rows are sequences flattened time-major: row = [x_0 | x_1 | ... | x_{T-1}],
/// each x_t having `input_dim / timesteps` features. Output is either the last
/// hidden state [units] or, with return_sequences, all states [T * units] in
/// the same layout, so recurrent layers can be stacked.
///
/// All input projections for every timestep are computed with one large GEMM;
/// only the recurrent h_{t-1} W_h product is sequential.
class RecurrentBase : public Layer {
public:
    std::size_t units() const { return m_units; }
    std::size_t timesteps() const { return m_T; }
    std::size_t features() const { return m_F; }
    bool return_sequences() const { return m_return_seq; }

protected:
    RecurrentBase() = default;
    RecurrentBase(std::size_t units, std::size_t timesteps, bool return_sequences)
        : m_units(units), m_T(timesteps), m_return_seq(return_sequences) {}

    void build_common(std::size_t input_dim, std::size_t gates, Rng& rng);
    void save_common(std::ostream& os) const;
    void load_common(std::istream& is, std::size_t gates);
    void params_common(std::vector<ParamRef>& p);
    /// Copy dL/dh_t for timestep t of the output gradient into dh (adds).
    void add_output_grad(const Matrix& grad_out, std::size_t t, std::size_t batch, real* dh) const;

    std::size_t m_units = 0, m_T = 1, m_F = 0;
    bool m_return_seq = false;
    // Wx: [F x G*H], Wh: [H x G*H], b: [G*H]
    Matrix m_wx, m_wh, m_b, m_dwx, m_dwh, m_db;
    Matrix m_xproj;  // [(B*T) x G*H] input projections / gate cache
    Matrix m_hs;     // [B x (T+1) x H] hidden states incl. h_{-1} = 0
    Matrix m_dgates; // [(B*T) x G*H]
    Matrix m_dh, m_dh_next;
};

/// Elman recurrent neurons: h_t = f(x_t Wx + h_{t-1} Wh + b).
class SimpleRNN final : public RecurrentBase {
public:
    SimpleRNN() = default;
    SimpleRNN(std::size_t units, std::size_t timesteps, bool return_sequences = false,
              Activation act = ActivationKind::Tanh)
        : RecurrentBase(units, timesteps, return_sequences), m_act(act) {}

    LayerType type() const override { return LayerType::SimpleRNN; }
    std::string name() const override;
    void build(std::size_t input_dim, Rng& rng) override;
    void forward(const Matrix& in, Matrix& out, bool training) override;
    void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                  Matrix* grad_in) override;
    void parameters(std::vector<ParamRef>& p) override { params_common(p); }
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

private:
    Activation m_act;
    Matrix m_z;  // pre-activations [(B*T) x H]
};

/// Long short-term memory cells (gates i, f, g, o; forget bias initialised to 1).
class LSTM final : public RecurrentBase {
public:
    LSTM() = default;
    LSTM(std::size_t units, std::size_t timesteps, bool return_sequences = false)
        : RecurrentBase(units, timesteps, return_sequences) {}

    LayerType type() const override { return LayerType::LSTM; }
    std::string name() const override;
    void build(std::size_t input_dim, Rng& rng) override;
    void forward(const Matrix& in, Matrix& out, bool training) override;
    void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                  Matrix* grad_in) override;
    void parameters(std::vector<ParamRef>& p) override { params_common(p); }
    void save(std::ostream& os) const override { save_common(os); }
    void load(std::istream& is) override { load_common(is, 4); }

private:
    Matrix m_cs;     // [B x (T+1) x H] cell states incl. c_{-1} = 0
    Matrix m_tanhc;  // [(B*T) x H]
    Matrix m_dc;
};

/// Gated recurrent units (PyTorch formulation, reset gate applied after W_h).
class GRU final : public RecurrentBase {
public:
    GRU() = default;
    GRU(std::size_t units, std::size_t timesteps, bool return_sequences = false)
        : RecurrentBase(units, timesteps, return_sequences) {}

    LayerType type() const override { return LayerType::GRU; }
    std::string name() const override;
    void build(std::size_t input_dim, Rng& rng) override;
    void forward(const Matrix& in, Matrix& out, bool training) override;
    void backward(const Matrix& in, const Matrix& out, const Matrix& grad_out,
                  Matrix* grad_in) override;
    void parameters(std::vector<ParamRef>& p) override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

private:
    Matrix m_bh, m_dbh;  // separate recurrent bias [3H]
    Matrix m_hproj;      // [(B*T) x 3H] h_{t-1} Wh + bh (cached)
    Matrix m_dhproj;     // [(B*T) x 3H]
};

}  // namespace nn
