#pragma once

#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

#include "config.hpp"

namespace nn {

/// Dense row-major float matrix with 64-byte aligned storage.
///
/// Rows are samples, columns are features. The buffer only grows: resizing to
/// a smaller or equal number of elements never reallocates, so matrices used
/// as per-batch workspaces cost nothing after the first iteration.
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols) { resize(rows, cols); zero(); }
    Matrix(std::size_t rows, std::size_t cols, real value) { resize(rows, cols); fill(value); }
    Matrix(std::initializer_list<std::initializer_list<real>> init);

    Matrix(const Matrix& o) { *this = o; }
    Matrix(Matrix&& o) noexcept { swap(o); }
    Matrix& operator=(const Matrix& o);
    Matrix& operator=(Matrix&& o) noexcept { swap(o); return *this; }

    /// Change shape. Contents are unspecified afterwards (no zeroing).
    void resize(std::size_t rows, std::size_t cols);
    void zero() { if (size()) std::memset(m_data.get(), 0, size() * sizeof(real)); }
    void fill(real v);

    std::size_t rows() const { return m_rows; }
    std::size_t cols() const { return m_cols; }
    std::size_t size() const { return m_rows * m_cols; }
    bool empty() const { return size() == 0; }

    real* data() { return m_data.get(); }
    const real* data() const { return m_data.get(); }
    real* row(std::size_t r) { return m_data.get() + r * m_cols; }
    const real* row(std::size_t r) const { return m_data.get() + r * m_cols; }

    real& operator()(std::size_t r, std::size_t c) { return m_data[r * m_cols + c]; }
    real operator()(std::size_t r, std::size_t c) const { return m_data[r * m_cols + c]; }
    real& operator[](std::size_t i) { return m_data[i]; }
    real operator[](std::size_t i) const { return m_data[i]; }

    void swap(Matrix& o) noexcept {
        std::swap(m_data, o.m_data);
        std::swap(m_rows, o.m_rows);
        std::swap(m_cols, o.m_cols);
        std::swap(m_capacity, o.m_capacity);
    }

    /// Copy selected rows of `src` into this matrix (used for mini-batching).
    void gather_rows(const Matrix& src, const std::size_t* indices, std::size_t count);

    std::vector<real> to_vector() const { return std::vector<real>(data(), data() + size()); }
    static Matrix from_vector(const std::vector<real>& v, std::size_t rows, std::size_t cols);

private:
    struct AlignedDelete { void operator()(real* p) const noexcept; };
    std::unique_ptr<real[], AlignedDelete> m_data;
    std::size_t m_rows = 0, m_cols = 0, m_capacity = 0;
};

/// Index of the maximum element in each row.
std::vector<std::size_t> argmax_rows(const Matrix& m);

}  // namespace nn
