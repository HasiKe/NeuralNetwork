#include "nn/matrix.hpp"

#include <algorithm>
#include <cstdlib>
#include <new>

namespace nn {

namespace {
constexpr std::size_t kAlign = 64;

real* aligned_alloc_floats(std::size_t n) {
    std::size_t bytes = ((n * sizeof(real) + kAlign - 1) / kAlign) * kAlign;
#if defined(_MSC_VER)
    void* p = _aligned_malloc(bytes, kAlign);
#else
    void* p = std::aligned_alloc(kAlign, bytes);
#endif
    if (!p) throw std::bad_alloc();
    return static_cast<real*>(p);
}
}  // namespace

void Matrix::AlignedDelete::operator()(real* p) const noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

Matrix::Matrix(std::initializer_list<std::initializer_list<real>> init) {
    std::size_t r = init.size(), c = r ? init.begin()->size() : 0;
    resize(r, c);
    std::size_t i = 0;
    for (const auto& row : init) {
        if (row.size() != c) throw std::invalid_argument("Matrix: ragged initializer list");
        for (real v : row) m_data[i++] = v;
    }
}

Matrix& Matrix::operator=(const Matrix& o) {
    if (this == &o) return *this;
    resize(o.m_rows, o.m_cols);
    if (size()) std::memcpy(m_data.get(), o.m_data.get(), size() * sizeof(real));
    return *this;
}

void Matrix::resize(std::size_t rows, std::size_t cols) {
    std::size_t n = rows * cols;
    if (n > m_capacity) {
        m_data.reset(aligned_alloc_floats(n));
        m_capacity = n;
    }
    m_rows = rows;
    m_cols = cols;
}

void Matrix::fill(real v) { std::fill(m_data.get(), m_data.get() + size(), v); }

void Matrix::gather_rows(const Matrix& src, const std::size_t* idx, std::size_t count) {
    resize(count, src.cols());
    const std::size_t c = src.cols();
    for (std::size_t i = 0; i < count; ++i)
        std::memcpy(row(i), src.row(idx[i]), c * sizeof(real));
}

Matrix Matrix::from_vector(const std::vector<real>& v, std::size_t rows, std::size_t cols) {
    if (v.size() != rows * cols) throw std::invalid_argument("Matrix::from_vector: size mismatch");
    Matrix m;
    m.resize(rows, cols);
    if (!v.empty()) std::memcpy(m.data(), v.data(), v.size() * sizeof(real));
    return m;
}

std::vector<std::size_t> argmax_rows(const Matrix& m) {
    std::vector<std::size_t> out(m.rows());
    for (std::size_t r = 0; r < m.rows(); ++r) {
        const real* p = m.row(r);
        out[r] = static_cast<std::size_t>(std::max_element(p, p + m.cols()) - p);
    }
    return out;
}

}  // namespace nn
