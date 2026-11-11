#include "nn/data.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace nn::data {

Matrix load_csv(const std::string& path, char delim, bool skip_header) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("nn::data: cannot open '" + path + "'");
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<real> values;
    values.reserve(buf.size() / 4);
    std::size_t cols = 0, rows = 0, line_no = 0;
    const char* p = buf.c_str();
    const char* end = p + buf.size();
    while (p < end) {
        const char* eol = static_cast<const char*>(std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
        if (!eol) eol = end;
        ++line_no;
        const char* q = p;
        while (q < eol && (*q == ' ' || *q == '\t' || *q == '\r')) ++q;
        if (q == eol || (skip_header && line_no == 1)) { p = eol + 1; continue; }
        std::size_t c = 0;
        while (q < eol) {
            char* next = nullptr;
            const float v = std::strtof(q, &next);
            if (next == q) throw std::runtime_error("nn::data: parse error in '" + path + "' line " + std::to_string(line_no));
            values.push_back(v);
            ++c;
            q = next;
            while (q < eol && (*q == ' ' || *q == '\t' || *q == '\r')) ++q;
            if (q < eol && *q == delim) ++q;
            else if (q < eol) throw std::runtime_error("nn::data: unexpected character in '" + path + "' line " + std::to_string(line_no));
        }
        if (cols == 0) cols = c;
        else if (c != cols) throw std::runtime_error("nn::data: inconsistent column count in '" + path + "' line " + std::to_string(line_no));
        ++rows;
        p = eol + 1;
    }
    return Matrix::from_vector(values, rows, cols);
}

std::pair<Matrix, Matrix> split_xy(const Matrix& m, int label_col) {
    if (m.cols() < 2) throw std::invalid_argument("nn::data::split_xy: need at least 2 columns");
    const std::size_t lc = label_col < 0 ? m.cols() + label_col : static_cast<std::size_t>(label_col);
    if (lc >= m.cols()) throw std::invalid_argument("nn::data::split_xy: label column out of range");
    Matrix x(m.rows(), m.cols() - 1), y(m.rows(), 1);
    for (std::size_t r = 0; r < m.rows(); ++r) {
        std::size_t k = 0;
        for (std::size_t c = 0; c < m.cols(); ++c) {
            if (c == lc) y(r, 0) = m(r, c);
            else x(r, k++) = m(r, c);
        }
    }
    return {std::move(x), std::move(y)};
}

std::size_t encode_labels(Matrix& labels, std::vector<real>* classes) {
    std::unordered_map<real, std::size_t> map;
    std::vector<real> order;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        auto it = map.find(labels[i]);
        if (it == map.end()) {
            it = map.emplace(labels[i], order.size()).first;
            order.push_back(labels[i]);
        }
        labels[i] = static_cast<real>(it->second);
    }
    if (classes) *classes = order;
    return order.size();
}

Matrix one_hot(const Matrix& labels, std::size_t k) {
    Matrix out(labels.rows(), k);
    for (std::size_t r = 0; r < labels.rows(); ++r) {
        const auto c = static_cast<std::size_t>(labels(r, 0));
        if (c >= k) throw std::invalid_argument("nn::data::one_hot: label out of range");
        out(r, c) = 1.0f;
    }
    return out;
}

Split train_test_split(const Matrix& x, const Matrix& y, real test_fraction, Rng& rng) {
    if (x.rows() != y.rows()) throw std::invalid_argument("nn::data::train_test_split: row mismatch");
    std::vector<std::size_t> idx(x.rows());
    std::iota(idx.begin(), idx.end(), 0);
    rng.shuffle(idx.begin(), idx.end());
    const std::size_t nt = static_cast<std::size_t>(std::round(x.rows() * test_fraction));
    const std::size_t ntr = x.rows() - nt;
    Split s;
    s.x_train.gather_rows(x, idx.data(), ntr);
    s.y_train.gather_rows(y, idx.data(), ntr);
    s.x_test.gather_rows(x, idx.data() + ntr, nt);
    s.y_test.gather_rows(y, idx.data() + ntr, nt);
    return s;
}

void MinMaxScaler::fit(const Matrix& x) {
    const std::size_t c = x.cols();
    min.assign(c, 0);
    range.assign(c, 1);
    if (x.rows() == 0) return;
    std::vector<real> mx(x.row(0), x.row(0) + c);
    min.assign(x.row(0), x.row(0) + c);
    for (std::size_t r = 1; r < x.rows(); ++r)
        for (std::size_t j = 0; j < c; ++j) {
            min[j] = std::min(min[j], x(r, j));
            mx[j] = std::max(mx[j], x(r, j));
        }
    for (std::size_t j = 0; j < c; ++j) range[j] = mx[j] - min[j] > 0 ? mx[j] - min[j] : 1.0f;
}

void MinMaxScaler::transform(Matrix& x) const {
    if (x.cols() != min.size()) throw std::invalid_argument("MinMaxScaler: column mismatch");
    for (std::size_t r = 0; r < x.rows(); ++r)
        for (std::size_t j = 0; j < x.cols(); ++j) x(r, j) = (x(r, j) - min[j]) / range[j];
}

void StandardScaler::fit(const Matrix& x) {
    const std::size_t c = x.cols(), n = x.rows();
    mean.assign(c, 0);
    inv_std.assign(c, 1);
    if (n == 0) return;
    std::vector<double> s(c, 0), sq(c, 0);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t j = 0; j < c; ++j) {
            s[j] += x(r, j);
            sq[j] += static_cast<double>(x(r, j)) * x(r, j);
        }
    for (std::size_t j = 0; j < c; ++j) {
        const double m = s[j] / n, var = sq[j] / n - m * m;
        mean[j] = static_cast<real>(m);
        inv_std[j] = var > 1e-12 ? static_cast<real>(1.0 / std::sqrt(var)) : 1.0f;
    }
}

void StandardScaler::transform(Matrix& x) const {
    if (x.cols() != mean.size()) throw std::invalid_argument("StandardScaler: column mismatch");
    for (std::size_t r = 0; r < x.rows(); ++r)
        for (std::size_t j = 0; j < x.cols(); ++j) x(r, j) = (x(r, j) - mean[j]) * inv_std[j];
}

}  // namespace nn::data
