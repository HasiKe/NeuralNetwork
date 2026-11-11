#pragma once

// Internal binary (de)serialisation helpers. Little-endian hosts assumed.

#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

#include "nn/matrix.hpp"

namespace nn::io {

template <class T>
void put(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
T get(std::istream& is) {
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!is) throw std::runtime_error("nn: unexpected end of model stream");
    return v;
}

inline void put_size(std::ostream& os, std::size_t v) { put<std::uint64_t>(os, v); }
inline std::size_t get_size(std::istream& is) { return static_cast<std::size_t>(get<std::uint64_t>(is)); }

inline void put_matrix(std::ostream& os, const Matrix& m) {
    put_size(os, m.rows());
    put_size(os, m.cols());
    os.write(reinterpret_cast<const char*>(m.data()), static_cast<std::streamsize>(m.size() * sizeof(real)));
}

inline void get_matrix(std::istream& is, Matrix& m) {
    std::size_t r = get_size(is), c = get_size(is);
    if (r * c > (std::size_t(1) << 34)) throw std::runtime_error("nn: corrupt matrix header");
    m.resize(r, c);
    is.read(reinterpret_cast<char*>(m.data()), static_cast<std::streamsize>(m.size() * sizeof(real)));
    if (!is) throw std::runtime_error("nn: unexpected end of model stream");
}

inline void expect(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(std::string("nn: ") + msg);
}

}  // namespace nn::io
