#pragma once

#include <cmath>
#include <cstdint>

#include "config.hpp"

namespace nn {

/// xoshiro256** PRNG: fast, small state, excellent statistical quality.
class Rng {
public:
    explicit Rng(std::uint64_t seed = 0x9E3779B97F4A7C15ull) { reseed(seed); }

    void reseed(std::uint64_t seed) {
        for (auto& s : m_s) {  // splitmix64 seeding
            seed += 0x9E3779B97F4A7C15ull;
            std::uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            s = z ^ (z >> 31);
        }
        m_has_spare = false;
    }

    std::uint64_t next_u64() {
        const std::uint64_t result = rotl(m_s[1] * 5, 7) * 9;
        const std::uint64_t t = m_s[1] << 17;
        m_s[2] ^= m_s[0];
        m_s[3] ^= m_s[1];
        m_s[1] ^= m_s[2];
        m_s[0] ^= m_s[3];
        m_s[2] ^= t;
        m_s[3] = rotl(m_s[3], 45);
        return result;
    }

    /// Uniform in [0, 1)
    real uniform() { return static_cast<real>(next_u64() >> 40) * (1.0f / 16777216.0f); }
    real uniform(real lo, real hi) { return lo + (hi - lo) * uniform(); }
    /// Uniform integer in [0, n)
    std::uint64_t below(std::uint64_t n) {
#if defined(__SIZEOF_INT128__)
        __extension__ typedef unsigned __int128 u128;
        return static_cast<std::uint64_t>((static_cast<u128>(next_u64()) * n) >> 64);
#else
        return next_u64() % n;
#endif
    }
    /// Standard normal (Box-Muller, caches the second value)
    real normal() {
        if (m_has_spare) { m_has_spare = false; return m_spare; }
        real u, v, s;
        do {
            u = uniform(-1.0f, 1.0f);
            v = uniform(-1.0f, 1.0f);
            s = u * u + v * v;
        } while (s >= 1.0f || s == 0.0f);
        const real f = std::sqrt(-2.0f * std::log(s) / s);
        m_spare = v * f;
        m_has_spare = true;
        return u * f;
    }
    real normal(real mean, real stddev) { return mean + stddev * normal(); }

    /// Fisher-Yates shuffle
    template <class It>
    void shuffle(It first, It last) {
        auto n = last - first;
        for (decltype(n) i = n - 1; i > 0; --i) {
            auto j = static_cast<decltype(n)>(below(static_cast<std::uint64_t>(i) + 1));
            std::swap(first[i], first[j]);
        }
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t m_s[4];
    real m_spare = 0;
    bool m_has_spare = false;
};

}  // namespace nn
