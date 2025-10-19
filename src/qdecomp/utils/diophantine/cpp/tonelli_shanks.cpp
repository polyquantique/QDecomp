#include <stdexcept>
#include <sstream>

#include "tonelli_shanks.hpp"


template <typename T>
T mod_pow(T const &base, T const &exp, T const &mod) {
    T result = 1;
    T b = base % mod;
    T e = exp;

    while (e) {
        if (e & 1) {
            result = (result * b) % mod;
        }
        b = (b * b) % mod;
        e >>= 1;
    }

    return result;
}

template <typename T>
int legendre_symbol(T const &a, T const &p) {
    return (int)mod_pow(a, (p - (T)1) / 2, p);
}

template <typename T>
T tonelli_shanks_algo(T const &a, T const &p) {
    if (legendre_symbol(a, p) != 1) {
        std::ostringstream oss;
        oss << "a = " << a << " is not a quadratic residue modulo p = " << p << ".";
        throw std::runtime_error(oss.str());
    }

    if (p % 4 == 3) {
        T r = mod_pow(a, (p + (T)1) / 4, p);
        return std::min(r, (T)(p-r));  // Return the smallest non-negative root
    }

    // Decomposition of p-1 = q * 2^s
    T s = 0;
    T q = p - 1;
    while (q % 2 == 0) {
        s++;
        q >>= 1;
    }

    // Find a quadratic non-residue z
    T z = 2;
    while (legendre_symbol(z, p) != p - 1) {
        z++;
    }

    T c = mod_pow(z, q, p);
    T r = mod_pow(a, (q + (T)1) / 2, p);
    T t = mod_pow(a, q, p);
    T m = s;

    while (t != 1) {
        T i = 1;
        T temp = mod_pow(t, (T)2, p);
        while (temp != 1) {
            temp = mod_pow(temp, (T)2, p);
            i++;
        }

        T b = mod_pow(c, (T)(std::pow(2, (long long int)(m - i - 1))), p);
        r = (r * b) % p;
        t = (t * mod_pow(b, (T)2, p)) % p;
        c = mod_pow(b, (T)2, p);
        m = i;
    }

    return std::min(r, (T)(p - r));  // Return the smallest non-negative root
}
