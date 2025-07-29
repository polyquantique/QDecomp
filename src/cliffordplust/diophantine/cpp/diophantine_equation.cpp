#include <vector>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <future>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <boost/multiprecision/cpp_int.hpp>

#include "..\..\rings\cpp\Rings.hpp"
#include "diophantine_equation.hpp"
#include "tonelli_shanks.hpp"


template <typename T>
T sqrt_generic(T const n) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::sqrt(n));
    } else {  // Newton method for arbitrary precision numbers
        if (n == 0 || n == 1) { return n; }

        T x = n / 2;
        T last_x = 0;

        while (x != last_x) {
            last_x = x;
            x = (x + n / x) / 2;

            if (x > last_x) {
                // To avoid infinite loop in case of non-convergence
                return last_x;
            }
        }

        return x;
    }
}

template <typename T>
bool is_square(T n) {
    // Check if n is a square
    if (n < 0) {return 0;}
    
    // A square must have a modulo 16 of 0, 1, 4 or 9
    short int mod = static_cast<short int>(n % 16);
    if (mod != 0 and mod != 1 and mod != 4 and mod != 9) {return false;}

    // Check if n is a square
    T sqrt_n = sqrt_generic<T>(n);
    return sqrt_n * sqrt_n == n;
}

template <typename T>
T solve_usquare_eq_a_mod_p(T a, T p) {
    // Solve the equation u^2 = -a (mod p) <=> u^2 = q * p - a
    if (p == 1 and a == 1) {
        return (T)1;  // Special case for p = 1
    }

    // Use the Tonelli-Shanks algorithm to find the square root of -a modulo p
    return tonelli_shanks_algo<T>(-a, p);
}


template <typename T>
std::vector<std::tuple<T, unsigned int>> int_fact(T n) {
    // Factorize an integer n
    if (n < 1) {throw std::invalid_argument("n must be greater than 1");}
    
    std::vector<std::tuple<T, unsigned int>> factors;

    // Factorize by 2
    unsigned int counter = 0;
    while ( !(n & 1) ) {
        n = (n >> 1);
        counter++;
    }
    if (counter) {factors.push_back({2, counter});}

    // Factorize by odd numbers
    T factor = 3;
    while (n > 1 and factor * factor <= n) {
        counter = 0;

        while (n % factor == 0) {
            n /= factor;
            counter++;
        }
        
        if (counter) {factors.push_back({factor, counter});}
        factor += 2;
    }

    // If n != 1 at this point, it is a prime number
    if (n > 1) {factors.push_back({n, 1});}

    return factors;
}

template <typename T>
std::vector<std::tuple<T, Zsqrt2<T>, unsigned int>> xi_fact(Zsqrt2<T> xi) {
    // Factorize a Zsqrt2 xi
    if (xi == 0) {throw std::invalid_argument("xi must be different from 0");}

    T p = (xi * xi.sqrt2_conjugate()).p();

    // If xi is a unit, it cannot be factorized
    if (p == 1 or p == -1) {
        return std::vector<std::tuple<T, Zsqrt2<T>, unsigned int>>{{p, xi, 1}};
    }

    // Vector of tuples (prime, exponent) to return
    std::vector<std::tuple<T, Zsqrt2<T>, unsigned int>> factors;

    // We can only factorize if p is positive. Since -1 is a prime number, we can ignore it because,
    // we want to factorize xi up to a prime.
    if (p < 0) {p = -p;}

    auto pi = int_fact(p);  // Factorize p
    for (auto [p_i, m_i] : pi) {
        // If p_i = 1 or -1, xi_i is a unit and we can ignore it.
        if ((p_i == 1) or (p_i == -1)) {continue;}

        switch (static_cast<int>(p_i % 8)) {
            // If p_i = 2, xi_i = sqrt(2)
            case 2:
                factors.push_back({2, Zsqrt2<T>(0, 1), m_i});
                break;

            // If p_i % 8 == 1 or 7, we the factorization ξ_i is of the form p_i = ξ_i * ξ_i⋅
            case 1:
            case 7:
            {
                Zsqrt2<T> xi_i = pi_fact_into_xi<T>(p_i);
                xi_i.unit_reduce();

                // Determine wether we need to add ξ_i or its conjugate to the factorization and how
                // many times
                Zsqrt2<T> xi_temp = xi;
                Zsqrt2<T> r(0, 0);
                unsigned int nb_fact;
                for (nb_fact=0; nb_fact<m_i+1; nb_fact++) {
                    std::tie(xi_temp, r) = euclidean_div(xi_temp, xi_i);
                    if (r != 0) {break;}
                }

                if (nb_fact != 0) {factors.push_back({p_i, xi_i, nb_fact});}
                if (nb_fact != m_i) {
                    factors.push_back({p_i, xi_i.sqrt2_conjugate(), m_i - nb_fact});
                }
                break;
            }
            
            // If p_i % 8 == 3 or 5, p_i is its own factorization in Z[\u221A2]
            case 3:
            case 5:
                factors.push_back({p_i, Zsqrt2<T>(p_i, 0), m_i >> 1});
                break;

            default:
                throw std::runtime_error(
                    "There has been an error while calculating the factorization of xi"
                );
        }
    }
   
    return factors;
}

template <typename T>
Zsqrt2<T> pi_fact_into_xi(T pi) {
    // Factorize an integer prime pi into a Zsqrt2 xi factor
    switch (static_cast<int>(pi % 8)) {
        case 2:
            return Zsqrt2<T>(0, 1);

        case 3:
        case 5:
            return Zsqrt2<T>(pi, 0);

        case 1:
        case 7:
        {
            T b = 1;
            if constexpr (std::is_integral_v<T>) {
                while ( not is_square<T>(pi + 2 * static_cast<T>(std::pow(b, 2))) ) {b++;}
                return Zsqrt2<T>(sqrt_generic<T>(pi + 2 * static_cast<T>(std::pow(b, 2))), b);
            } else {
                while ( not is_square<T>(pi + 2 * boost::multiprecision::pow(b, 2))) {b++;}
                return Zsqrt2<T>(sqrt_generic<T>(pi + 2 * boost::multiprecision::pow(b, 2)), b);
            }
        }

        default:
            std::ostringstream oss;
            oss << pi;
            throw std::invalid_argument("pi must be a prime number. Got " + oss.str());
    }
    return Zsqrt2<T>(0, 0);
}

template <typename T>
Zomega<T> xi_i_fact_into_ti(T pi, Zsqrt2<T> xi_i) {
    // Factorize a Zsqrt2 xi_i into a Zomega ti factor if possible
    switch (static_cast<int>(pi % 8)) {
        case 2:
            return Zomega<T>(0, 0, 1, 1);

        case 1:
        case 5:
        {
            T u = solve_usquare_eq_a_mod_p((T)1, pi);
            return gcd(xi_i.to_Zomega(), Zomega<T>(0, 1, 0, u));
        }

        case 3:
        {
            T u = solve_usquare_eq_a_mod_p((T)2, pi);
            return gcd(xi_i.to_Zomega(), Zomega<T>(1, 0, 1, u));
        }
        
        case 7:  // No solution possible
            return Zomega<T>(0, 0, 0, 0);

        default:
            std::ostringstream oss;
            oss << pi;
            throw std::invalid_argument("pi must be a prime number. Got " + oss.str());
    }
}


template <typename T>
Zomega<T> solve_xi_sim_ttdag_in_z(Zsqrt2<T> xi) {
    auto xi_fact_list = xi_fact<T>(xi);
    
    for (auto [pi, xi_i, m_i] : xi_fact_list) {
        if ((m_i & 1) and (pi % 8 == 7)) {return Zomega<T>(0, 0, 0, 0);}  // No solution possible
    }

    std::vector<std::future<Zomega<T>>> threads;
    for (auto [pi, xi_i, m_i] : xi_fact_list) {
        if (m_i & 1 and pi != -1) {
            threads.push_back(std::async(std::launch::async, xi_i_fact_into_ti<T>, pi, xi_i));
        }
    }

    Zomega<T> result(0, 0, 0, 1);
    unsigned int thread_nb = 0;
    for (auto [pi, xi_i, m_i] : xi_fact_list) {
        if (m_i & 1 and pi != -1) {
            result = result * (threads[thread_nb].get()).pow(m_i);
            thread_nb++;
        } else {
            result = result * (xi_i.pow(m_i >> 1)).to_Zomega();
        }
    }

    return result;
}

template <typename T>
Domega<T> solve_xi_eq_ttdag_in_d(Dsqrt2<T> xi) {
    // xi must be doubly positive for this equation to have a solution.
    if (xi.to_long_double() < 0 or xi.sqrt2_conjugate().to_long_double() < 0) {
        return Domega<T>(0, 0, 0, 0, 0, 0, 0, 0);
    }

    unsigned int l = (xi * xi.sqrt2_conjugate()).p().denom();
    Zsqrt2 xi_prime = (xi.sqrt2_multiply(l)).to_Zsqrt2();

    Zomega<T> s = solve_xi_sim_ttdag_in_z<T>(xi_prime);
    if (s == 0) {return Domega<T>(0, 0, 0, 0, 0, 0, 0, 0);}

    Domega<T> delta(0, 0, 0, 0, 1, 0, 1, 0);  // delta = 1 + omega
    Domega<T> delta_inv(-1, 1, 1, 1, -1, 1, 1, 1);  // Inverse of delta
    Domega<T> delta_inv_l = delta_inv.pow(l);  // delta^-l

    Domega<T> t = delta_inv_l * s.to_Domega();

    // Find u such that xi = u * t * t†, u = λ**2n, t = δ^-l * s, δ * δ† = sqrt(2) * λ
    // => xi = λ^2n (λ sqrt(2))^-l * s * s† = λ^(2n-l) * s * s†
    // => ln(xi) = (2n-l) ln(λ) - l ln(sqrt(2)) + ln (s * s†)
    // => 2n - l = ( ln(xi) + l ln(sqrt(2)) - ln(s * s†) ) / ln(λ)
    // => n = ( ln(xi) + l ln(sqrt(2)) - ln(s * s†) ) / (2 ln(λ)) + l / 2
    Zsqrt2<T> ss = (s * s.complex_conjugate()).to_Zsqrt2();
    long double sqrt2 = std::sqrt(static_cast<long double>(2));
    int n = static_cast<int>( std::round(
        (std::log(xi.to_long_double()) + l * std::log(sqrt2) - std::log(ss.to_long_double())) / 
        (2 * std::log(1 + sqrt2))
        + static_cast<float>(l) / 2
    ));

    // v**2 = u => v = λ**n
    Domega<T> v(0, 0, 0, 0, 0, 0, 0, 0);
    if (n > 0) {
        v = Zsqrt2<T>(1, 1).pow(static_cast<unsigned int>(n)).to_Domega();  // λ**n
    } else if (n == 0) {
        v = Domega<T>(0, 0, 0, 0, 0, 0, 1, 0);
    } else {
        v = Zsqrt2<T>(-1, 1).pow(static_cast<unsigned int>(-n)).to_Domega();  // (λ**-1)**n
    }

    return t * v;
}
