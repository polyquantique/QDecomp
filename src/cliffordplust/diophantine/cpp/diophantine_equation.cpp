#include <vector>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <future>
#include <iostream>

#include "..\..\rings\cpp\Rings.hpp"
#include "diophantine_equation.hpp"


bool is_square(int n) {
    // Check if n is a square
    if (n < 0) {return 0;}
    
    // A square must have a modulo 16 of 0, 1, 4 or 9
    int mod = n % 16;
    if (mod != 0 and mod != 1 and mod != 4 and mod != 9) {return false;}

    // Check if n is a square
    int sqrt_n = std::sqrt(n);
    return sqrt_n * sqrt_n == n;
}

int solve_usquare_eq_a_mod_p(int a, int p) {
    // Solve the equation u^2 = a (mod p) <=> u^2 = q * p - a
    int x = p - a;
    while (not is_square(x)) {
        x += p;
    }
    return std::sqrt(x);
}


std::vector<std::tuple<int, int>> int_fact(int n) {
    // Factorize an integer n
    if (n < 1) {throw std::invalid_argument("n must be greater than 1");}
    
    std::vector<std::tuple<int, int>> factors;

    // Factorize by 2
    int counter = 0;
    while ( !(n & 1) ) {
        n = (n >> 1);
        counter++;
    }
    if (counter) {factors.push_back({2, counter});}

    // Factorize by odd numbers
    int factor = 3;
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

std::vector<std::tuple<int, Zsqrt2, int>> xi_fact(Zsqrt2 xi) {
    // Factorize a Zsqrt2 xi
    if (xi == 0) {throw std::invalid_argument("xi must be different from 0");}

    std::vector<std::tuple<int, Zsqrt2, int>> factors;

    int p = (xi * xi.sqrt2_conjugate()).p();

    // If xi is a prime number, it cannot be factorized
    if (p == 1 or p == -1) {return std::vector<std::tuple<int, Zsqrt2, int>>{{p, xi, 1}};}

    // We can only factorize if p is positive. Since -1 is a prime number, we can ignore it because,
    // we want to factorize xi up to a prime.
    if (p < 0) {p = -p;}

    auto pi = int_fact(p);  // Factorize p
    for (auto [p_i, m_i] : pi) {
        // If p_i = 1 or -1, xi_i is a unit and we can ignore it.
        if (p_i == 1) or (p_i == -1) {continue;}

        switch (p_i % 8) {
            // If p_i = 2, xi_i = sqrt(2)
            case 2:
                factors.push_back({2, Zsqrt2(0, 1), m_i});
                break;

            // If p_i % 8 == 1 or 7, we the factorization ξ_i is of the form p_i = ξ_i * ξ_i⋅
            case 1:
            case 7:
            {
                Zsqrt2 xi_i = pi_fact_into_xi(p_i);

                // Determine wether we need to add ξ_i or its conjugate to the factorization and how
                // many times
                Zsqrt2 xi_temp = xi;
                Zsqrt2 r(0, 0);
                int nb_fact;
                for (nb_fact=0; nb_fact<m_i+1; nb_fact++) {
                    std::tie(xi_temp, r) = euclidean_div(xi_temp, xi_i);
                    if (r != 0) {break;}
                }

                if (nb_fact != 0) {factors.push_back({p_i, xi_i, nb_fact});}
                if (nb_fact != m_i) {factors.push_back({p_i, xi_i.sqrt2_conjugate(), m_i - nb_fact});}
                break;
            }
            
            // If p_i % 8 == 3 or 5, p_i is its own factorization in Z[\u221A2]
            case 3:
            case 5:
                factors.push_back({p_i, Zsqrt2(p_i, 0), m_i >> 1});
                break;

            default:
                throw std::runtime_error("There has been an error while calculating the factorization of xi");
        }
    }
   
    return factors;
}

Zsqrt2 pi_fact_into_xi(int pi) {
    // Factorize an integer prime pi into a Zsqrt2 xi factor
    switch (pi % 8) {
        case 2:
            return Zsqrt2(0, 1);

        case 3:
        case 5:
            return Zsqrt2(pi, 0);

        case 1:
        case 7:
        {
            int b = 1;
            while ( not is_square(pi + 2 * std::pow(b, 2)) ) {b++;}

            return Zsqrt2( std::sqrt(pi + 2 * pow(b, 2)) , b);
        }

        default:
            throw std::invalid_argument("pi must be a prime number. Got " + std::to_string(pi));
    }
    return Zsqrt2(0, 0);
}

Zomega xi_i_fact_into_ti(int pi, Zsqrt2 xi_i) {
    // Factorize a Zsqrt2 xi_i into a Zomega ti factor if possible
    switch (pi % 8) {
        case 2:
            return Zomega(0, 0, 1, 1);

        case 1:
        case 5:
        {
            int u = solve_usquare_eq_a_mod_p(1, pi);
            return gcd(xi_i.to_Zomega(), Zomega(0, 1, 0, u));
        }

        case 3:
        {
            int u = solve_usquare_eq_a_mod_p(2, pi);
            return gcd(xi_i.to_Zomega(), Zomega(1, 0, 1, u));
        }
        
        case 7:  // No solution possible
            return Zomega(0, 0, 0, 0);

        default:
            throw std::invalid_argument("pi must be a prime number. Got " + std::to_string(pi));
    }
}


Zomega solve_xi_sim_ttdag_in_z(Zsqrt2 xi) {
    auto xi_fact_list = xi_fact(xi);
    
    for (auto [pi, xi_i, m_i] : xi_fact_list) {
        if ((m_i & 1) and (pi % 8 == 7)) {return Zomega(0, 0, 0, 0);}  // No solution possible
    }

    std::vector<std::future<Zomega>> threads;
    for (auto [pi, xi_i, m_i] : xi_fact_list) {
        if (m_i & 1 and pi != -1) {threads.push_back(std::async(std::launch::async, xi_i_fact_into_ti, pi, xi_i));}
    }

    Zomega result(0, 0, 0, 1);
    int thread_nb = 0;
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

Domega solve_xi_eq_ttdag_in_d(Dsqrt2 xi) {
    // xi must be doubly positive for this equation to have a solution.
    if (xi.to_float() < 0 or xi.sqrt2_conjugate().to_float() < 0) {
        return Domega(0, 0, 0, 0, 0, 0, 0, 0);
    }

    int l = (xi * xi.sqrt2_conjugate()).p().denom();
    Zsqrt2 xi_prime = (Dsqrt2(0, 0, 1, 0).pow(l) * xi).to_Zsqrt2();

    Zomega s = solve_xi_sim_ttdag_in_z(xi_prime);
    if (s == 0) {return Domega(0, 0, 0, 0, 0, 0, 0, 0);}

    Domega delta(0, 0, 0, 0, 1, 0, 1, 0);  // delta = 1 + omega
    Domega delta_inv(-1, 1, 1, 1, -1, 1, 1, 1);  // Inverse of delta
    Domega delta_inv_l = delta_inv.pow(l);  // delta^-l

    Domega t = delta_inv_l * s.to_Domega();
    Dsqrt2 tt = (t * t.complex_conjugate()).to_Dsqrt2();

    // Find u such that xi = u * t * t†
    D denom = (tt * tt.sqrt2_conjugate()).to_D();
    int num_ = denom.num();
    int denom_ = denom.denom();
    Zsqrt2 u_temp = (xi * tt.sqrt2_conjugate() * Dsqrt2(1 << denom_, 0, 0, 0)).to_Zsqrt2();
    Zsqrt2 u(u_temp.p() / num_, u_temp.q() / num_);


    // u is of the form u = λ**2n => n = ln(u) / 2 ln(λ)
    int n = std::round(std::log(u.to_float()) / (2 * std::log(1+std::sqrt(2))));

    // v**2 = u => v = λ**n
    Domega v(0, 0, 0, 0, 0, 0, 0, 0);
    if (n > 0) {
        v = Zsqrt2(1, 1).pow(n).to_Domega();  // λ**n
    } else if (n == 0) {
        v = Domega(0, 0, 0, 0, 0, 0, 1, 0);
    } else {
        v = Zsqrt2(-1, 1).pow(-n).to_Domega();  // (λ**-1)**n
    }

    return t * v;
}
