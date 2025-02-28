#include <stdexcept>

#include "..\..\src\cliffordplust\rings\cpp\Rings.hpp"
#include "..\..\src\cliffordplust\diophantine\cpp\diophantine_equation.cpp"


void test_integer_fact() {
    long long int res;
    for (int i=2; i<21; i++) {
        std::cout << "Testing integer_fact() for " << i << " ...";
        res = 1;

        std::vector<std::tuple<long long int, unsigned short>> factors = int_fact(i);
        for (auto [pi, mi] : factors) {
            res *= static_cast<long long int>(std::pow(pi, mi));
        }

        if (res != i) {
            throw std::runtime_error("Error in int_fact. Expected " + std::to_string(i) + " but got " + std::to_string(res));
        }
        std::cout << " OK!" << std::endl;
    }
    std::cout << std::endl;
}

void test_xi_fact() {
    Zsqrt2 xi_calculated(0, 0);
    for (int a=-1; a<20; a++) {
    for (int b=-5; b<5; b++) {
        xi_calculated = Zsqrt2(1, 0);
        Zsqrt2 xi(a, b);

        std::cout << "Testing xi_fact() for " << xi.to_string() << " ...";

        if (xi == 0) {
            std::cout << " OK!" << std::endl;
            continue;
        }

        std::vector<std::tuple<long long int, Zsqrt2, unsigned short>> factors = xi_fact(xi);

        for (auto [pi, xi_i, mi] : factors) {
            xi_calculated = xi_calculated * xi_i.pow(mi);
        }

        if (xi == 0 and xi_calculated != 0) {
            throw std::runtime_error("Error in xi_fact. Expected 0 but got " + xi_calculated.to_string());
        } else if (not (xi || xi_calculated)) {
            throw std::runtime_error("Error in xi_fact. xi and xi_calculated are not similar. xi = " + xi.to_string() + ", xi_calculated = " + xi_calculated.to_string());
        }
        std::cout << " OK!" << std::endl;
    }
    }
    std::cout << std::endl;
}

void test_pi_fact_into_xi() {
    std::vector<int> pi_values = {  // List of prime numbers
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101
    };

    for (int pi : pi_values) {
        std::cout << "Testing pi_fact_into_xi() for " << pi << " ...";
        switch (pi % 8) {
            case 2:
            {
                Zsqrt2 xi = pi_fact_into_xi(pi);
                if (xi != Zsqrt2(0, 1)) {
                    throw std::runtime_error("Error in pi_fact_into_xi. Expected 0 + \u221A2 but got " + xi.to_string());
                }
            } break;

            case 1:
            case 7:
            {
                Zsqrt2 xi = pi_fact_into_xi(pi);
                if (xi * xi.sqrt2_conjugate() != pi) {
                    throw std::runtime_error("Error in pi_fact_into_xi.");
                }
            } break;

            case 3:
            case 5:
            {
                Zsqrt2 xi = pi_fact_into_xi(pi);
                if (xi != pi) {
                    throw std::runtime_error("Error in pi_fact_into_xi. Expected " + std::to_string(pi) + ", but got " + xi.to_string());
                }
            } break;

            default:
                throw std::invalid_argument("pi must be a prime number. Got " + std::to_string(pi));
        }
        std::cout << " OK!" << std::endl;
    }
    std::cout << std::endl;
}

void test_xi_i_fact_into_ti() {
    std::vector<int> n_list = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101
    };

    for (int n : n_list) {
        std::cout << "Testing xi_i_fact_into_ti() for " << n << " ...";

        Zsqrt2 xi_i = pi_fact_into_xi(n);

        Zomega xi_i_fact = xi_i_fact_into_ti(n, xi_i);


        if (n % 8 == 7) {
            if (xi_i_fact != Zomega(0, 0, 0, 0)) {
                throw std::runtime_error("Error in xi_i_fact_into_ti. Expected 0 (no solution) but got " + xi_i_fact.to_string());
            }
        } else {
            Zsqrt2 xi_i_calculated = (xi_i_fact * xi_i_fact.complex_conjugate()).to_Zsqrt2();
            if (not (xi_i_calculated || xi_i)) {
                throw std::runtime_error(
                    "Error in xi_i_fact_into_ti. xi_i and xi_i_calculated are not similar. xi_i = " 
                    + xi_i.to_string() + ", xi_i_calculated = " + xi_i_calculated.to_string()
                );
            } 
        }
        std::cout << " OK!" << std::endl;
    }
    std::cout << std::endl;
}

void test_solve_usquare_eq_a_mod_p() {
    std::vector<int> p_list = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101
    };

    for (int p : p_list) {
        std::cout << "Testing solve_usquare_eq_a_mod_p() for " << p << " ...";

        int a = 0;
        switch (p % 8) {
            case 2: break;  // No solution
            case 7: break;  // No solution

            case 1: a = 1; break;
            case 5: a = 1; break;

            case 3: a = 2; break;
        }

        if (a == 0) {  // No solution
            std::cout << " OK!" << std::endl;
            continue;
        }

        long long int u = solve_usquare_eq_a_mod_p(a, p);
        if ((u * u) % p != -a + p) {
            throw std::runtime_error("Error in solve_usquare_eq_a_mod_p. Expected u^2 = " + std::to_string((u*u)%p+p) + " (mod p) but got " + std::to_string(-a+p));
        }

        std::cout << " OK!" << std::endl;
    }
    std::cout << std::endl;
}

void test_solve_xi_sim_ttdag_in_z() {
    for (int a=-1; a<21; a++) {
    for (int b=-5; b<6; b++) {
        Zsqrt2 xi(a, b);
        std::cout << "Testing solve_xi_sim_ttdag_in_z() for xi = " << xi.to_string() << " ...";

        if (xi == 0) {  // No solution
            std::cout << " OK!" << std::endl;
            continue;
        }

        Zomega t = solve_xi_sim_ttdag_in_z(xi);

        if (t != 0) {
            Zsqrt2 recombination = (t * t.complex_conjugate()).to_Zsqrt2();
            if (not (recombination || xi)) {
                throw std::runtime_error(
                    "Error in solve_xi_sim_ttdag_in_z. xi and recombination are not similar. xi = " 
                    + xi.to_string() + ", recombination = " + recombination.to_string()
                );
            }
        }
        std::cout << " OK!" << std::endl;
    }
    }
    std::cout << std::endl;
}

void test_solve_xi_eq_ttdag_in_d() {
    int n = 0;
    for (long long int a=-1; a<1000; a++) {
    for (unsigned short a_=0; a_<10; a_++) {
    for (long long int b=-5; b<400; b++) {
    for (unsigned short b_=0; b_<10; b_++) {
        Dsqrt2 xi(a, a_, b, b_);
        // std::cout << "Testing solve_xi_eq_ttdag_in_d() for xi = " << xi.to_string() << " ...";

        if (xi == 0) {
            // std::cout << " OK!" << std::endl;
            continue;
        }

        Domega t = solve_xi_eq_ttdag_in_d(xi);

        if (t != 0) {
            Dsqrt2 recombination = (t * t.complex_conjugate()).to_Dsqrt2();
            if (xi != recombination) {
                throw std::runtime_error(
                    "Error in solve_xi_eq_ttdag_in_d. xi and its recombination are not equal. xi = " 
                    + xi.to_string() + ", recombination = " + recombination.to_string()
                );
            }
            if (xi.to_float() < 0 or xi.sqrt2_conjugate().to_float() < 0) {
                throw std::runtime_error(
                    "Error in solve_xi_eq_ttdag_in_d. xi is not doubly positive. xi = " 
                    + xi.to_string()
                );
            }
        }
        // std::cout << " OK!" << std::endl;
    }
    }
    }
    std::cout << a << std::endl;
    }
    std::cout << std::endl;
}


int main() {
    test_integer_fact();
    test_xi_fact();
    test_pi_fact_into_xi();
    test_xi_i_fact_into_ti();
    test_solve_usquare_eq_a_mod_p();
    test_solve_xi_sim_ttdag_in_z();
    test_solve_xi_eq_ttdag_in_d();

    return 0;
}
