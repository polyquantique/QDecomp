#include "..\..\src\cliffordplust\grid_problem\cpp\grid_algorithms.cpp"


void test_grid_problem_1D_class() {
    std::cout << "Testing the GridProblem1D class ..." << std::endl;
    for (long long int amin=-10; amin<11; amin++) {
    for (long long int amax=amin; amax<11; amax++) {
    for (long long int bmin=-10; bmin<11; bmin++) {
    for (long long int bmax=bmin; bmax<11; bmax++) {
        // std::cout << "Testing the GridAlgorithm1D class for A = [" << amin << ", " << amax
        //     << "] and B = [" << bmin << ", " << bmax << "] ...";

        GridProblem1D gp(amin, amax, bmin, bmax);
        for (auto sol : gp) {
            double alpha_f = sol.to_float();
            double alpha_conj_f = sol.sqrt2_conjugate().to_float();

            if ((alpha_f >= amin) and (alpha_f <= amax) and
                (alpha_conj_f >= bmin) and (alpha_conj_f <= bmax)) {
                continue;
            } else {
                throw std::runtime_error(
                    "Error in GridProblem1D. The solution " + sol.to_string()
                    + " (or its conjugate) is not in the interval A (B)."
                );
            }
        }
        // std::cout << " OK!" << std::endl;
    }
    }
    }
    }
    std::cout << "Successfully tested the GridProblem1D class!" << std::endl << std::endl;
}


int main() {
    test_grid_problem_1D_class();

    return 0;
}
