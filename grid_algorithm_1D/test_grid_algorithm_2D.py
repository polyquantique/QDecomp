from random import randint as rint

import matplotlib.pyplot as plt
import pytest

from grid_algorithm_2D import plot_solutions, solve_grid_problem_2D
from Rings import Zomega


@pytest.mark.parametrize(
    "A", [([rint(-25, 25), rint(-25, 25)], [rint(-25, 25), rint(-25, 25)]) for _ in range(2)]
)
@pytest.mark.parametrize(
    "B", [([rint(-25, 25), rint(-25, 25)], [rint(-25, 25), rint(-25, 25)]) for _ in range(53)]
)
def test_grid_algorithm_2d(A, B) -> None:
    """Test the validity of the solutions when solving 2D-grid-problems."""
    A[0].sort()
    A[1].sort()
    B[0].sort()
    B[1].sort()
    s = solve_grid_problem_2D(A, B)

    for solution in s:
        assert (
            solution.real() <= A[0][1]
            and solution.real() >= A[0][0]
            and solution.imag() <= A[1][1]
            and solution.imag() >= A[1][0]
        )
        assert (
            solution.sqrt2_conjugate().real() <= B[0][1]
            and solution.sqrt2_conjugate().real() >= B[0][0]
            and solution.sqrt2_conjugate().imag() <= B[1][1]
            and solution.sqrt2_conjugate().imag() >= B[1][0]
        )
        assert isinstance(solution, Zomega)


def test_plot_grid_problem():
    """
    Test the plot_solutions() function.
    """
    plt.switch_backend("Agg")  # To test a function that creates a plot.
    A = ((1, 2), (3, 4))
    B = ((1, 2), (3, 4))
    solutions = solve_grid_problem_2D(A=A, B=B)
    plot_solutions(A, B, solutions, show=False)
    plot_solutions(A, B, solutions, show=True)
    assert True  # The code has run so far
