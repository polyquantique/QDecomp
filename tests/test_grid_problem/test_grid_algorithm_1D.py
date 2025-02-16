# Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.random import uniform

from cliffordplust.grid_problem import plot_grid_problem, solve_grid_problem_1d
from cliffordplust.rings import Zsqrt2


@pytest.mark.parametrize("A", [(1, 2, 3), "1, 2", (1.0j, 2), "12"])
def test_grid_algorithm_errors(A):
    """Test the raise of type errors when solving the 1D grid problem if the input intervals are not valid."""
    with pytest.raises(TypeError, match="Input intervals must be real sequences of length 2"):
        solve_grid_problem_1d(A, (-1, 1))
    with pytest.raises(TypeError, match="Input intervals must be real sequences of length 2"):
        solve_grid_problem_1d((-1, 1), A)


@pytest.mark.parametrize(
    "A",
    [np.sort(uniform(-50, 50, 2)) for _ in range(10)]
    + [
        [0, 1],
        [-1, 0],
        [0, 0.9],
        [-0.9, 0],
        [0, 0.4],
        [-0.4, 0],
        [-1, 1],
        [-5, 5],
        [-10, 10],
        [-10, 0],
        [0, 10],
    ],
)
@pytest.mark.parametrize(
    "B",
    [np.sort(uniform(-50, 50, 2)) for _ in range(9)]
    + [[-1, 1], [-5, 5], [-10, 10], [-10, 0], [0, 10]],
)
def test_grid_algorithm_1D_solutions(A, B):
    """Test the validity of the solutions found for the 1D grid problem for A and B."""
    solutions = solve_grid_problem_1d(A, B)
    if len(solutions) > 0:
        assert all(
            [
                (
                    float(solution) <= A[1]
                    and float(solution) >= A[0]
                    and float(solution.sqrt2_conjugate()) <= B[1]
                    and float(solution.sqrt2_conjugate()) >= B[0]
                )
                for solution in solutions
            ]
        )


@pytest.mark.parametrize("not_subscriptable", [1, 1.0, True, {1, 2}])
def test_indexable_type_error_plot_function(not_subscriptable) -> None:
    """Test the raise of type errors when plotting if the given intervals are not subscriptable."""
    with pytest.raises(TypeError, match="Expected input intervals to be sequences of length 2"):
        plot_grid_problem(not_subscriptable, (1, 2), [])
    with pytest.raises(TypeError, match="Expected input intervals to be sequences of length 2"):
        plot_grid_problem((1, 2), not_subscriptable, [])
    with pytest.raises(TypeError, match="Expected input intervals to be sequences of length 2"):
        plot_grid_problem(not_subscriptable, not_subscriptable, [])
    with pytest.raises(TypeError, match="Expected solutions to be subscriptable"):
        plot_grid_problem([1, 2], [1, 2], not_subscriptable)


@pytest.mark.parametrize(
    "interval", [(1, 2, 3), (1,), [1, 2, 3], [1], "123", "1", np.array([1, 2, 3]), np.array([1])]
)
def test_len_type_error_plot_function(interval: Sequence[float | int]) -> None:
    """Test the raise of type errors when plotting if giving intervals that are not of length 2."""
    with pytest.raises(TypeError, match="Expected input intervals to be sequences of length 2"):
        plot_grid_problem(interval, (1, 2), [])
    with pytest.raises(TypeError, match="Expected input intervals to be sequences of length 2"):
        plot_grid_problem((1, 2), interval, [])
    with pytest.raises(TypeError, match="Expected input intervals to be sequences of length 2"):
        plot_grid_problem(interval, interval, [])


@pytest.mark.parametrize("a", [1 + 1.0j, None, "1", (1,)])
def test_float_type_error_plot_function(a: Any) -> None:
    """Test the raise of a type error when plotting if the interval limits are not real numbers"""
    A: tuple = (1, 2)
    B: tuple = (a, 2)
    C: tuple = (1, a)
    D: tuple = (a, a)
    for arg1 in (A, B, C, D):
        for arg2 in (A, B, C, D):
            if not (arg1 == A and arg2 == A):
                with pytest.raises(TypeError, match="Interval limits must be real numbers."):
                    plot_grid_problem(arg1, arg2, [])


@pytest.mark.parametrize("interval", [(0, -1), (1, 0), (1, 1)])
def test_interval_ascending_value_error_plot_function(interval: Sequence[float]) -> None:
    """Test the raise of value error when plotting if the intervals limits are not in ascending order."""
    with pytest.raises(ValueError, match="Interval bounds must be in ascending order"):
        plot_grid_problem((1, 2), interval, [])
    with pytest.raises(ValueError, match="Interval bounds must be in ascending order"):
        plot_grid_problem(interval, (1, 2), [])
    with pytest.raises(ValueError, match="Interval bounds must be in ascending order"):
        plot_grid_problem(interval, interval, [])


@pytest.mark.parametrize("solutions", [(Zsqrt2(1, 1), 1), (1, Zsqrt2(1, 1)), (1, 1)])
def test_solutions_type_error_plot_function(solutions: Sequence[Any]) -> None:
    """Test the raise of a type error if the given solutions are not Zsqrt objects"""
    with pytest.raises(TypeError, match="Solutions must be Zsqrt2 objects"):
        plot_grid_problem([1, 2], [1, 2], solutions)


def test_plot_grid_problem():
    """
    Test the plot_grid_problem() function.
    """
    plt.switch_backend("Agg")  # To test a function that creates a plot.
    A = (1, 2)
    B = (1, 2)
    solutions = solve_grid_problem_1d(A=A, B=B)
    plot_grid_problem(A, B, solutions, show=False)
    plot_grid_problem(A, B, solutions, show=True)
    assert True  # The code has run so far
