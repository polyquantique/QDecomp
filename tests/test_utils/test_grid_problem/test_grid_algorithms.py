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


from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import mpmath as mp
import pytest

from qdecomp.plot import plot_grid_problem_1d, plot_grid_problem_2d
from qdecomp.rings import Zomega, Zsqrt2
from qdecomp.utils.grid_problem import solve_grid_problem_1d, solve_grid_problem_2d


@pytest.mark.parametrize("A", [(1, 2, 3), "1, 2", [mp.mpf(1.5)], [[1, 2]]])
def test_1d_grid_algorithm_type_error_bounds(A):
    """Test the raise of a TypeError when solving the 1D grid problem if the input intervals don't have two bounds."""
    with pytest.raises(TypeError, match="Input intervals must have two bounds"):
        list(solve_grid_problem_1d(A, (-1, 1)))
    with pytest.raises(TypeError, match="Input intervals must have two bounds"):
        list(solve_grid_problem_1d((-1, 1), A))

@pytest.mark.parametrize("A", [("1", "2"),  (1.0j, 2), [mp.mpc(1, 1), mp.mpc(2, 2)]])
def test_1d_grid_algorithm_type_error_real(A):
    """Test the raise of a TypeError when solving the 1D grid problem if the input intervals are not real numbers."""
    with pytest.raises(TypeError, match="The bounds of the interval must be real numbers"):
        list(solve_grid_problem_1d(A, (-1, 1)))
    with pytest.raises(TypeError, match="The bounds of the interval must be real numbers"):
        list(solve_grid_problem_1d((-1, 1), A))


@pytest.mark.parametrize(
    "A",
    [
        [0, 1],
        [-1, 0],
        [-0.8, 0],
        [0, 0.35],
        [-0.35, 0],
        [-1, 1],
        [-10, 10],
        [-47.9, 12.4],
        [33.33, 44.44],
        [mp.mpf("-9"), mp.mpf("2.00000000000001")],
    ],
)
@pytest.mark.parametrize(
    "B",
    [
        [0, 1],
        [-1, 0],
        [0, 0.8],
        [0, 0.35],
        [-0.35, 0],
        [-1, 1],
        [-10, 10],
        [-47.9, 12.4],
        [33.33, 44.44],
        [mp.pi, mp.pi + 5],
    ],
)
def test_grid_algorithm_1D_solutions(A, B):
    """Test the validity of the solutions found for the 1D grid problem for A and B."""
    solutions = solve_grid_problem_1d(A, B)
    for solution in solutions:
        assert (
            float(solution) >= A[0]
            and float(solution) <= A[1]
            and float(solution.sqrt2_conjugate()) >= B[0]
            and float(solution.sqrt2_conjugate()) <= B[1]
        )


@pytest.mark.parametrize(
    "A",
    [
        ((1, 2, 3), (1, 2)),
        [[1, 2], [1, 2, 3]],
        [[1, 1], [mp.mpc(1, 1)]],
        (1, 2, 3, 4)
    ],
)
def test_2d_grid_algorithm_type_error_bounds(A):
    """Test the raise of a TypeError when solving the 2D grid problem if the input intervals don't have two bounds."""
    with pytest.raises(TypeError, match="Input intervals must have two bounds"):
        list(solve_grid_problem_2d(A, ((-1, 1), (-1, 1))))
    with pytest.raises(TypeError, match="Input intervals must have two bounds"):
        list(solve_grid_problem_2d(((-1, 1), (-1, 1)), A))

@pytest.mark.parametrize(
    "A",
    [
        ((0, 1j), (1, 2)),
        ((1, 2), ([1], 2)),
        [[1, 1], [1, mp.mpc(1, 1)]],
    ],
)
def test_2d_grid_algorithm_type_error_real(A):
    """Test the raise of a TypeError when solving the 2D grid problem if the bounds are not real numbers."""
    with pytest.raises(TypeError, match="The bounds of the interval must be real numbers"):
        list(solve_grid_problem_2d(A, ((-1, 1), (-1, 1))))
    with pytest.raises(TypeError, match="The bounds of the interval must be real numbers"):
        list(solve_grid_problem_2d(((-1, 1), (-1, 1)), A))


@pytest.mark.parametrize(
    "A",
    list(
        combinations_with_replacement(
            [
                [-1, 1],
                [-23.5, -6.7],
                [-11.2, 0],
                [mp.mpf("-3.333333333333333"), mp.mpf("0.999999999999999")],
            ],
            2,
        )
    ),
)
@pytest.mark.parametrize(
    "B",
    list(
        combinations_with_replacement(
            [
                [-1, 1],
                [13.2, 27.7],
                [0, 19.9],
                [mp.mpf("-3.333333333333333"), mp.mpf("0.999999999999999")],
            ],
            2,
        )
    ),
)
def test_grid_algorithm_2d_solutions(A, B):
    """Test the validity of the solutions found for the 2D grid problem for A and B."""
    solutions = solve_grid_problem_2d(A, B)
    for solution in solutions:
        assert (
            solution.real() >= A[0][0]
            and solution.real() <= A[0][1]
            and solution.imag() >= A[1][0]
            and solution.imag() <= A[1][1]
            and solution.sqrt2_conjugate().real() >= B[0][0]
            and solution.sqrt2_conjugate().real() <= B[0][1]
            and solution.sqrt2_conjugate().imag() >= B[1][0]
            and solution.sqrt2_conjugate().imag() <= B[1][1]
        )


def test_plot_solutions_1d_type_errors_intervals():
    """Test the raise of a TypeError when plotting the solutions of the 1D grid problem if the input intervals are not of the correct form."""
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="Input intervals must be real sequences of length 2"):
        plot_grid_problem_1d(ax, (1, 2, 3), (-1, 1), [])
    with pytest.raises(TypeError, match="Input intervals must be real sequences of length 2"):
        plot_grid_problem_1d(ax, (1, 2), (1.0j, 0), [])
    with pytest.raises(TypeError, match="Input intervals must be real sequences of length 2"):
        plot_grid_problem_1d(ax, (1, 2), {1, 2}, [])


def test_plot_solutions_1d_type_errors_solutions():
    """Test the raise of a TypeError when plotting the solutions of the 1D grid problem if the solutions are not of the correct form."""
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="Solutions must be Zsqrt2 objects."):
        plot_grid_problem_1d(ax, (1, 2), (-1, 1), [1, 2, 3])


def test_plot_solutions_2d_type_errors_intervals():
    """Test the raise of a TypeError when plotting the solutions of the 2D grid problem if the input intervals are not of the correct form."""
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="Input intervals must be real 2 x 2 matrices"):
        plot_grid_problem_2d(ax, ((1, 2, 3), (1, 2)), ((-1, 1), (-1, 1)), [])
    with pytest.raises(TypeError, match="Input intervals must be real 2 x 2 matrices"):
        plot_grid_problem_2d(ax, ((1, 2), (1, 2)), ((1.0j, 0), (1, 2)), [])


def test_plot_solutions_2d_type_errors_solutions():
    """Test the raise of a TypeError when plotting the solutions of the 2D grid problem if the solutions are not of the correct form."""
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="Solutions must be Zomega objects."):
        plot_grid_problem_2d(ax, ((1, 2), (1, 2)), ((-1, 1), (-1, 1)), [1, 2, 3])


def test_plot_solutions_1d():
    """Test the plotting of the solutions of the 1D grid problem."""
    fig, ax = plt.subplots()
    plot_grid_problem_1d(ax, (-1, 1), (-1, 1), [Zsqrt2(0, 0), Zsqrt2(-4, 5), Zsqrt2(2, -1)])
    assert True


def test_plot_solutions_2d():
    """Test the plotting of the solutions of the 2D grid problem."""
    fig, ax = plt.subplots()
    plot_grid_problem_2d(
        ax,
        ((-1, 1), (-1, 1)),
        ((-1, 1), (-1, 1)),
        [Zomega(0, 0, 0, 0), Zomega(-4, 5, 0, 1), Zomega(2, -1, 1, 0)],
    )
    assert True
