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


import matplotlib.pyplot as plt
import pytest
from itertools import combinations_with_replacement

from cliffordplust.grid_problem import solve_grid_problem_1d, solve_grid_problem_2d


@pytest.mark.parametrize("A", [(1, 2, 3), "1, 2", (1.0j, 2), "12", 1])
def test_1d_grid_algorithm_type_error(A):
    """Test the raise of a TypeError when solving the 1D grid problem if the input intervals are not of the correct form."""
    with pytest.raises(TypeError, match="Input intervals must be real sequences of length 2"):
        solve_grid_problem_1d(A, (-1, 1))
    with pytest.raises(TypeError, match="Input intervals must be real sequences of length 2"):
        solve_grid_problem_1d((-1, 1), A)


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
    ],
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


@pytest.mark.parametrize("A", [((1, 2, 3), (1, 2)),
                                [[1, 2], [1, 2, 3]],
                                ((0, 1j), (1, 2)),
                                ((1, 2), ([1], 2))])
def test_2d_grid_algorithm_type_error(A):
    """Test the raise of a TypeError when solving the 2D grid problem if the input intervals are not of the correct form."""
    with pytest.raises(TypeError, match="Input intervals must be real 2x2 matrices"):
        solve_grid_problem_2d(A, ((-1, 1), (-1, 1)))
    with pytest.raises(TypeError, match="Input intervals must be real 2x2 matrices"):
        solve_grid_problem_2d(((-1, 1), (-1, 1)), A)


@pytest.mark.parametrize("A", list(combinations_with_replacement([
        [-1, 1],
        [-10, 10],
        [-23.5, -6.7],
        [-11.2, 0],
    ], 2)))
@pytest.mark.parametrize("B", list(combinations_with_replacement([
        [-1, 1],
        [13.2, 27.7],
        [0, 19.9],
    ], 2)))
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


