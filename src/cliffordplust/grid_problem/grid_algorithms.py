# Copyright 2022-2023 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
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

"""
This file contains the functions solve_grid_problem_1d and solve_grid_problem_2d.

1) solve_grid_problem_1d
Given two real closed intervals A and B, solve_grid_problem_1d finds all the solutions alpha in 
the ring of quadratic integers with radicand 2, \u2124[\u221A2], such that alpha is in A and the \u221A2-conjugate 
of alpha is in B. This function is a sub-algorithm to solve 2D grid problems for upright
rectangles. For more information, see
Neil J. Ross and Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations, https://arxiv.org/pdf/1403.2975

2) solve_grid_problem_2d
Given two upright rectangles A and B in the complex plane, solve_grid_problem_2d finds all the solutions alpha in the ring of 
cyclotomic integers with radicand 2 \u2124[\u03C9] such that alpha is in A and the \u221A2-conjugate of alpha is in B. 
The solutions are used as candidates for the Clifford+T approximation of z-rotation gates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
import numbers as num

import numpy as np
from cliffordplust.rings import INVERSE_LAMBDA, LAMBDA, Zomega, Zsqrt2
from numpy.typing import NDArray

SQRT2 = math.sqrt(2)

__all__ = ["solve_grid_problem_1d", "solve_grid_problem_2d"]


def solve_grid_problem_1d(
    A: Sequence[float] | NDArray[np.floating], B: Sequence[float] | NDArray[np.floating]
) -> list[Zsqrt2]:
    """Solve the 1-dimensional grid problem for two intervals and return the result.

    Given two real closed intervals A and B, find all the solutions x in the ring \u2124[\u221A2] such that
    x \u2208 A and \u221A2-conjugate(x) \u2208 B.

    Args:
        A (Sequence[float, float]): (A0, A1): Bounds of the first interval.
        B (Sequence[float, float]): (B0, B1): Bounds of the second interval.

    Returns:
        list[Zsqrt2]: The list of solutions to the grid problem.

    Raises:
        TypeError: If intervals A and B are not real sequences of length 2.
    """
    for interval in (A, B):
        if not isinstance(interval, (Sequence, np.ndarray)):
            raise TypeError(
                f"Expected input intervals to be sequences of length 2 but got {interval}"
            )
        elif len(interval) != 2:
            raise TypeError(
                f"Expected input intervals to be sequences of length 2 but got {interval}"
            )
        elif not all([isinstance(bound, num.Real) for bound in interval]):
            raise TypeError("Interval limits must be real numbers.")
        elif interval[0] >= interval[1]:
            raise ValueError(f"Interval bounds must be in ascending order, but got {interval}.")

    # Definitions of the intervals and deltaA
    A_interval = np.array(A)
    B_interval = np.array(B)
    deltaA: float = A_interval[1] - A_interval[0]

    # Scaling of the intervals to have LAMBDA^-1 <= deltaA < 1
    n_scaling = math.floor(math.log(float(LAMBDA) * deltaA) / math.log(float(LAMBDA)))
    A_scaled = A_interval * float(LAMBDA) ** -n_scaling
    B_scaled = B_interval * float(-LAMBDA) ** n_scaling

    # Flip the interval if multiplied by a negative number
    if n_scaling % 2 == 1:
        B_scaled = np.flip(B_scaled)

    # Interval in which to find b (√2 coefficient of the ring element)
    b_interval_scaled: list[float] = [
        (A_scaled[0] - B_scaled[1]) / SQRT2**3,
        (A_scaled[1] - B_scaled[0]) / SQRT2**3,
    ]

    # Integers in the interval
    if math.isclose(b_interval_scaled[0], round(b_interval_scaled[0])):
        b_start = round(b_interval_scaled[0])
    else:
        b_start = math.ceil(b_interval_scaled[0])
    if math.isclose(b_interval_scaled[-1], round(b_interval_scaled[-1])):
        b_end = round(b_interval_scaled[-1])
    else:
        b_end = math.floor(b_interval_scaled[-1])

    # List of solutions
    alpha: list[Zsqrt2] = []
    for bi in range(b_start, b_end + 1):
        # Interval to look for a (Integer coefficient of the ring element)
        a_interval_scaled: list[float] = [
            A_scaled[0] - bi * SQRT2,
            A_scaled[1] - bi * SQRT2,
        ]
        for index, bound in enumerate(a_interval_scaled):
            if math.isclose(bound, round(bound)):
                a_interval_scaled[index] = round(bound)

        # If there is an integer if this interval
        if math.ceil(a_interval_scaled[0]) == math.floor(a_interval_scaled[1]):
            ai = math.ceil(a_interval_scaled[0])
            alpha_i_scaled = Zsqrt2(a=ai, b=bi)

            # Compute the unscaled solution
            alpha_i: Zsqrt2 = alpha_i_scaled * (
                lambda n_scaling: LAMBDA if n_scaling >= 0 else INVERSE_LAMBDA
            )(n_scaling) ** abs(n_scaling)
            fl_alpha_i = float(alpha_i)
            fl_alpha_i_conjugate = float(alpha_i.sqrt2_conjugate())

            # See if the solution is a solution to the unscaled grid problem for A and B
            if (
                fl_alpha_i >= A[0]
                and fl_alpha_i <= A[1]
                and fl_alpha_i_conjugate >= B[0]
                and fl_alpha_i_conjugate <= B[1]
            ):
                alpha.append(alpha_i)
    return alpha


def solve_grid_problem_2d(
    A: Sequence[Sequence[float]] | NDArray[np.floating],
    B: Sequence[Sequence[float]] | NDArray[np.floating],
) -> list[Zomega]:
    """
    Solve the 2-dimensional grid problem for two upright rectangle and return the result.

    Given two real 2D closed sets A and B of the form [a b] x [c, d], find all the solutions x in \u2124[\u03C9] such that x \u2208 A and \u221a2-conjugate(x) \u2208 B.

    Args:
        A (Sequence[Sequence[float, float]]): ((A0, A1), (A2, A3)): Bounds of the first upright rectangle. Rows correspond to the x and y axis respectively.
        B (Sequence[Sequence[float, float]]): ((B0, B1), (B2, B3)): Bounds of the second upright rectangle. Rows correspond to the x and y axis respectively.

    Returns:
        list[Zomega]: The list of solutions to the grid problem.

    Raises:
        TypeError: If intervals A and B are not real 2x2 matrices.
    """
    try:
        # Define the intervals for A and B rectangles.
        Ax: np.ndarray = np.asarray(A[0], dtype=float)
        Ay: np.ndarray = np.asarray(A[1], dtype=float)
        Bx: np.ndarray = np.asarray(B[0], dtype=float)
        By: np.ndarray = np.asarray(B[1], dtype=float)
        for interval in (Ax, Ay, Bx, By):
            if interval.shape != (2,):
                raise TypeError(
                    f"Input intervals must have two bounds (lower, upper) but received {interval}."
                )
            interval.sort()
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input intervals must be real 2x2 matrices.\nOrigin: {e}") from e

    # List of solutions.
    solutions: list[Zomega] = []

    # Solve two 1D grid problems for solutions of the form a + bi, where a and b are in Z[√2].
    alpha: list[Zsqrt2] = solve_grid_problem_1d(Ax, Bx)
    beta: list[Zsqrt2] = solve_grid_problem_1d(Ay, By)
    for a in alpha:
        for b in beta:
            solutions.append(Zomega(a=b.b - a.b, b=b.a, c=b.b + a.b, d=a.a))

    # Solve two 1D grid problems for solutions of the form a + bi + ω, where a and b are in Z[√2] and ω = (1 + i)/√2.
    alpha2: list[Zsqrt2] = solve_grid_problem_1d(Ax - 1 / SQRT2, Bx + 1 / SQRT2)
    beta2: list[Zsqrt2] = solve_grid_problem_1d(Ay - 1 / SQRT2, By + 1 / SQRT2)
    for a in alpha2:
        for b in beta2:
            solutions.append(Zomega(b.b - a.b, b.a, b.b + a.b + 1, a.a))

    return solutions
