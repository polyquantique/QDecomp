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

"""
The ``grid_algorithms`` module contains functions to solve 1D and 2D grid problems.

The module provides two main functions:

- :func:`solve_grid_problem_1d`: Given two real closed intervals `A` and `B`, :func:`solve_grid_problem_1d` finds all the solutions ``x`` in the ring of quadratic integers with radicand 2, :math:`\\mathbb{Z}[\\sqrt{2}]`, such that ``x`` is in A and the :math:`\\sqrt{2}`-conjugate of ``x`` is in B. This function is a sub-algorithm to solve 2D grid problems for upright rectangles.

- :func:`solve_grid_problem_2d`: Given two upright rectangles `A` and `B` in the complex plane, :func:`solve_grid_problem_2d` finds all the solutions ``x`` in the ring of cyclotomic integers with radicand 2, :math:`\\mathbb{Z}[\\omega]`, such that ``x`` is in A and the :math:`\\sqrt{2}`-conjugate of ``x`` is in B. The solutions are used as candidates for unitary matrix entries in the Clifford+T approximation of z-rotation gates.

For more information on solving 1D and 2D grid problems, see :cite:`grid_problem_ross`.
"""

from __future__ import annotations

import math
from collections.abc import Generator, Sequence
from numbers import Real

import mpmath as mp
import numpy as np
from numpy.typing import NDArray

from qdecomp.rings import INVERSE_LAMBDA, LAMBDA, Zomega, Zsqrt2

SQRT2 = mp.sqrt(2)

__all__ = ["solve_grid_problem_1d", "solve_grid_problem_2d"]


def solve_grid_problem_1d(
    A: Sequence[Real] | NDArray, B: Sequence[Real] | NDArray
) -> Generator[Zsqrt2]:
    """Solve the 1-dimensional grid problem for two intervals and return the result.

    Given two real closed sets `A` and `B`, this function finds all the solutions ``x`` in the ring :math:`\\mathbb{Z}[\\sqrt{2}]` such that
    ``x`` is in `A` and the :math:`\\sqrt{2}`-conjugate of ``x`` is in B.

    Args:
        A (Sequence[Real, Real]): (A0, A1): Bounds of the first interval.
        B (Sequence[Real, Real]): (B0, B1): Bounds of the second interval.

    Returns:
        Generator[Zsqrt2]: A generator of all solutions to the grid problem. The solutions are given as Zsqrt2 objects.

    Raises:
        TypeError: If intervals A and B are not real sequences of length 2.
    """
    # Convert the input intervals to numpy arrays
    A_interval: np.ndarray = np.asarray(A, dtype=object)
    B_interval: np.ndarray = np.asarray(B, dtype=object)

    if A_interval.shape != (2,) or B_interval.shape != (2,):
        raise TypeError(
            f"Input intervals must have two bounds (lower, upper) but received: {A if A_interval.shape != (2,) else B}."
        )
    for bound in np.concatenate((A_interval, B_interval)):
        if not isinstance(bound, Real):
            raise TypeError(
                f"The bounds of the interval must be real numbers but received: {bound}."
            )
    A_interval.sort()
    B_interval.sort()

    # Compute the width of the interval A
    deltaA = A_interval[1] - A_interval[0]

    # Scaling of the intervals to have LAMBDA^-1 <= deltaA < 1
    n_scaling = int(mp.floor(mp.log(LAMBDA.mpfloat() * deltaA) / mp.log(LAMBDA.mpfloat())))
    A_scaled = A_interval * LAMBDA.mpfloat() ** -n_scaling
    B_scaled = B_interval * (-LAMBDA).mpfloat() ** n_scaling

    # Flip the interval if multiplied by a negative number
    if n_scaling % 2 == 1:
        B_scaled = np.flip(B_scaled)

    # Interval in which to find b (√2 coefficient of the ring element)
    b_interval_scaled = [
        (A_scaled[0] - B_scaled[1]) / SQRT2**3,
        (A_scaled[1] - B_scaled[0]) / SQRT2**3,
    ]

    # Find the integer bounds of the interval to find candidates for b.
    # If a bound is close to an integer, round it to the nearest integer to avoid losing solutions because of numerical inaccuracy.
    if math.isclose(b_interval_scaled[0], mp.nint(b_interval_scaled[0]), rel_tol=1e-10):
        b_start = mp.nint(b_interval_scaled[0])
    else:
        b_start = mp.ceil(b_interval_scaled[0])
    if math.isclose(b_interval_scaled[-1], mp.nint(b_interval_scaled[-1]), rel_tol=1e-10):
        b_end = mp.nint(b_interval_scaled[-1])
    else:
        b_end = mp.floor(b_interval_scaled[-1])

    # Enumerate all solutions to the grid problem
    for bi in range(int(b_start), int(b_end) + 1):
        # Interval to look for a (Integer coefficient of the ring element)
        a_interval_scaled = [
            A_scaled[0] - bi * SQRT2,
            A_scaled[1] - bi * SQRT2,
        ]
        # If one of the bound is close to an integer, round it to the nearest integer
        # to avoid losing solutions because of numerical inaccuracy.
        for index, bound in enumerate(a_interval_scaled):
            if math.isclose(bound, mp.nint(bound), rel_tol=mp.mpf("1e-10")):
                a_interval_scaled[index] = mp.nint(bound)

        # If there is an integer in this interval, compute the scaled solution for ai and bi
        if mp.ceil(a_interval_scaled[0]) == mp.floor(a_interval_scaled[1]):
            ai = int(mp.ceil(a_interval_scaled[0]))
            alpha_i_scaled = Zsqrt2(a=ai, b=bi)

            # Compute the unscaled solution
            alpha_i: Zsqrt2 = alpha_i_scaled * (
                lambda n_scaling: LAMBDA if n_scaling >= 0 else INVERSE_LAMBDA
            )(n_scaling) ** abs(n_scaling)
            fl_alpha_i = alpha_i.mpfloat()
            fl_alpha_i_conjugate = alpha_i.sqrt2_conjugate().mpfloat()

            # See if the solution is a solution to the unscaled grid problem for A and B
            if A[0] <= fl_alpha_i <= A[1] and B[0] <= fl_alpha_i_conjugate <= B[1]:
                # Yield the solution
                yield alpha_i


def solve_grid_problem_2d(
    A: Sequence[Sequence[Real]] | NDArray,
    B: Sequence[Sequence[Real]] | NDArray,
) -> Generator[Zomega]:
    """Solve the 2-dimensional grid problem for two upright rectangles and return the result.

    Given two real 2D closed sets `A` and `B` of the form [a, b] x [c, d], find all the solutions ``x`` in the ring :math:`\\mathbb{Z}[\\omega]`
    such that ``x`` is in `A` and the :math:`\\sqrt{2}`-conjugate of ``x`` is in `B`.

    Args:
        A (Sequence[Sequence[Real, Real]]): [(A0, A1), (A2, A3)]: Bounds of the first upright rectangle. Rows correspond to the x and y axis respectively.
        B (Sequence[Sequence[Real, Real]]): [(B0, B1), (B2, B3)]: Bounds of the second upright rectangle. Rows correspond to the x and y axis respectively.

    Returns:
        Generator[Zomega]: A generator of all solutions to the grid problem. The solutions are given as Zomega objects.

    Raises:
        TypeError: If intervals A and B are not real 2 x 2 nested sequences.
    """

    # Define the intervals for A and B.
    Ax: np.ndarray = np.asarray(A[0], dtype=object)
    Ay: np.ndarray = np.asarray(A[1], dtype=object)
    Bx: np.ndarray = np.asarray(B[0], dtype=object)
    By: np.ndarray = np.asarray(B[1], dtype=object)

    for interval in (Ax, Ay, Bx, By):
        if interval.shape != (2,):
            raise TypeError(
                f"Input intervals must have two bounds (lower, upper) but received {interval}."
            )
    for bound in np.concatenate((Ax, Ay, Bx, By)):
        if not isinstance(bound, Real):
            raise TypeError(
                f"The bounds of the interval must be real numbers but received: {bound}."
            )
    for interval in (Ax, Ay, Bx, By):
        interval.sort()

    # Solve two 1D grid problems for solutions of the form a + bi, where a and b are in Z[√2].
    alpha: Generator[Zsqrt2] = solve_grid_problem_1d(Ax, Bx)
    beta: list[Zsqrt2] = list(solve_grid_problem_1d(Ay, By))
    for a in alpha:
        for b in beta:
            yield Zomega(a=b.b - a.b, b=b.a, c=b.b + a.b, d=a.a)

    # Solve two 1D grid problems for solutions of the form a + bi + ω, where a and b are in Z[√2] and ω = (1 + i)/√2.
    alpha2: Generator[Zsqrt2] = solve_grid_problem_1d(Ax - 1 / SQRT2, Bx + 1 / SQRT2)
    beta2: list[Zsqrt2] = list(solve_grid_problem_1d(Ay - 1 / SQRT2, By + 1 / SQRT2))
    for a in alpha2:
        for b in beta2:
            yield Zomega(b.b - a.b, b.a, b.b + a.b + 1, a.a)
