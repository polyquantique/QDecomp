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
The ``grid_algorithm_1D`` module contains functions to solve the 1D grid problem for two intervals A and B.

The module provides two main functions:

1) :func:`solve_grid_problem_1d`:
Given two real closed intervals A and B, :func:`solve_grid_problem_1d` finds all the solutions ``alpha`` in 
the ring of quadratic integers with radicand 2, :math:`\\mathbb{Z}[\\sqrt{2}]`, such that ``alpha`` is in A and the :math:`\\sqrt{2}`-conjugate 
of ``alpha`` is in B. This function is a sub-algorithm to solve 2D grid problems for upright
rectangles. 

2) :func:`plot_grid_problem`:
The function :func:`plot_grid_problem` plots the solutions of the grid problem for the intervals A and B and their 
conjugate on the real axis. The plot also contains the intervals A and B.

For more information on solving 1D grid problems, see [1]_.

.. [1] Neil J. Ross and Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations, https://arxiv.org/pdf/1403.2975.
"""

from __future__ import annotations

import math
import numbers as num
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from qdecomp.rings import INVERSE_LAMBDA, LAMBDA, Zsqrt2

__all__ = ["solve_grid_problem_1d", "plot_grid_problem"]


def solve_grid_problem_1d(A: Sequence[num.Real], B: Sequence[num.Real]) -> list[Zsqrt2]:
    """
    Solves the 1 dimensional grid problem for two sets and returns the result.

    Given two real closed sets A and B, this function finds all the solutions x in the ring :math:`\\mathbb{Z}[\\sqrt{2}]` such that
    x is in A and the :math:`\\sqrt{2}`-conjugate of x is in B.

    Args:
        A (Sequence[Real, Real]): Bounds of the first interval.
        B (Sequence[Real, Real]): Bounds of the second interval.

    Returns:
        list[Zsqrt2]: The list of solutions to the grid problem.

    Raises:
        TypeError: If A and B intervals are not real sequences of length 2.
    """
    try:
        A_interval: np.ndarray = np.asarray(A, dtype=float)
        B_interval: np.ndarray = np.asarray(B, dtype=float)
        if A_interval.shape != (2,) or B_interval.shape != (2,):
            raise TypeError(
                f"Input intervals must have two bounds (lower, upper) but received {A if A_interval.shape != (2,) else B}."
            )
        A_interval.sort()
        B_interval.sort()

    except (TypeError, ValueError) as e:
        raise TypeError(f"Input intervals must be real sequences of length 2.\nOrigin: {e}") from e

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
        (A_scaled[0] - B_scaled[1]) / math.sqrt(8),
        (A_scaled[1] - B_scaled[0]) / math.sqrt(8),
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
            A_scaled[0] - bi * math.sqrt(2),
            A_scaled[1] - bi * math.sqrt(2),
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
                fl_alpha_i >= A_interval[0]
                and fl_alpha_i <= A_interval[1]
                and fl_alpha_i_conjugate >= B_interval[0]
                and fl_alpha_i_conjugate <= B_interval[1]
            ):
                alpha.append(alpha_i)
    return alpha


def plot_grid_problem(
    A: Sequence[float | int],
    B: Sequence[float | int],
    solutions: Sequence[Zsqrt2],
    show: Optional[bool] = False,
) -> None:
    """
    Plot the solutions of the 1D grid problem on the real axis.

    Given the two real intervals A and B and the solution to their 1D grid problems,
    plot the solutions and their conjugate on the real axis.

    Args:
        A (Sequence[float | int]): Bounds of the first interval.
        B (Sequence[float | int]): Bounds of the second interval.
        solutions (Sequence[Zsqrt2]): Solutions of the 1D grid problem for A and B in \u2124[\u221A2].
        show (bool, optional): If set to True, show the plot in a window. Default = False.

    Raises:
        TypeError: If A and B intervals are not sequences of length 2.
        TypeError: If intervals limits are not real numbers.
        ValueError: If intervals limits are not in ascending order.
        TypeError: If solutions are not a sequence of Zsqrt2 objects.
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
    if not isinstance(solutions, (Sequence, np.ndarray)):
        raise TypeError(f"Expected solutions to be subscriptable, but got {solutions}.")
    if not all([isinstance(solution, Zsqrt2) for solution in solutions]):
        raise TypeError("Solutions must be Zsqrt2 objects.")

    plt.figure(figsize=(8, 3))
    plt.axhline(color="k", linestyle="--", linewidth=0.7)
    plt.axvline(color="k", linestyle="--", linewidth=0.7)
    plt.grid(axis="x", which="both")
    plt.scatter(
        [float(i) for i in solutions],
        [0] * len(solutions),
        color="blue",
        s=25,
        label=r"$\alpha$",
    )
    plt.scatter(
        [float(i.sqrt2_conjugate()) for i in solutions],
        [0] * len(solutions),
        color="red",
        s=20,
        marker="x",
        label=r"$\alpha^\bullet$",
    )
    plt.ylim((-1, 1))
    plt.axvspan(A[0], A[1], color="blue", alpha=0.2, label="A")
    plt.axvspan(B[0], B[1], color="red", alpha=0.2, label="B")
    plt.title(f"Solutions for the 1 dimensional grid problem for A = {list(A)} and B = {list(B)}")
    plt.yticks([])
    plt.legend()
    if show:
        plt.show()
    else:
        Path("Solutions").mkdir(exist_ok=True)
        plt.savefig(
            os.path.join("Solutions", f"solutions_1D_A{A[0]}_{A[1]}_B{B[0]}_{B[1]}.png"), dpi=200
        )
