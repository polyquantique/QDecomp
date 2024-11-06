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
This file contains the functions solve_grid_problem_1d and plot_grid_problem.

1) solve_grid_problem_1d
Given two real closed intervals A and B, solve_grid_problem_1d finds all the solutions alpha in 
the ring of quadratic integers with radicand 2, Z[√2], such that alpha is in A and the √2-conjugate 
of alpha is in B. This function is a sub-algorithm to solve 2D grid problems for upright
rectangles. For more information, see
Neil J. Ross and Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations, https://arxiv.org/pdf/1403.2975

2) plot_grid_problem
The function plot_grid_problem plots the solutions of the grid problem for the intervals A and B and their 
conjugate on the real axis. The plot also contains the intervals A and B.
"""

import math
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from Zsqrt2 import Zsqrt2, inv_lamb, lamb


def solve_grid_problem_1d(A: Sequence[float | int], B: Sequence[float | int]) -> list[Zsqrt2]:
    """Solves the 1 dimensional grid problem for two sets and returns the result.

    Given two real closed sets A and B, this function finds all the solutions x in the ring Z[√2] such that
    x is in A and the √2-conjugate of x is in B.

    Args:
        A (Sequence[float | int]): Bounds of the first interval.
        B (Sequence[float | int]): Bounds of the second interval.

    Returns:
        list[Zsqrt2]: The list of solutions to the grid problem.

    Raises:
        TypeError: If A and B intervals are not subscriptable sequences of length 2.

        TypeError: If intervals limits are not real numbers.
        ValueError: If intervals limits are not in ascending order.
    """
    if not hasattr(A, "__getitem__") or not hasattr(B, "__getitem__"):
        raise TypeError(
            f"Expected input intervals to be subscriptable, but got {A if not hasattr(A, "__getitem__") else B}."
        )
    elif len(list(A)) != 2 or len(list(B)) != 2:
        raise TypeError(
            f"Intervals must be of length 2 but got length {len(list(A)) if len(list(A)) != 2 else len(list(B))}"
        )
    elif not all([isinstance(i, (float, int, np.int32, np.int64)) for i in list(A) + list(B)]):
        raise TypeError("Interval limits must be real numbers.")
    elif A[0] >= A[1] or B[0] >= B[1]:
        raise ValueError("Intervals A and B must have A[0] < A[1] and B[0] < B[1].")

    # Definitions of the intervals and deltaA
    A_interval: np.ndarray = np.array(A)
    B_interval: np.ndarray = np.array(B)
    deltaA: float = A_interval[1] - A_interval[0]

    # Scaling of the intervals to have lamb^-1 <= deltaA < 1
    is_smaller_case: bool = False
    if deltaA > 1:
        n_scaling: int = math.ceil(-math.log(deltaA) / math.log(float(inv_lamb)))
        A_scaled: np.ndarray = A_interval * float(inv_lamb) ** n_scaling
        B_scaled: np.ndarray = B_interval * float(-lamb) ** n_scaling
    elif deltaA == 1:
        n_scaling = 1
        A_scaled = A_interval * float(inv_lamb)
        B_scaled = B_interval * float(-lamb)
    elif deltaA < float(inv_lamb):
        is_smaller_case = True
        n_scaling = math.ceil(math.log(float(inv_lamb) / deltaA) / math.log(float(lamb)))
        A_scaled = A_interval * float(lamb) ** n_scaling
        B_scaled = B_interval * float(lamb.conjugate()) ** n_scaling
    else:
        n_scaling = 0
        A_scaled = A_interval
        B_scaled = B_interval

    # Flip the interval if multiplied by a negative number
    if n_scaling % 2 == 1:
        B_scaled = np.flip(B_scaled)

    # Interval to find b (√2 coefficient of the ring element)
    b_interval_scaled: list[float] = [
        (A_scaled[0] - B_scaled[1]) / math.sqrt(8),
        (A_scaled[1] - B_scaled[0]) / math.sqrt(8),
    ]

    b_start: int = math.ceil(b_interval_scaled[0])
    b_end: int = math.floor(b_interval_scaled[-1])

    alpha: list[Zsqrt2] = []
    for bi in range(b_start, b_end + 1):
        # Interval to look for a (Integer coefficient of the ring element)
        a_interval_scaled: list[float] = [
            A_scaled[0] - bi * math.sqrt(2),
            A_scaled[1] - bi * math.sqrt(2),
        ]
        # If there is an integer if this interval
        if math.ceil(a_interval_scaled[0]) == math.floor(a_interval_scaled[1]):
            ai: int = math.ceil(a_interval_scaled[0])
            alpha_scaled = ai + bi * math.sqrt(2)
            alpha_conjugate_scaled = ai - bi * math.sqrt(2)
            # If the solutions for ai and bi is a solution for the scaled grid problem for A and B
            if (
                alpha_scaled >= A_scaled[0]
                and alpha_scaled <= A_scaled[1]
                and alpha_conjugate_scaled >= B_scaled[0]
                and alpha_conjugate_scaled <= B_scaled[1]
            ):
                # Append the unscaled solution to the list of solutions
                alpha.append(
                    Zsqrt2(ai, bi)
                    * (lambda is_smaller: inv_lamb if is_smaller else lamb)(is_smaller_case)
                    ** n_scaling
                )

    return alpha


def plot_grid_problem(A: Sequence[float | int], B: Sequence[float | int], solutions: Sequence[Zsqrt2]) -> None:
    """Plot the solutions of the 1D grid problem on the real axis.

    Given the two real intervals A and B and the solution to their 1D grid problems, 
    plot the solutions and their conjugate on the real axis.

    Args:
        A (Sequence[float | int]): Bounds of the first interval.
        B (Sequence[float | int]): Bounds of the second interval.
        solutions (Sequence[Zsqrt2]): Solutions of the 1D grid problem for A and B in Z[√2].

    Raises:
        TypeError: If A and B intervals are not subscriptable sequences of length 2.
        TypeError: If intervals limits are not real numbers.
        ValueError: If intervals limits are not in ascending order.
        TypeError: If solutions are not subscriptable of Zsqrt2 objects.
    """
    if not hasattr(A, "__getitem__") or not hasattr(B, "__getitem__"):
        raise TypeError(
            f"Expected input intervals to be subscriptable, but got {A if not hasattr(A, "__getitem__") else B}."
        )
    elif len(list(A)) != 2 or len(list(B)) != 2:
        raise TypeError(
            f"Intervals must be of length 2 but got length {len(list(A)) if len(list(A)) != 2 else len(list(B))}"
        )
    elif not all([isinstance(i, (float, int, np.int32, np.int64)) for i in list(A) + list(B)]):
        raise TypeError("Interval limits must be real numbers.")
    elif A[0] >= A[1] or B[0] >= B[1]:
        raise ValueError("Intervals A and B must have A[0] < A[1] and B[0] < B[1].")
    if not hasattr(solutions, "__getitem__"):
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
        [float(i.conjugate()) for i in solutions],
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
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Solutions")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, "solutions.png"), dpi=200)
