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
This module contains functions to plot the solutions of 1D and 2D grid problems.

The module provides two functions:

- :func:`plot_grid_problem_1d`: Plots the solutions of the 1D grid problem on the real axis.

- :func:`plot_grid_problem_2d`: Plots the solutions of the 2D grid problem for upright rectangles on the complex plane.
"""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from qdecomp.rings import Zomega, Zsqrt2

__all__ = ["plot_grid_problem_1d", "plot_grid_problem_2d"]


def plot_grid_problem_1d(
    ax: plt.Axes, A: NDArray[np.floating], B: NDArray[np.floating], solutions: Sequence[Zsqrt2]
) -> None:
    """Plot the solutions of the 1D grid problem on the real axis.

    Given the two real intervals `A` and `B` and the list of solutions to their 1D grid problem,
    plot the solutions and their :math:`\\sqrt{2}`-conjugate on the real axis.

    Args:
        ax (plt.Axes): Matplotlib axis on which to plot the solutions.
        A (Sequence[Real, Real]): (A0, A1): Bounds of the first interval.
        B (Sequence[Real, Real]): (B0, B1): Bounds of the second interval.
        solutions (Sequence[Zsqrt2]): List of solutions to the 1D grid problem for A and B as Zsqrt objects.

    Raises:
        TypeError: If intervals A and B are not real sequences of length 2.
        TypeError: If solutions is not a sequence of Zsqrt2 objects.
    """
    try:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if A.shape != (2,) or B.shape != (2,):
            raise TypeError(
                f"Input intervals must have two bounds (lower, upper) but received {A if A.shape != (2,) else B}."
            )
        A.sort()
        B.sort()

    except (TypeError, ValueError) as e:
        raise TypeError(f"Input intervals must be real sequences of length 2.\nOrigin: {e}") from e

    if not all([isinstance(solution, Zsqrt2) for solution in solutions]):
        raise TypeError("Solutions must be Zsqrt2 objects.")

    ax.axhline(color="k", linestyle="--", linewidth=0.7)
    ax.axvline(color="k", linestyle="--", linewidth=0.7)
    ax.grid(axis="x", which="both")
    ax.scatter(
        [float(i) for i in solutions],
        [0] * len(solutions),
        color="blue",
        s=25,
        label=r"$\alpha$",
    )

    ax.scatter(
        [float(i.sqrt2_conjugate()) for i in solutions],
        [0] * len(solutions),
        color="red",
        s=20,
        marker="x",
        label=r"$\alpha^\bullet$",
    )

    ax.set_ylim((-1, 1))
    ax.axvspan(A[0], A[1], 0.25, 0.75, color="blue", alpha=0.25, label="A")
    ax.axvspan(B[0], B[1], 0.25, 0.75, color="red", alpha=0.25, label="B")
    ax.set_title(f"Solutions to the 1D grid problem for A = {A} and B = {B}")
    ax.set_yticks([])
    ax.legend()


def plot_grid_problem_2d(
    ax: plt.Axes, A: NDArray[np.floating], B: NDArray[np.floating], solutions: Sequence[Zomega]
) -> None:
    """
    Plot the solutions of the 2D grid problem for upright rectangles on the complex plane.

    Given the two upright rectangles `A` and `B` and the solutions to their 2D grid problem, plot the solutions and their
    :math:`\\sqrt{2}`-conjugate on the complex plane.

    Args:
        ax (plt.Axes): Matplotlib axis on which to plot the solutions.
        A (Sequence[Sequence[Real, Real]]): ((Ax0, Ax1), (Ay0, Ay1)): Bounds of the first upright rectangle. Rows correspond to the x and y axis respectively.
        B (Sequence[Sequence[Real, Real]]): ((Bx0, Bx1), (By0, By1)): Bounds of the second upright rectangle. Rows correspond to the x and y axis respectively.
        solutions (Sequence[Zomega]): List of solutions to the 2D grid problem for A and B as Zomega objects.

    Raises:
            TypeError: If intervals A and B are not real 2 x 2 nested sequences.
            TypeError: If solutions is not a sequence of Zomega objects.
    """

    try:
        # Define the intervals for A and B rectangles.
        Ax = np.asarray(A[0], dtype=float)
        Ay = np.asarray(A[1], dtype=float)
        Bx = np.asarray(B[0], dtype=float)
        By = np.asarray(B[1], dtype=float)
        for interval in (Ax, Ay, Bx, By):
            if interval.shape != (2,):
                raise TypeError(
                    f"Input intervals must have two bounds (lower, upper) but received {interval}."
                )
            interval.sort()
    except (TypeError, ValueError) as e:
        raise TypeError(f"Input intervals must be real 2 x 2 matrices.\nOrigin: {e}") from e

    if not all([isinstance(solution, Zomega) for solution in solutions]):
        raise TypeError("Solutions must be Zomega objects.")

    alpha_x = [solution.real() for solution in solutions]
    alpha_y = [solution.imag() for solution in solutions]
    alpha_conjugate_x = [solution.sqrt2_conjugate().real() for solution in solutions]
    alpha_conjugate_y = [solution.sqrt2_conjugate().imag() for solution in solutions]

    ax.scatter(alpha_x, alpha_y, color="blue", s=25, label="$\\alpha$")
    ax.scatter(
        alpha_conjugate_x,
        alpha_conjugate_y,
        color="red",
        s=10,
        marker="x",
        label="$\\alpha^\\bullet$",
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.7)

    A_rect_x = [A[0][0], A[0][1], A[0][1], A[0][0]]
    A_rect_y = [A[1][0], A[1][0], A[1][1], A[1][1]]

    B_rect_x = [B[0][0], B[0][1], B[0][1], B[0][0]]
    B_rect_y = [B[1][0], B[1][0], B[1][1], B[1][1]]

    ax.fill(A_rect_x, A_rect_y, color="blue", alpha=0.25, label="A")
    ax.fill(B_rect_x, B_rect_y, color="red", alpha=0.25, label="B")

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(f"Solutions to the 2D grid problem for A = {Ax} x {Ay} and B = {Bx} x {By}")
    ax.legend()
