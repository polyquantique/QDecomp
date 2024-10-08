import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from Zsqrt2 import Zsqrt2, inv_lamb, lamb


def solve_grid_problem_1d(
    A: Iterable[float], B: Iterable[float], plot_solutions: bool = False
) -> list[Zsqrt2]:
    """Solves the 1 dimensional grid problem for two sets and returns the result.

    Given two real closed sets A and B, this function finds all the solutions x in the ring Z[sqrt(2)] such that
    x is in A and conjugate(x) i in B.

    Args:
        A (Iterable[float]): Bounds of the first interval.
        B (Iterable[float]): Bounds of the second interval.
        plot_solutions (bool): Plot all solutions to the problem on the real axis is set to True.

    Returns:
        list[Zsqrt2]: The list of solutions to the grid problem.

    Raises:
        TypeError: If function argument are not iterable.
        TypeError: If intervals limits are not real numbers.
    """
    if not isinstance(A, Iterable) or not isinstance(B, Iterable):
        raise TypeError(f"Expected input intervals to be iterable")
    elif not all([(isinstance(i, float) or isinstance(i, int)) for i in list(A) + list(B)]):
        raise TypeError("Interval limits must be real numbers")

    A_interval: np.ndarray = np.array(A)
    B_interval: np.ndarray = np.array(B)
    deltaA: float = A_interval[1] - A_interval[0]

    is_smaller_case: bool = False
    if deltaA >= 1:
        # Tester cas où deltaA == 1
        n_scaling: int = math.ceil(-math.log(deltaA) / math.log(inv_lamb))
        A_scaled: np.ndarray = A_interval * float(inv_lamb**n_scaling)
        B_scaled: np.ndarray = B_interval * float((-lamb) ** n_scaling)
    elif deltaA < float(inv_lamb):
        is_smaller_case = True
        n_scaling = math.ceil(math.log(float(inv_lamb) / deltaA) / math.log(lamb))
        A_scaled = A_interval * float(lamb**n_scaling)
        B_scaled = B_interval * float(lamb.conjugate() ** n_scaling)
    else:
        A_scaled = A_interval
        B_scaled = B_interval

    assert (A_scaled[1] - A_scaled[0]) < 1 and (A_scaled[1] - A_scaled[0]) >= float(inv_lamb)

    if n_scaling % 2 == 1:
        B_scaled = np.flip(B_scaled)

    b_interval_scaled: np.ndarray = np.array(
        [
            (A_scaled[0] - B_scaled[1]) / math.sqrt(8),
            (A_scaled[1] - B_scaled[0]) / math.sqrt(8),
        ]
    )
    b_start: int = math.ceil(b_interval_scaled[0])
    b_end: int = math.floor(b_interval_scaled[-1])

    assert b_start <= b_end

    alpha: list[Zsqrt2] = []
    for bi in list(range(b_start, b_end + 1)):
        a_interval_scaled: list[float] = [
            A_scaled[0] - bi * math.sqrt(2),
            A_scaled[1] - bi * math.sqrt(2),
        ]
        if math.ceil(a_interval_scaled[0]) == math.floor(a_interval_scaled[1]):
            alpha_scaled = math.ceil(a_interval_scaled[0]) + bi * math.sqrt(2)
            alpha_conjugate_scaled = math.ceil(a_interval_scaled[0]) - bi * math.sqrt(2)
            if (
                alpha_scaled >= A_scaled[0]
                and alpha_scaled <= A_scaled[1]
                and alpha_conjugate_scaled >= B_scaled[0]
                and alpha_conjugate_scaled <= B_scaled[1]
            ):
                alpha.append(
                    Zsqrt2(math.ceil(a_interval_scaled[0]), bi)
                    * (lambda is_smaller: inv_lamb if is_smaller else lamb)(is_smaller_case)
                    ** n_scaling
                )

    if len(alpha) == 0:
        print("No solutions were found for the grid problem.")
    else:
        print(f"{len(alpha)} solutions were found.")

    for alpha_i in alpha:
        assert float(alpha_i) <= A_interval[1] and float(alpha_i) >= A_interval[0]
        assert (
            float(alpha_i.conjugate()) <= B_interval[1]
            and float(alpha_i.conjugate()) >= B_interval[0]
        )
    if plot_solutions:
        plt.figure(figsize=(8, 3))
        plt.axhline(color="k", linestyle="--", linewidth=0.7)
        plt.axvline(color="k", linestyle="--", linewidth=0.7)
        plt.grid(axis="x", which="both")
        plt.scatter(
            [float(i) for i in alpha],
            [0] * len(alpha),
            color="blue",
            s=25,
            label=r"$\alpha$",
        )
        plt.scatter(
            [float(i.conjugate()) for i in alpha],
            [0] * len(alpha),
            color="red",
            s=20,
            marker="x",
            label=r"$\alpha^\bullet$",
        )
        plt.ylim((-1, 1))
        plt.axvspan(A_interval[0], A_interval[1], color="blue", alpha=0.2, label="A")
        plt.axvspan(B_interval[0], B_interval[1], color="red", alpha=0.2, label="B")
        plt.title(
            f"Solutions for the 1 dimensional grid problem for A = {list(A)} and B = {list(B)}"
        )
        plt.yticks([])
        plt.legend()
        Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Solutions")).mkdir(
            parents=True, exist_ok=True
        )
        plt.savefig(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "Solutions", "solutions.png"),
            dpi=200,
        )

    return alpha


if __name__ == "__main__":
    solve_grid_problem_1d((-8, 0), (-3, 3), plot_solutions=True)
