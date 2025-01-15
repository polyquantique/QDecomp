from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from grid_algorithm_1D import solve_grid_problem_1d
from Rings import Zomega, Zsqrt2


def solve_grid_problem_2D(A: Sequence[Sequence[float | int]], B: Sequence[Sequence[float | int]]) -> list[Zomega]:
    """
    Solves the 2-dimensional grid problem for two upright rectangle sets and returns the result.

    Given two real 2D closed sets A and B of the form [a b] x [c, d], find all the solutions x in 
    the ring of cyclotomic integers of degree 8 \u2124[\u03C9] such that x is in A and the \u221a2-conjugate of x is in B.

    Args:
        A (Sequence[Sequence[float | int]]): Bounds of the first upright rectangle. Rows correspond to the x and y axis respectively. 
        B (Sequence[Sequence[float | int]]): Bounds of the second upright rectangle. Rows correspond to the x and y axis respectively.

    Returns: 
        list[Domega]: The list of solutions to the grid problem.

    Raises:
        TypeError: If A or B are not 2x2 sequences.
        TypeError: If interval entries are not real numbers.
        ValueError: If interval bounds are not in ascending order.
    """

    # Type check of function arguments
    for interval in (A, B):
        if not isinstance(interval, Sequence):
            raise TypeError(f"Expected input intervals to be 2x2 sequences but got {interval}.")
        if len(interval) != 2:
            raise TypeError(f"Expected input intervals to be 2x2 sequences but got {interval}.")
        for axis in interval:
            if not isinstance(axis, Sequence):
                raise TypeError(f"Expected input intervals to be 2x2 sequences but got {interval}.")
            if len(axis) != 2:
                raise TypeError(f"Expected input intervals to be 2x2 sequences but got {interval}.")

    # Define the intervals for A and B rectangles
    Ax: np.ndarray = np.array(A[0])
    Ay: np.ndarray = np.array(A[1])
    Bx: np.ndarray = np.array(B[0])
    By: np.ndarray = np.array(B[1])
    
    solutions: list[Zomega] = []

    # Solve two 1D grid problems for solutions of the form a + bi, where a and b are in Z[√2]
    alpha: list[Zsqrt2] = solve_grid_problem_1d(Ax, Bx)
    beta: list[Zsqrt2] = solve_grid_problem_1d(Ay, By)
    for a in alpha:
        for b in beta:
            solutions.append(Zomega(a = b.q - a.q, b = b.p, c = b.q + a.q, d = a.p))
    
    # Solve two 1D grid problems for solutions of the form a + bi + ω, where a and b are in Z[√2] and ω = (1 + i)/√2
    alpha2: list[Zsqrt2] = solve_grid_problem_1d(Ax - 1/math.sqrt(2), Bx + 1/math.sqrt(2))
    beta2: list[Zsqrt2] = solve_grid_problem_1d(Ay - 1/math.sqrt(2), By + 1/math.sqrt(2))
    for a in alpha2:
        for b in beta2:
            solutions.append(Zomega(b.q - a.q, b.p, b.q + a.q + 1, a.p))

    return solutions

def plot_solutions(A: Sequence[Sequence[float | int]], B: Sequence[Sequence[float | int]], solutions: list[Zomega], show: bool = False):
    
    alpha_x = [solution.real() for solution in solutions]
    alpha_y = [solution.imag() for solution in solutions]
    alpha_conjugate_x = [solution.sqrt2_conjugate().real() for solution in solutions]
    alpha_conjugate_y = [solution.sqrt2_conjugate().imag() for solution in solutions]
    
    plt.scatter(alpha_x, alpha_y, color='blue', label='$\\alpha$')
    plt.scatter(alpha_conjugate_x, alpha_conjugate_y, color="red", marker="x", label="$\\alpha^\\bullet$")

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    Ax = [A[0][0], A[0][1], A[0][1], A[0][0]]
    Ay = [A[1][0], A[1][0], A[1][1], A[1][1]]

    Bx = [B[0][0], B[0][1], B[0][1], B[0][0]]
    By = [B[1][0], B[1][0], B[1][1], B[1][1]]
    plt.fill(Ax, Ay, color='blue', alpha=0.2, label='A')
    plt.fill(Bx, By, color='red', alpha=0.2, label='B')

    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title(f'2D grid problem for A = {list(A[0])} x {list(A[1])} and B = {list(B[0])} x {list(B[1])}')

    plt.legend()
    if show:
        plt.show()
    else:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Solutions")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_path, "solutions.png"), dpi=200)
