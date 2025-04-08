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

# # Using the `grid_algorithm_1D.py` module
#
# In this notebook, the `grid_algorithm_1D.py` module is presented with examples. This module allows to find all the solutions of a **1-dimensional grid problem** in the ring $\mathbf{\mathbb{Z}[\sqrt{2}]}$.

# ## $\mathbb{Z}[\sqrt{2}]$
#
# $\mathbb{Z}[\sqrt{2}]$ is the ring of quadratic integers with radicand 2. It consist of the set $\{a + b\sqrt{2}\ | a, b \in \mathbb{Z}\}$. Let $\alpha = a + b\sqrt{2}$ be an element of the ring. The $\sqrt{2}$-conjugation of $\alpha$, denoted $\alpha^{\bullet}$, is given by $a - b\sqrt{2}$. For more details on this topic, see [Ross and Selinger, 2014](https://arxiv.org/pdf/1403.2975). The class `Zsqrt2`, found in the `Zsqrt2.py` module, allows to do symbolic calculation with elements of the ring $\mathbb{Z}[\sqrt{2}]$. The following code shows examples of basic arithmetic operations supported by the class.
#
#


from qdecomp.rings import Zsqrt2

# +
alpha = Zsqrt2(a=2, b=5)
beta = Zsqrt2(a=6, b=-3)

# String representation
print(f"{alpha = }, {beta = }")

# Conjugation
print(f"Conjugation: {alpha.conjugate() = }")

# Summation
print(f"Summation: {alpha + beta = }")

# Subtraction
print(f"Subtraction: {alpha - beta = }")

# Multiplication
print(f"Multiplication: {alpha * beta = }")

# Power
print(f"Power: {alpha ** 3 = }")

# Float representation
print(f"Float: {float(alpha) = }")

# Rounding
print(f"Round: {round(alpha) = }")
# -

# ## 1-dimensional grid problem
#
# Consider two real closed intervals $A = [x_0, x_1]$ and $B = [y_0, y_1]$. The grid problem for $A$ and $B$ consists of finding all the solutions $\alpha \in \mathbb{Z}[\sqrt{2}]$ such that $\alpha \in A$ and $\alpha^{\bullet} \in B$. If $A$ and $B$ are finite intervals, the grid problem is guaranteed to have a finite number of solutions. For more information, see [Ross and Selinger, 2014](https://arxiv.org/pdf/1403.2975). The `grid_algorithm_1D.py` Python module contains functions to work with 1-dimensional grid problems:
#
# 1. `solve_grid_problem_1d()`
#
# This function takes as arguments the limits of the two intervals and returns the solutions of the associated 1D grid problem. The solutions are given as a list of Zsqrt2 objects.
#
# 2. `plot_grid_problem()`
#
# This function takes as argument the solutions of the grid problem and the two intervals limits. It plots the solutions on the real axis and their conjugate. It also shows the intervals $A$ and $B$.
#
# # Usage example

from qdecomp.grid_problem import plot_grid_problem, solve_grid_problem_1d

# Let's consider the two arbitrary intervals $A = [1, 6]$ and $B = [-11, -5]$. We can define the limits of those intervals as tuples:

A = (1, 6)
B = (-11, -5)

# The 1D grid problem for those intervals can be solved using the `solve_grid_problem_1d()` function:

solutions: list[Zsqrt2] = solve_grid_problem_1d(A=A, B=B)
print(f"{len(solutions)} solutions were found.")
print("Solutions: ", solutions)

# We can see that each solution $\alpha_i$ is in the interval $A$ and that their conjugate $\alpha_i^{\bullet}$ is in $B$. Let's look at the first solutions for instance:

solution_1: Zsqrt2 = solutions[0]
print(f"{solution_1 = } = {float(solution_1):.3f}")
print(f"{solution_1.conjugate() = } = {float(solution_1.conjugate()):.3f}")

# We can also use the `plot_grid_problem()` function to plot the solutions of the grid problem:

plot_grid_problem(A=A, B=B, solutions=solutions)

# An important fact is that the number of solutions found increase linearly with the size of A and B. If the intervals are too small, no solution will be found.

A = (4, 9)
B = (0.2, 1)
solutions: list[Zsqrt2] = solve_grid_problem_1d(A=A, B=B)
print(f"Solutions: {solutions}")
plot_grid_problem(A=A, B=B, solutions=solutions)
