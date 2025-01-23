# # Using the `diophantine_equation.py` module

# This module solves the Diophantine equation $\xi = t^\dagger t$ for $t \in \mathbb{D[\omega]}$ where $\xi \in \mathbb{D[\sqrt{2}]}$ is given.
# The solution $t$ is returned if it existes, or `None` otherwise.
# This module is an implementation of the algorithm presented in Section 6 and Appendix C of [Optimal ancilla-free Clifford+T approximation of z-rotations](https://arxiv.org/abs/1403.2975) by Ross and Selinger.
#
# **Input:** $\xi \in \mathbb{D[\sqrt{2}]}$ \
# **Output:** $t \in \mathbb{D[\omega]}$, the solution to the equation $\xi = t^\dagger t$, or `None` if no solution exists for the specified $\xi$
#
# ---

# The three following cells presents an example on how to use this module.

# +
# Import libraries
import sys

sys.path.append("..")  # Required to load the Rings module from parent directory

from grid_algorithm_1D.Rings import *
from diophantine_equation import solve_xi_eq_ttdag_in_d

# +
# Solve the Diophantine equation
xi = Dsqrt2(D(13, 1), D(4, 1))  # Input
t = solve_xi_eq_ttdag_in_d(xi)  # Compute the solution

print(f"{xi = }")
print(f"{t = }")

# +
# Check the solution
xi_calculated_in_Domega = t * t.complex_conjugate()      # Calculate (t * t†)
xi_calculated = xi_calculated_in_Domega.convert(Dsqrt2)  # Convert the result from D[omega] to D[sqrt(2)]

print(f"{xi_calculated = }")
print(f"{xi == xi_calculated = }")
# -

# ---
#
# This is an example where the Diophantine equation doesn't have any solution

# +
# Solve the Diophantine equation
xi = Dsqrt2(D(9, 1), D(3, 1))   # Input
t = solve_xi_eq_ttdag_in_d(xi)  # Compute the solution

print(f"{xi = }")
print(f"{t = }")
