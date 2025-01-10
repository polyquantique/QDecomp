# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Using the `zyz_decomposition.py` module

# Any single qubit gate can be decomposed into a series of three rotations around the Z, Y, and Z axis and a phase (see Section 4.1 of [Quantum Gates](https://threeplusone.com/pubs/on_gates.pdf) by Gavin E. Crooks).
# This module defines the function that computes this decomposition of an unitary 2x2 matrix.
# It returns the three rotation angles and the phase.
#
# **Input:** U, a unitary 2x2 matrix \
# **Output:** A tuple containing the three angles and the phase of the decomposition ($\theta_0$, $\theta_1$, $\theta_2$, $\alpha$) [rad] such that $U = e^{i \alpha} R_z(\theta_2) R_y(\theta_1) R_z(\theta_0)$

import numpy as np
from zyz_decomposition import *

Rx = lambda teta: np.array([[np.cos(teta / 2), -1.0j * np.sin(teta / 2)], [-1.0j * np.sin(teta / 2), np.cos(teta / 2)]])
Ry = lambda teta: np.array([[np.cos(teta / 2), -np.sin(teta / 2)], [np.sin(teta / 2), np.cos(teta / 2)]])
Rz = lambda teta: np.array([[np.exp(-1.0j * teta / 2), 0], [0, np.exp(1.0j * teta / 2)]])
phase = lambda alpha: np.exp(1.0j * alpha)

# +
a = complex(1, 1) / np.sqrt(3)
b = np.sqrt(complex(1, 0) - np.abs(a) ** 2)  # Ensure that U is unitary
alpha = np.pi/3
U = np.exp(1.0j * alpha) * np.array([[a, -b.conjugate()], [b, a.conjugate()]])  # Unitary matrix

t0, t1, t2, alpha_ = zyz_decomposition(U)  # Compute the decomposition of U

U_calculated = phase(alpha_) * Rz(t2) @ Ry(t1) @ Rz(t0)  # Recreate U from the decomposition

print(f"U =\n{U}\n")
print(f"U_calculated =\n{U_calculated}\n")
print(f"Error = {np.linalg.norm(U - U_calculated)}")
