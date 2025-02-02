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
Any single qubit gate can be decomposed into a series of three rotations around the Z, Y, and Z axis
and a phase (refer Section 4.1 of "Quantum Gates" by Gavin E. Crooks at
https://threeplusone.com/pubs/on_gates.pdf).

This module defines the function that computes this decomposition of a unitary 2x2 matrix.
It returns the three rotation angles and the phase.

Input: U, a unitary 2x2 matrix
Output: Angles of the decomposition (t0, t1, t2, alpha) [rad] such that
        U = e**(i alpha) R_z(t2) R_y(t1) R_z(t0)
"""

import numpy as np


def zyz_decomposition(U: np.ndarray | list[list[float]]) -> tuple[float, ...]:
    """
    Compute the ZYZ decomposition of a 2x2 unitary matrix U.
    U = e**(i alpha) * Rz(t2) * Ry(t1) * Rz(t0)

    Args:
        U (np.ndarray): A 2x2 unitary matrix.

    Returns:
        tuple (float, ...): (t0, t1, t2, alpha), the ZYZ rotation angles (rad) and the global phase (rad)

    Raises:
        ValueError: If the input matrix is not 2x2.
        ValueError: If the input matrix is not unitary.

    Example:
        # Define rotation and phase matrices
        ry = lambda teta: np.array([[np.cos(teta / 2), -np.sin(teta / 2)], [np.sin(teta / 2), np.cos(teta / 2)]])
        rz = lambda teta: np.array([[np.exp(-1.0j * teta / 2), 0], [0, np.exp(1.0j * teta / 2)]])
        phase = lambda alpha: np.exp(1.0j * alpha)

        # Create a unitary matrix U
        a = complex(1, 1) / np.sqrt(3)
        b = np.sqrt(complex(1, 0) - np.abs(a) ** 2)  # Ensure that U is unitary
        alpha = np.pi/3
        U = np.exp(1.0j * alpha) * np.array([[a, -b.conjugate()], [b, a.conjugate()]])

        # Compute the decomposition of U
        t0, t1, t2, alpha_ = zyz_decomposition(U)

        # Recreate U from the decomposition
        U_calculated = phase(alpha_) * Rz(t2) @ Ry(t1) @ Rz(t0)

        # Print the results
        print(f"U =\n{U}\n")
        print(f"U_calculated =\n{U_calculated}\n")
        print(f"Error = {np.linalg.norm(U - U_calculated)}")
    """
    # Convert U to a np.ndarray if it is not already
    U = np.asarray(U)

    # Check the input matrix
    if not U.shape == (2, 2):
        raise ValueError(f"The input matrix must be 2x2. Got a matrix with shape {U.shape}.")

    det = np.linalg.det(U)
    if not np.isclose(abs(det), 1):
        raise ValueError(f"The input matrix must be unitary. Got a matrix with determinant {det}.")

    # Compute the global phase and the special unitary matrix V
    alpha = np.atan2(det.imag, det.real) / 2  # det = exp(2 i alpha)
    V = np.exp(-1.0j * alpha) * U  # V = exp(-i alpha)*U is a special unitary matrix

    # Avoid divisions by zero if U is diagonal
    if np.isclose(abs(V[0, 0]), 1, rtol=1e-14, atol=1e-14):
        t2 = -2 * np.angle(V[0, 0])
        return 0, 0, t2, alpha

    # Compute the first rotation angle
    if abs(V[0, 0]) >= abs(V[0, 1]):
        t1 = 2 * np.acos(abs(V[0, 0]))
    else:
        t1 = 2 * np.asin(abs(V[0, 1]))

    # Useful variables for the next steps
    V11_ = V[1, 1] / np.cos(t1 / 2)
    V10_ = V[1, 0] / np.sin(t1 / 2)

    a = 2 * np.atan2(V11_.imag, V11_.real)
    b = 2 * np.atan2(V10_.imag, V10_.real)

    # The following system of equations is solved to find t0 and t2
    # t0 + t2 = a
    # t0 - t2 = b
    t0 = (a + b) / 2
    t2 = (a - b) / 2

    return t0, t1, t2, alpha
