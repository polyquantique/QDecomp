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
This module defines the function that computes the ZYZ decomposition of an unitary 2x2 matrix.

Input: U, a unitary 2x2 matrix
Output: Angles of the decomposition (t0, t1, t2, alpha) [rad] such that
        U = e**(i alpha) R_z(t2) R_y(t1) R_z(t0)
"""

import cmath
import math

import numpy as np


def zyz_decomposition(U: np.ndarray) -> tuple[float, ...]:
    """
    Compute the ZYZ decomposition of a 2x2 unitary matrix U.
    U = e**(i alpha) * Rz(t2) * Ry(t1) * Rz(t0)

    Args:
        U (np.ndarray): A 2x2 unitary matrix.

    Returns:
        tuple (float, ...): (t0, t1, t2, alpha), the ZYZ rotation angles (rad) and the global phase (rad)
    """
    if not isinstance(U, np.ndarray):
        raise TypeError(f"Input must be a numpy array. Got {type(U)}.")

    if not U.shape == (2, 2):
        raise ValueError(f"The input matrix must be 2x2. Got a matrix with shape {U.shape}.")

    det = np.linalg.det(U)
    if not math.isclose(abs(det), 1):
        raise ValueError(f"The input matrix must be unitary. Got a matrix with determinant {det}.")

    alpha = math.atan2(det.imag, det.real) / 2  # det = exp(2 i alpha)
    V = cmath.exp(-1.0j * alpha) * U  # V = exp(-i alpha)*U is a special unitary matrix

    # Avoid divisions by zero if U is diagonal
    if math.isclose(abs(V[0, 0]), 1, rel_tol=1e-14, abs_tol=1e-14):
        t0 = 0
        t1 = 0
        t2 = -2 * cmath.phase(V[0, 0])
        return t0, t1, t2, alpha

    # Compute the first rotation angle
    if abs(V[0, 0]) >= abs(V[0, 1]):
        t1 = 2 * math.acos(abs(V[0, 0]))
    else:
        t1 = 2 * math.asin(abs(V[0, 1]))

    # Useful variables for the next steps
    V11_ = V[1, 1] / math.cos(t1 / 2)
    V10_ = V[1, 0] / math.sin(t1 / 2)

    a = 2 * math.atan2(V11_.imag, V11_.real)
    b = 2 * math.atan2(V10_.imag, V10_.real)

    # The following system of equations is solved to find t0 and t2
    # t0 + t2 = a
    # t0 - t2 = b
    t0 = (a + b) / 2
    t2 = (a - b) / 2

    return t0, t1, t2, alpha
