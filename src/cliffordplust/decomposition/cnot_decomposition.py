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

"""This module contains functions to decompose general 2-qubits quantum gates into single-qubit and canonical gates.

The canonical gate is a 3 parameters gate that can be decomposed into CNOT gates and single-qubit gates. It is defined as 
Can(tx, ty, tz) = exp(-i*pi/2 * (tx * XX + ty * YY + tz * ZZ)), where XX, YY, and ZZ are Kronecker products of the Pauli matrices.

The module contains the following functions:
- kronecker_decomposition: Decompose a 4x4 matrix into two 2x2 matrices such that their Kronecker product is the closet to the original matrix.
- canonical_decomposition: Decompose a 4x4 unitary matrix into a global phase, two local 4x4 matrices, and the three parameters of the canonical gate.
- can: Return the matrix form of the canonical gate for the given parameters.

For more details on the theory, see 
G. E. Crooks, “Quantum gates,” March 2024, version 0.11.0, https://threeplusone.com/pubs/on_gates.pdf
and
Jun Zhang, Jiri Vala, Shankar Sastry, and K. Birgitta Whaley. Geometric theory of nonlocal two-qubit operations. Phys. Rev. A, 67:042313 (2003), https://arxiv.org/pdf/quant-ph/0209120

The module also contains tests for the functions. The tests are written using pytest and can be run with the command `pytest` in the terminal.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm


def kronecker_decomposition(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a 4x4 matrix M, find the two 2x2 matrix A and B such that their Kronecker
    product is the closest to the matrix M in the Frobenius norm.

    Args:
        M (np.ndarray): 4x4 matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The two 2x2 matrix of the decomposition.

    Raises:
        TypeError: If M is not a numpy matrix.
        ValueError: If M is not a 4x4 matrix.
    """
    if not isinstance(M, np.ndarray):
        raise TypeError(f"Matrix must be a numpy object, but got {type(M).__name__}.")
    elif M.shape != (4, 4):
        raise ValueError(f"Matrix must be 4x4, but received {M.shape}.")

    M = M.reshape(2, 2, 2, 2)
    M = M.transpose(0, 2, 1, 3)
    M = M.reshape(4, 4)

    u, sv, vh = np.linalg.svd(M)

    A = np.sqrt(sv[0]) * u[:, 0].reshape(2, 2)
    B = np.sqrt(sv[0]) * vh[0, :].reshape(2, 2)
    return A, B


def canonical_decomposition(
    U: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], float]:
    """
    Perform the canonical decomposition of a given 4x4 unitary.

    Given a 4x4 unitary matrix, find the phase alpha, the two 4x4 local unitaries A and B, and
    the three parameters of the canonical gate to decompose the input matrix U like

    U = exp(i*alpha) * B @ Can(tx, ty, tz) @ A.

    Args:
        U (np.ndarray): 4x4 unitary matrix.

    Returns:
        np.ndarray: 4x4 matrix A of the decomposition. A is the Kronecker product of two 2x2 matrices.
        np.ndarray: 4x4 matrix B of the decomposition. B is the Kronecker product of two 2x2 matrices.
        tuple[float, float, float]: Sequence of the three canonical parameters (tx, ty, tz).
        float: Phase of the unitary matrix.

    Raises:
        TypeError: If the input is not a numpy object.
        ValueError: If the matrix is not 4x4 and unitary.
    """
    if not isinstance(U, np.ndarray):
        raise TypeError(f"Matrix U must be a numpy object, but received {type(U).__name__}.")
    elif U.shape != (4, 4):
        raise ValueError("U must be 4x4.")
    elif not np.allclose(U @ U.T.conj(), np.identity(4)):
        raise ValueError("U must be unitary.")

    # Magic gate
    M = (
        1
        / math.sqrt(2)
        * np.array([[1, 1.0j, 0, 0], [0, 0, 1.0j, 1], [0, 0, 1.0j, -1], [1, -1.0j, 0, 0]])
    )
    det_U = np.complex128(np.linalg.det(U))
    phase = np.angle(det_U) / 4
    U = U * det_U ** (-1 / 4)

    # Transform U into the magic basis to get V and diagonalize V.T@V.
    V = M.T.conj() @ U @ M
    VV = V.T @ V

    # For numerical precision purpose, we use the eigh function when dealing with hermitian or symmetric matrices.
    if np.allclose(VV.T.conj(), VV):
        eigenval, eigenvec = np.linalg.eigh((VV + VV.T.conj()) / 2)
    elif np.allclose(1.0j * VV, -1.0j * VV.T.conj()):
        VV = 1.0j * VV
        eigenval, eigenvec = np.linalg.eigh((VV + VV.T.conj()) / 2)
        eigenval = -1.0j * eigenval
    else:
        eigenval, eigenvec = np.linalg.eig(VV)

    # Q1 must be a special unitary matrix. If its determinant is -1, swap two eigenvalues
    # and the two associated eigenvectors to get invert the sign of the determinant.
    if np.linalg.det(eigenvec) < 0:
        eigenvec[:, [0, 1]] = eigenvec[:, [1, 0]]
        eigenval[[0, 1]] = eigenval[[1, 0]]

    # Compute Q1 and D from the eigenvectors and the eigenvalues of the decomposition.
    Q1 = eigenvec.T
    D = np.sqrt(np.complex128(eigenval))

    # Q2 must be a special unitary matrix. Since Q2 = V@Q1.T@D^-1, and det(V) = det(Q1) = 1, det(D) must be 1.
    # D is obtained from a sqrt(D^2) and all its values are defined up to a sign. We can thus ensure det(D) = 1 by changing the
    # sign to one of its value without influencing Q1.
    if np.prod(D) < 0:
        D[0] = -D[0]
    Q2 = V @ Q1.T @ np.diag(1 / D)

    # Compute the canonical parameters.
    diag_angles = -np.angle(D) / np.pi
    tx = diag_angles[0] + diag_angles[2]
    ty = diag_angles[1] + diag_angles[2]
    tz = diag_angles[0] + diag_angles[1]

    return M @ Q1 @ M.T.conj(), M @ Q2 @ M.T.conj(), (tx, ty, tz), phase


def can(tx: float, ty: float, tz: float) -> np.ndarray:
    """Return the matrix form of the canonical gate for the given parameters.

    Args:
        tx, ty, tz (floats): Parameters of the canonical gates

    Returns:
        np.ndarray: Matrix form of the canonical gate.
    """
    XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    YY = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
    ZZ = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    exponent = -1.0j * math.pi / 2 * (tx * XX + ty * YY + tz * ZZ)
    return expm(exponent)
