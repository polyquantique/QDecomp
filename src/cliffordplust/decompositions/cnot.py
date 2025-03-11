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
This module contains functions to decompose general 2-qubits quantum gates into single-qubit and
canonical gates.

The canonical gate is a 3 parameters gate that can be decomposed into CNOT gates and single-qubit
gates. It is defined as Can(tx, ty, tz) = exp(-i*pi/2 * (tx * XX + ty * YY + tz * ZZ)), where XX,
YY, and ZZ are Kronecker products of the Pauli matrices.

The module contains the following functions:
- kronecker_decomposition: Decompose a 4x4 matrix into two 2x2 matrices such that their Kronecker product is the closest to the original matrix.
- canonical_decomposition: Decompose a 4x4 unitary matrix into a global phase, two local 4x4 matrices, and the three parameters of the canonical gate.
- can: Return the matrix form of the canonical gate for the given parameters.

For more details on the theory, see 
G. E. Crooks, “Quantum gates,” March 2024, version 0.11.0, https://threeplusone.com/pubs/on_gates.pdf,
C. F. Van Loan, “The ubiquitous Kronecker product”, J. Comput. Appl. Math., vol. 123, no. 1–2, pp. 85–100, Nov. 2000, https://doi.org/10.1016/S0377-0427(00)00393-9,
and
Jun Zhang, Jiri Vala, Shankar Sastry, and K. Birgitta Whaley. Geometric theory of nonlocal two-qubit operations. Phys. Rev. A, 67:042313 (2003), https://arxiv.org/pdf/quant-ph/0209120.
"""

from collections import namedtuple
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from cliffordplust.circuit import QGate, QCircuit
from cliffordplust.decompositions import gates

SQRT_2 = np.sqrt(2)

# Definition of the magic gate used in many decompositions
MAGIC = (
    1 / SQRT_2 * np.array([[1, 1.0j, 0, 0], [0, 0, 1.0j, 1], [0, 0, 1.0j, -1], [1, -1.0j, 0, 0]])
)
MAGIC_DAG = MAGIC.T.conj()

def magic_decomp(q0, q1) -> QCircuit:
    magic_circuit = QCircuit(
        [
            QGate(name='S', matrix=gates['S'], target=q0),
            QGate(name='S', matrix=gates['S'], target=q1),
            QGate(name='H', matrix=gates['H'], target=q1),
            QGate(name='CNOT', matrix=gates['CNOT1'], target=q0, control=q1),
        ]
    )
    return magic_circuit

def magic_dag_decomp(q0, q1) -> QCircuit:
    magic_dag_circuit = QCircuit(
        [
            QGate(name='CNOT', matrix=gates['CNOT1'], target=q0, control=q1),
            QGate(name='H', matrix=gates['H'], target=q1),
            QGate(name='SDAG', matrix=gates['S'].T.conj(), target=q1),
            QGate(name='SDAG', matrix=gates['S'].T.conj(), target=q0),
        ]
    )
    return magic_dag_circuit

def is_special(matrix: NDArray[np.floating]) -> bool:
    """
    Check if a matrix is special.
    
    Return True if the determinant of the matrix is 1 and False otherwise.

    Args:
        matrix (NDArray[float]): Matrix to check.

    Returns:
        bool: True if the matrix is special, False otherwise.
    """
    return np.isclose(np.linalg.det(matrix), 1)


def is_orthogonal(matrix: NDArray[np.floating]) -> bool:
    """
    Check if a matrix is orthogonal.

    Return True if the matrix inverse is equal to its transpose and False otherwise.

    Args:
        matrix (NDArray[float]): Matrix to check.

    Returns:
        bool: True if the matrix is orthogonal, False otherwise.
    """
    return np.allclose(matrix @ matrix.T, np.identity(matrix.shape[0]))


def is_unitary(matrix: NDArray[np.floating]) -> bool:
    """
    Check if a matrix is unitary.

    Return True if the matrix inverse is equal to its conjugate transpose and False otherwise.

    Args:
        matrix (NDArray[float]): Matrix to check.

    Returns:
        bool: True if the matrix is unitary, False otherwise.
    """
    return np.allclose(matrix @ matrix.T.conj(), np.identity(matrix.shape[0]))


def is_hermitian(matrix: NDArray[np.floating]) -> bool:
    """
    Check if a matrix is Hermitian.

    Return True if the matrix is equal to its conjugate transpose and False otherwise.

    Args:
        matrix (NDArray[float]): Matrix to check.

    Returns:
        bool: True if the matrix is Hermitian, False otherwise.
    """
    return np.allclose(matrix, matrix.T.conj())


def kronecker_decomposition(M: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the Kronecker decomposition of a 4x4 matrix.
    
    Given a 4x4 matrix `M`, find the two 2x2 matrix `A` and `B` such that their Kronecker
    product is the closest to the matrix `M` in the Frobenius norm.

    Args:
        M (NDArray[float]): 4x4 matrix.

    Returns:
        tuple[NDArray[float], NDArray[float]]: The two 2x2 matrix of the decomposition.

    Raises:
        TypeError: If M is not a NumPy array.
        ValueError: If M is not a 4x4 matrix.

    References:
        G. E. Crooks, “Quantum gates,” March 2024, version 0.11.0, https://threeplusone.com/pubs/on_gates.pdf.
        C. F. Van Loan, “The ubiquitous Kronecker product”, J. Comput. Appl. Math., vol. 123, no. 1–2, pp. 85–100, Nov. 2000,
        https://doi.org/10.1016/S0377-0427(00)00393-9.
    """
    if not isinstance(M, np.ndarray):
        raise TypeError(f"The input matrix must be a numpy object, but got {type(M).__name__}.")
    elif M.shape != (4, 4):
        raise ValueError(f"The input matrix must be 4x4, but received {M.shape}.")

    M = M.reshape(2, 2, 2, 2)
    M = M.transpose(0, 2, 1, 3)
    M = M.reshape(4, 4)

    u, sv, vh = np.linalg.svd(M)

    A = np.sqrt(sv[0]) * u[:, 0].reshape(2, 2)
    B = np.sqrt(sv[0]) * vh[0, :].reshape(2, 2)
    return A, B


class CanonicalDecomposition(NamedTuple):
    """Output of the `canonical_decomposition` function."""
    A: NDArray[np.floating]
    B: NDArray[np.floating]
    t: tuple[float, float, float]
    phase: float | np.floating

def canonical_decomposition(U: NDArray[np.floating]) -> CanonicalDecomposition:
    """
    Perform the canonical decomposition of a given 4x4 unitary matrix.

    Given a 4x4 unitary matrix `U`, find the phase `alpha`, the two 4x4 local unitaries `A` and `B`, and
    the three parameters of the canonical gate to decompose the input matrix `U` like

    `U = exp(i*alpha) * B @ Can(tx, ty, tz) @ A`.

    Can(tx, ty, tz) is the canonical gate defined as `exp(-i*pi/2 * (tx * XX + ty * YY + tz * ZZ))`, where
    XX, YY, and ZZ are the Kronecker products of the Pauli matrices.

    Args:
        U (NDArray[float]): 4x4 unitary matrix.

    Returns:
        CanonicalDecomposition:
        A namedtuple with the following attributes:
            
            A (NDArray[float]): 4 x 4 matrix A of the decomposition. A is the Kronecker product of two 2 x 2 matrices.
            B (NDArray[float]): 4 x 4 matrix B of the decomposition. B is the Kronecker product of two 2 x 2 matrices.
            t (tuple[float, float, float]): The three coordinates (tx, ty, tz) of the canonical gate.
            phase (float): Phase of the unitary matrix.

    Raises:
        TypeError: If the matrix U is not a numpy object.
        ValueError: If U is not a 4 x 4 unitary matrix.

    References:
        G. E. Crooks, “Quantum gates,” March 2024, version 0.11.0, https://threeplusone.com/pubs/on_gates.pdf
        Jun Zhang, Jiri Vala, Shankar Sastry, and K. Birgitta Whaley. Geometric theory of nonlocal two-qubit operations.
        Phys. Rev. A, 67:042313 (2003), https://arxiv.org/pdf/quant-ph/0209120
    """
    if not isinstance(U, np.ndarray):
        raise TypeError(f"Matrix U must be a numpy object, but received {type(U).__name__}.")
    elif U.shape != (4, 4):
        raise ValueError(f"U must be a 4x4 matrix but has shape {U.shape}.")
    elif not is_unitary(U):
        raise ValueError("U must be a unitary matrix.")

    # Magic gate M is used to transform U into the magic basis to get V and diagonalize V.T@V.
    # The magic basis has two interesting properties:
    # 1. The Kronecker product of two single-qubit gates is a special orthogonal matrix Q in the magic basis;
    # 2. The canonical gate is a diagonal matrix D in the magic basis.

    # Extract the phase of U and normalize the matrix to remove its global phase.
    det_U = np.complex128(np.linalg.det(U))
    phase = np.angle(det_U) / 4
    U = U * np.exp(-1.0j * phase)

    # Transform U into the magic basis to get V and diagonalize V.T@V.
    v_matrix = MAGIC_DAG @ U @ MAGIC
    v_vt_matrix = v_matrix.T @ v_matrix

    # The matrix V.T@V is diagonalized. The eigenvectors are the lines of Q1.
    # For numerical precision purpose, we use the eigh function when dealing with hermitian or symmetric matrices. We also need to symmetrize the matrix
    # to ensure that the eigenvectors are real. If the matrix is not hermitian, we use the eig function.
    if is_hermitian(v_vt_matrix):
        eigenval, eigenvec = np.linalg.eigh((v_vt_matrix + v_vt_matrix.T.conj()) / 2)
    elif is_hermitian(1.0j * v_vt_matrix):
        v_vt_matrix = 1.0j * v_vt_matrix
        eigenval, eigenvec = np.linalg.eigh((v_vt_matrix + v_vt_matrix.T.conj()) / 2)
        eigenval = -1.0j * eigenval
    else:
        eigenval, eigenvec = np.linalg.eig(v_vt_matrix)

    # Q1 must be a special unitary matrix. If its determinant is -1, swap two eigenvalues
    # and the two associated eigenvectors to invert the sign of the determinant.
    if np.linalg.det(eigenvec) < 0:
        eigenvec[:, [0, 1]] = eigenvec[:, [1, 0]]
        eigenval[[0, 1]] = eigenval[[1, 0]]

    # Compute Q1 and D from the eigenvectors and the eigenvalues of the decomposition.
    q1_matrix = eigenvec.T
    d_matrix = np.sqrt(np.complex128(eigenval))

    # Q2 must be a special unitary matrix. Since Q2 = V@Q1.T@D^-1, and det(V) = det(Q1) = 1, det(D) must be 1.
    # D is obtained from sqrt(D^2) and all its values are defined up to a sign. We can thus ensure det(D) = 1 by changing the
    # sign to one of its value without influencing Q1.
    if np.prod(d_matrix) < 0:
        d_matrix[0] = -d_matrix[0]
    q2_matrix = v_matrix @ q1_matrix.T @ np.diag(1 / d_matrix)

    # Compute the canonical parameters from the eigenvalues.
    diag_angles = -np.angle(d_matrix) / np.pi
    tx = diag_angles[0] + diag_angles[2]
    ty = diag_angles[1] + diag_angles[2]
    tz = diag_angles[0] + diag_angles[1]

    # Construct the function output to return the canonical decomposition
    return_tuple = CanonicalDecomposition(
        A=MAGIC @ q1_matrix @ MAGIC_DAG,
        B=MAGIC @ q2_matrix @ MAGIC_DAG,
        t=(tx, ty, tz),
        phase=phase,
    )

    return return_tuple


def so4_decomposition(U: NDArray[np.floating] | QGate) -> QCircuit:
    """
    Decompose a 4x4 matrix in SO(4) (special orthogonal group) into a series of 4 S gates, 2 H
    gates, 2 CNOT gates and 2 one-qubit unitary gates. The output is a list of tuples containing
    the gates of the decomposition and the qubit on which they act.

    Args:
        U (np.ndarray): 4x4 matrix in SO(4).
        qubit_no (tuple[int, int]): Tuple containing the qubit numbers on which the gate acts.

    Returns:
        list[tuple]: List of tuples containing the gates of the decomposition and the qubit on which
        they act.

    Raises:
        TypeError: If the input matrix is not a numpy object.
        ValueError: If the input matrix is not in SO(4).
    """
    # Check the input matrix
    if isinstance(U, QGate):
        U._calculate_matrix()
        matrix = U.matrix
        if matrix.shape != (4, 4) or not is_orthogonal(matrix) or not is_special(matrix):
            raise ValueError("The input matrix must be a 4x4 special orthogonal matrix.")
        q0, q1 = U.target
        
    elif isinstance(U, np.ndarray):
        if U.shape != (4, 4) or not is_orthogonal(U) or not is_special(U):
            raise ValueError("The input matrix must be a 4x4 special orthogonal matrix.")
        q0, q1 = (0, 1)
    
    else:
        raise TypeError(f"The input matrix must be a numpy object, but received {type(U).__name__}.")

    
    # Decompose the matrix
    a_tensor_b = MAGIC @ matrix @ MAGIC_DAG

    # Extract A and B
    a, b = kronecker_decomposition(a_tensor_b)

    # List of gates to return
    decomposition_circuit = magic_decomp(q0, q1)
    decomposition_circuit.append(
        QCircuit([
            QGate(name = 'A', matrix=a, target=q0),
            QGate(name = 'B', matrix=b, target=q1),
        ])
    )
    decomposition_circuit.append(magic_dag_decomp(q0, q1))

    return decomposition_circuit


def o4_det_minus1_decomposition(
    U: np.ndarray, qubit_no: tuple[int, int] = (0, 1)
) -> list[tuple]:
    """
    Decompose a 4x4 matrix in O(4) (orthogonal group) with a determinant of -1 into a series of 4 S
    gates, 2 H gates, 3 CNOT gates and 2 one-qubit unitary gates. The output is a list of tuples
    containing the gates of the decomposition and the qubit on which they act.

    Args:
        U (np.ndarray): 4x4 matrix in O(4) with a determinant of -1.
        qubit_no (tuple[int, int]): Tuple containing the qubit numbers on which the gate acts.

    Returns:
        list[tuple]: List of tuples containing the gates of the decomposition and the qubit on which
        they act.

    Raises:
        TypeError: If the input matrix is not a numpy object.
        ValueError: If the input matrix is not in O(4) with a determinant of -1.
    """
    # Check the input matrix
    if not isinstance(U, np.ndarray):
        raise TypeError(f"The input matrix must be a numpy object, but received {type(U).__name__}.")
    if U.shape != (4, 4) or not is_orthogonal(U) or not np.isclose(np.linalg.det(U), -1):
        raise ValueError("The input matrix must be a 4x4 orthogonal matrix with a determinant of -1.")
    
    # Decompose the matrix
    a_tensor_b = (
        MAGIC
        @ U
        @ MAGIC_DAG
        @ np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    )

    # Extract A and B
    a, b = kronecker_decomposition(a_tensor_b)

    # List of gates to return
    q0, q1 = qubit_no  # Qubits on which the decomposition acts
    gates = [
        ("S", (q0,)),
        ("S H", (q1,)),
        ("CNOT", (q0, q1)),
        ("CNOT", (q1, q0)),
        (a, (q0,)),
        (b, (q1,)),
        ("CNOT", (q1, q0)),
        ("SDAG", (q0,)),
        ("H SDAG", (q1,)),
    ]

    return gates


def u4_decomposition(
    matrix: np.ndarray, qubit_no: tuple[int, int] = (0, 1)
) -> list[tuple]:
    """
    Decompose a 4x4 matrix in U(4) (unitary group) into a series of 2 S gates, 3 CNOT gates, 7
    one-qubit unitary gates. The output is a list of tuples containing the gates of the
    decomposition and the qubit on which they act.

    Args:
        matrix (np.ndarray): 4x4 matrix.
        qubit_no (tuple[int, int]): Tuple containing the qubit numbers on which the gate acts.

    Returns:
        list[tuple]: List of tuples containing the gates of the decomposition and the qubit on which
        they act.

    Raises:
        ValueError: If the input matrix is not 4x4.
        ValueError: If the input matrix is not unitary.
    """
    # Check the input matrix
    if matrix.shape != (4, 4):
        raise ValueError(f"The input matrix must be 4x4. Got {matrix.shape}.")
    if not is_unitary(matrix):
        raise ValueError("The input matrix must be unitary.")

    # Decompose the matrix
    can_decomp = canonical_decomposition(matrix)
    a_tensor = can_decomp.A
    b_tensor = can_decomp.B
    tx, ty, tz = can_decomp.t
    alpha = can_decomp.alpha

    # Extract A1, A2, B1 and B2
    a1, a2 = kronecker_decomposition(a_tensor)
    b1, b2 = kronecker_decomposition(b_tensor)

    # List of gates to return
    q0, q1 = qubit_no  # Qubits on which the decomposition acts
    gates = [
        # A matrix
        (a1, (q0,)),
        (a2, (q1,)),
        # Canonical gate
        ("S", (q1,)),
        ("CNOT", (q1, q0)),
        (power_pauli_z(tz - 0.5), (q0,)),
        (power_pauli_y(tx - 0.5), (q1,)),
        ("CNOT", (q0, q1)),
        (power_pauli_y(0.5 - ty), (q1,)),
        ("CNOT", (q1, q0)),
        ("SDAG", (q0,)),
        # B matrix
        (b1, (q0,)),
        (b2, (q1,)),
    ]

    return gates


def known_decomposition(matrix: np.ndarray, qubit_no: tuple[int, int] = (0, 1)) -> list[tuple] | None:
    """
    Decompose a 4x4 matrix into a series of CNOT and single-qubit gates using predefined
    decompositions for common gates (SWAP, identity, CNOT).

    Args:
        matrix (np.ndarray): 4x4 matrix.
        qubit_no (tuple[int, int]): Tuple containing the qubit numbers on which the gate acts.

    Returns:
        list[tuple]: List of tuples containing the gates of the decomposition and the qubit on which
        they act, or None if the input matrix is not a known gate.
    
    Raises:
        ValueError: If the input matrix is not 4x4.
        ValueError: If the qubit number is not a tuple of two integers.
    """
    # Check the input matrix
    if matrix.shape != (4, 4):
        raise ValueError(f"The input matrix must be 4x4. Got {matrix.shape}.")
    if not isinstance(qubit_no, tuple) or len(qubit_no) != 2:
        raise ValueError(f"The qubit number must be a tuple of two integers. Got {qubit_no}.")
    
    # Check if the matrix is a known gate
    if (matrix == np.eye(4)).all():  # Identity
        return []
        
    if (matrix == np.eye(4)[[0, 2, 1, 3]]).all():  # SWAP
        return [("CNOT", qubit_no), ("CNOT", qubit_no[::-1]), ("CNOT", qubit_no)]
        
    if (matrix == np.eye(4)[[0, 1, 3, 2]]).all():  # CNOT
        return [("CNOT", qubit_no)]
        
    if (matrix == np.eye(4)[[0, 3, 2, 1]]).all():  # CNOT (flipped)
        return [("CNOT", qubit_no[::-1])]

    if (matrix == np.eye(4)[[0, 2, 3, 1]]).all():  # CNOT, then CNOT_flipped
        return [("CNOT", qubit_no), ("CNOT", qubit_no[::-1])]

    if (matrix == np.eye(4)[[0, 3, 1, 2]]).all():  # CNOT_flipped, then CNOT
        return [("CNOT", qubit_no[::-1]), ("CNOT", qubit_no)]
        
    return None


def tqg_decomposition(
    matrix: np.ndarray, qubit_no: tuple[int, int] = (0, 1)
) -> list[tuple[np.ndarray, tuple[int]]]:
    """
    Decompose any two-qubits gate into a series of CNOT and single-qubit gates. This function
    determines which decomposition to use based on the Lie group of the input matrix (SO(4), O(4),
    U(4)) or uses a predefined decomposition if the gate is a common one (SWAP, identity, CNOT).

    Args:
        matrix (np.ndarray): 4x4 matrix.
        qubit_no (tuple[int, int]): Tuple containing the qubit numbers on which the gate acts.
    
    Returns:
        list[tuple]: List of tuples containing the gates of the decomposition and the qubit on which
        they act.
    
    Raises:
        ValueError: If the input matrix is not 4x4.
        ValueError: If the input matrix is not unitary.
    """
    # Check the input matrix
    if matrix.shape != (4, 4):
        raise ValueError(f"The input matrix must be 4x4. Got {matrix.shape}.")
    if not is_unitary(matrix):
        raise ValueError("The input matrix must be unitary.")

    # Check if the matrix is a known gate
    known_decomp = known_decomposition(matrix, qubit_no)
    if known_decomp is not None:
        return known_decomp

    # Check the Lie group of the matrix
    if is_orthogonal(matrix):
        if is_special(matrix):  # Special orthogonal group SO(4)
            return so4_decomposition(matrix, qubit_no)
        else:  # Orthogonal group O(4) with det = -1
            return o4_det_minus1_decomposition(matrix, qubit_no)

    else:  # Unitary group U(4)
        return u4_decomposition(matrix, qubit_no)

