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
This module contains functions to decompose general 2-qubits quantum gates into single-qubit and CNOT gates.

The module contains the following functions:

- :func:`kronecker_decomposition`: Decompose a 4 x 4 matrix into two 2 x 2 matrices such that their Kronecker product is the closest to the original matrix.
- :func:`so4_decomposition`: Decompose a 4 x 4 matrix in SO(4) into a circuit of 2 CNOT gates and 8 single-qubit gates.
- :func:`o4_det_minus1_decomposition`: Decompose a 4 x 4 matrix in O(4) with a determinant of -1 into a circuit of 3 CNOT gates and 8 single-qubit gates.
- :func:`canonical_decomposition`: Decompose a 4 x 4 unitary matrix into a global phase, two local 4 x 4 matrices, and the three parameters of the canonical gate.
- :func:`u4_decomposition`: Decompose a 4 x 4 matrix in U(4) into a circuit of 3 CNOT and 7 single-qubit gates.
- :func:`known_decomposition`: Decompose a 4 x 4 matrix into a circuit of CNOT and single-qubit gates using predefined decompositions for common gates.
- :func:`cnot_decomposition`: Decompose any two-qubits gate into a circuit of CNOT and single-qubit gates.

The function ``cnot_decomposition`` is the main function of the module. It decomposes any 4 x 4 unitary matrix into a
circuit of CNOT and single-qubit gates. The function determines which decomposition to use based on the Lie group of
the input matrix (SO(4), O(4), U(4)) or uses a predefined decomposition if the gate is common (SWAP, identity, CNOT).
The function returns a list of ``QGate`` objects representing the circuit decomposition.

For more details on the theory, see

.. [1] Crooks, G. E., “Quantum gates - Gates, States, and Circuits”, version 0.11.0., Mar. 2024, https://threeplusone.com/pubs/on_gates.pdf
.. [2] Van Loan, C. F., “The ubiquitous Kronecker product”, J. Comput. Appl. Math., vol. 123, no. 1-2, pp. 85-100, Nov. 2000, https://doi.org/10.1016/S0377-0427(00)00393-9
.. [3] Jun Zhang, Jiri Vala, Shankar Sastry, and K. Birgitta Whaley. Geometric theory of nonlocal two-qubit operations. Phys. Rev. A, 67:042313 (2003), https://arxiv.org/pdf/quant-ph/0209120
.. [4] Vatan, F., Williams, C., “Optimal quantum circuits for general two-qubit gates”, arXiv [quant-ph], 2003, https://arxiv.org/abs/quant-ph/0308006
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from qdecomp.utils import gates
from qdecomp.utils import QGate
from qdecomp.decompositions.common_gate_decompositions import common_decompositions
from qdecomp.utils import is_hermitian, is_orthogonal, is_special, is_unitary

__all__ = [
    "kronecker_decomposition",
    "so4_decomposition",
    "o4_det_minus1_decomposition",
    "canonical_decomposition",
    "u4_decomposition",
    "cnot_decomposition",
    "known_decomposition",
]

SQRT2 = np.sqrt(2)

# The magic gate is a 4 x 4 matrix used in many decompositions of quantum gates.
MAGIC = gates.MAGIC
MAGIC_DAG = MAGIC.T.conj()


def kronecker_decomposition(
    matrix: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the Kronecker decomposition of a 4 x 4 matrix.

    Given a 4 x 4 matrix ``M``, find the two 2 x 2 matrix ``A`` and ``B`` such that their Kronecker
    product is the closest to the matrix `M` in the Frobenius norm.

    Args:
        matrix (NDArray[float]): 4 x 4 matrix.

    Returns:
        tuple[NDArray[float], NDArray[float]]: The two 2 x 2 matrix of the decomposition.

    Raises:
        TypeError: If matrix is not a numpy array.
        ValueError: If matrix is not a 4 x 4 matrix.

    Examples:
        >>> # Define two 2 x 2 matrices
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])

        >>> # Compute the Kronecker decomposition
        >>> a, b = kronecker_decomposition(np.kron(A, B))
        >>> print(np.allclose(np.kron(A, B), np.kron(a, b)))
        True

    References:
        .. [1] Crooks, G. E., “Quantum gates - Gates, States, and Circuits”, version 0.11.0., Mar. 2024, https://threeplusone.com/pubs/on_gates.pdf
        .. [2] Van Loan, C. F., “The ubiquitous Kronecker product”, J. Comput. Appl. Math., vol. 123, no. 1-2, pp. 85-100, Nov. 2000, https://doi.org/10.1016/S0377-0427(00)00393-9
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(
            f"The input matrix must be a numpy object, but got {type(matrix).__name__}."
        )
    elif matrix.shape != (4, 4):
        raise ValueError(f"The input matrix must be 4 x 4, but received {matrix.shape}.")

    matrix = matrix.reshape(2, 2, 2, 2)
    matrix = matrix.transpose(0, 2, 1, 3)
    matrix = matrix.reshape(4, 4)

    u, sv, vh = np.linalg.svd(matrix)

    a_matrix = np.sqrt(sv[0]) * u[:, 0].reshape(2, 2)
    b_matrix = np.sqrt(sv[0]) * vh[0, :].reshape(2, 2)
    return a_matrix, b_matrix


def so4_decomposition(U: NDArray[np.floating] | QGate) -> list[QGate]:
    """
    Circuit decomposition of SO(4) matrices.

    Decompose a 4 x 4 matrix in SO(4) (special orthogonal group) into a circuit of 2 CNOT gates
    and 8 single-qubit gates. The output is a list of QGate objects.

    Args:
        U (NDArray[float]): 4 x 4 matrix in SO(4).

    Returns:
        list[QGate]: Circuit decomposition of the input matrix. The output is a list of QGate objects.

    Raises:
        TypeError: If the input matrix is not a numpy array or a QGate object.
        ValueError: If the input matrix is not in SO(4).

    References:
        .. [1] Crooks, G., E. “Quantum gates - Gates, States, and Circuits”, version 0.11.0., Mar. 2024, https://threeplusone.com/pubs/on_gates.pdf
        .. [2] Vatan, F., Williams, C., “Optimal quantum circuits for general two-qubit gates”, arXiv [quant-ph], 2003, https://arxiv.org/abs/quant-ph/0308006
    """
    # Check the input matrix
    if isinstance(U, QGate):
        matrix = U.matrix
        if matrix.shape != (4, 4) or not is_orthogonal(matrix) or not is_special(matrix):
            raise ValueError("The input matrix must be a 4 x 4 special orthogonal matrix.")
        q0, q1 = U.matrix_target

    elif isinstance(U, np.ndarray):
        if U.shape != (4, 4) or not is_orthogonal(U) or not is_special(U):
            raise ValueError("The input matrix must be a 4 x 4 special orthogonal matrix.")
        matrix = U
        q0, q1 = (0, 1)

    else:
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Decompose the matrix
    a_tensor_b = MAGIC @ matrix @ MAGIC_DAG

    # Extract A and B
    a, b = kronecker_decomposition(a_tensor_b)

    # List of gates to return
    decomposition_circuit = (
        common_decompositions("MAGIC", q0, q1)
        + [
            QGate.from_matrix(a, name="A", matrix_target=(q0,)),
            QGate.from_matrix(b, name="B", matrix_target=(q1,)),
        ]
        + common_decompositions("MAGIC_DAG", q0, q1)
    )

    return decomposition_circuit


def o4_det_minus1_decomposition(U: NDArray[np.floating] | QGate) -> list[QGate]:
    """
    Circuit decomposition of O(4) matrices with a determinant of -1.

    Decompose a 4 x 4 matrix in O(4) (orthogonal group) with a determinant of -1 into a circuit of
    3 CNOT and 8 single-qubit gates. The output is a list of QGate objects.

    Args:
        U (NDArray[float]): 4 x 4 matrix in O(4) with a determinant of -1.

    Returns:
        list[QGate]: Circuit decomposition of the input matrix. The output is a list of QGate objects.

    Raises:
        TypeError: If the input matrix is not a numpy array or a QGate object.
        ValueError: If the input matrix is not in O(4) with a determinant of -1.

    References:
        .. [1] Crooks, G. E., “Quantum gates - Gates, States, and Circuits”, version 0.11.0., Mar. 2024, https://threeplusone.com/pubs/on_gates.pdf
        .. [2] Vatan, F., Williams, C., “Optimal quantum circuits for general two-qubit gates”, arXiv [quant-ph], 2003, https://arxiv.org/abs/quant-ph/0308006
    """
    # Check the input matrix
    if isinstance(U, QGate):
        matrix = U.matrix
        if (
            matrix.shape != (4, 4)
            or not is_orthogonal(matrix)
            or not np.isclose(np.linalg.det(matrix), -1)
        ):
            raise ValueError(
                "The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1."
            )
        q0, q1 = U.matrix_target

    elif isinstance(U, np.ndarray):
        if U.shape != (4, 4) or not is_orthogonal(U) or not np.isclose(np.linalg.det(U), -1):
            raise ValueError(
                "The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1."
            )
        matrix = U
        q0, q1 = (0, 1)

    else:
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Decompose the matrix
    a_tensor_b = MAGIC @ matrix @ MAGIC_DAG @ gates.SWAP

    # Extract A and B
    a, b = kronecker_decomposition(a_tensor_b)

    # List of gates to return
    decomposition_circuit = (
        common_decompositions("MAGIC", q0, q1)[:-1]
        + [
            QGate.from_tuple(("CNOT", (q0, q1), 0)),
            QGate.from_tuple(("CNOT", (q1, q0), 0)),
            QGate.from_matrix(a, name="A", matrix_target=(q0,)),
            QGate.from_matrix(b, name="B", matrix_target=(q1,)),
        ]
        + common_decompositions("MAGIC_DAG", q0, q1)
    )

    return decomposition_circuit


class CanonicalDecomposition(NamedTuple):
    """
    Output of the `canonical_decomposition` function.

    Attributes:
        A (NDArray[float]): 4 x 4 matrix A of the decomposition. A is the Kronecker product of two 2 x 2 matrices.
        B (NDArray[float]): 4 x 4 matrix B of the decomposition. B is the Kronecker product of two 2 x 2 matrices.
        t (tuple[float, float, float]): The three coordinates (tx, ty, tz) of the canonical gate.
        phase (float): Phase of the unitary matrix.
    """

    A: NDArray[np.floating]
    """4 x 4 matrix A of the decomposition."""

    B: NDArray[np.floating]
    """4 x 4 matrix B of the decomposition."""

    t: tuple[float, float, float]
    """Coordinates (tx, ty, tz) of the canonical gate."""

    phase: float | np.floating
    """Phase of the unitary matrix."""


def canonical_decomposition(U: NDArray[np.floating]) -> CanonicalDecomposition:
    """
    Perform the canonical decomposition of a given 4 x 4 unitary matrix.

    Given a 4 x 4 unitary matrix ``U``, find the phase ``alpha``, the two 4 x 4 local unitaries ``A`` and ``B``, and
    the three parameters of the canonical gate to decompose the input matrix ``U`` like

    .. math:: U = e^{i \\alpha} B \\times Can(t_x, t_y, t_z) \\times A.

    ``Can(tx, ty, tz)`` is the canonical gate defined as

    .. math:: Can(t_x, t_y, t_z) = exp(-i\\frac{\\pi}{2} (t_x X\\otimes X + t_y Y\\otimes Y + t_z Z\\otimes Z)),

    where `X`, `Y`, and `Z` are the Pauli matrices.

    Args:
        U (NDArray[float]): 4 x 4 unitary matrix.

    Returns:
        CanonicalDecomposition:
        A namedtuple with the following attributes:
            - A (NDArray[float]): 4 x 4 matrix A of the decomposition. A is the Kronecker product of two 2 x 2 matrices.
            - B (NDArray[float]): 4 x 4 matrix B of the decomposition. B is the Kronecker product of two 2 x 2 matrices.
            - t (tuple[float, float, float]): The three coordinates (tx, ty, tz) of the canonical gate.
            - phase (float): Phase of the unitary matrix.

    Raises:
        TypeError: If the matrix U is not a numpy object.
        ValueError: If U is not a 4 x 4 unitary matrix.

    Examples:
        >>> # Define a 4 x 4 unitary matrix
        >>> from scipy.stats import unitary_group
        >>> U = unitary_group.rvs(4)

        >>> # Perform the canonical decomposition and reconstruct the matrix
        >>> decomp = canonical_decomposition(U)
        >>> reconstructed_matrix = np.exp(1.j * decomp.phase) * decomp.B @ gates.canonical_gate(*decomp.t) @ decomp.A

        >>> # Check if the decomposition is correct
        >>> print(np.allclose(U, reconstructed_matrix))
        True

    References:
        .. [1] Crooks, G. E., “Quantum gates - Gates, States, and Circuits”, version 0.11.0., Mar. 2024, https://threeplusone.com/pubs/on_gates.pdf
        .. [2] Jun Zhang, Jiri Vala, Shankar Sastry, and K. Birgitta Whaley. Geometric theory of nonlocal two-qubit operations. Phys. Rev. A, 67:042313 (2003), https://arxiv.org/pdf/quant-ph/0209120
    """
    if not isinstance(U, np.ndarray):
        raise TypeError(f"Matrix U must be a numpy object, but received {type(U).__name__}.")
    elif U.shape != (4, 4):
        raise ValueError(f"U must be a 4 x 4 matrix but has shape {U.shape}.")
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


def u4_decomposition(U: NDArray[np.floating] | QGate) -> list[QGate]:
    """
    Circuit decomposition of U(4) matrices.

    Decompose a 4 x 4 matrix in U(4) (unitary group) into a circuit of 3 CNOT a 7 single-ubit gates.
    The output is a list of QGate objects.

    Args:
        U (NDArray[float]): 4 x 4 matrix in U(4).

    Returns:
        list[QGate]: Circuit decomposition of the input matrix. The output is a list of QGate objects.

    Raises:
        TypeError: If the input matrix is not a numpy array or a QGate object.
        ValueError: If the input matrix is not in U(4).

    References:
        .. [1] Crooks, G. E., “Quantum gates - Gates, States, and Circuits”, version 0.11.0., Mar. 2024, https://threeplusone.com/pubs/on_gates.pdf
        .. [2] Vatan, F., Williams, C., “Optimal quantum circuits for general two-qubit gates”, arXiv [quant-ph], 2003, https://arxiv.org/abs/quant-ph/0308006
    """
    # Check the input matrix
    if isinstance(U, QGate):
        matrix = U.matrix
        if matrix.shape != (4, 4) or not is_unitary(matrix):
            raise ValueError("The input matrix must be a 4 x 4 unitary matrix.")
        q0, q1 = U.matrix_target

    elif isinstance(U, np.ndarray):
        if U.shape != (4, 4) or not is_unitary(U):
            raise ValueError("The input matrix must be a 4 x 4 unitary matrix.")
        matrix = U
        q0, q1 = (0, 1)

    else:
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Decompose the matrix
    canonical_decomp = canonical_decomposition(matrix)
    a_matrix = canonical_decomp.A
    b_matrix = canonical_decomp.B
    tx, ty, tz = canonical_decomp.t

    # Extract A1, A2, B1 and B2
    a1, a2 = kronecker_decomposition(a_matrix)
    b1, b2 = kronecker_decomposition(b_matrix)
    a2 = gates.S @ a2
    b1 = b1 @ gates.S.conj()

    # List of gates to return
    decomposition_circuit = [
        QGate.from_matrix(a1, name="A1", matrix_target=(q0,)),
        QGate.from_matrix(a2, name="A2", matrix_target=(q1,)),
        QGate.from_tuple(("CNOT", (q1, q0), 0)),
        QGate.from_matrix(gates.power_pauli_z(tz - 0.5), name="PZ", matrix_target=(q0,)),
        QGate.from_matrix(gates.power_pauli_y(tx - 0.5), name="PY", matrix_target=(q1,)),
        QGate.from_tuple(("CNOT", (q0, q1), 0)),
        QGate.from_matrix(gates.power_pauli_y(0.5 - ty), name="PY", matrix_target=(q1,)),
        QGate.from_tuple(("CNOT", (q1, q0), 0)),
        QGate.from_matrix(b1, name="B1", matrix_target=(q0,)),
        QGate.from_matrix(b2, name="B2", matrix_target=(q1,)),
    ]

    return decomposition_circuit


def known_decomposition(U: NDArray[np.floating] | QGate) -> list[QGate] | None:
    """
    Circuit decompositions of common 4 x 4 matrices.

    Decompose a 4 x 4 matrix into a circuit of CNOT and single-qubit gates using predefined
    decompositions for common gates (SWAP, identity, CNOT, etc.). The output is a list of QGate objects.
    If the decomposition is not known, the function returns None.

    Args:
        U (NDArray[float]): 4 x 4 matrix in U(4).

    Returns:
        (list[QGate] | None): Circuit decomposition of the input matrix.
        The output is a list of QGate objects. Return None if the decomposition is not known.

    Raises:
        TypeError: If the input matrix is not a numpy array or a QGate object.
        ValueError: If the input matrix is not in U(4).
    """
    # Check the input matrix
    if isinstance(U, QGate):
        matrix = U.matrix
        if matrix.shape != (4, 4) or not is_unitary(matrix):
            raise ValueError("The input matrix must be a 4 x 4 unitary matrix.")
        q0, q1 = U.matrix_target

    elif isinstance(U, np.ndarray):
        if U.shape != (4, 4) or not is_unitary(U):
            raise ValueError("The input matrix must be a 4 x 4 unitary matrix.")
        matrix = U
        q0, q1 = (0, 1)

    else:
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Check if the matrix is a known gate
    if (matrix == np.eye(4)).all():  # Identity
        return []

    if (matrix == gates.CNOT).all():  # CNOT
        return [QGate.from_tuple(("CNOT", (q0, q1), 0))]

    if (matrix == gates.CNOT1).all():  # CNOT (flipped)
        return [QGate.from_tuple(("CNOT", (q1, q0), 0))]

    if (matrix == gates.DCNOT).all():  # DCNOT (CNOT, then CNOT flipped)
        return common_decompositions("DCNOT", q0, q1)

    if (matrix == gates.INV_DCNOT).all():  # INV_DCNOT (CNOT flipped, then CNOT)
        return common_decompositions("INV_DCNOT", q0, q1)

    if (matrix == gates.SWAP).all():  # SWAP
        return common_decompositions("SWAP", q0, q1)

    if (matrix == gates.ISWAP).all():  # ISWAP
        return common_decompositions("ISWAP", q0, q1)

    if (matrix == gates.CY).all():  # Controlled Y
        return common_decompositions("CY", q0, q1)

    if (matrix == gates.CZ).all():  # Controlled Z
        return common_decompositions("CZ", q0, q1)

    if (matrix == gates.CH).all():  # Controlled Hadamard
        return common_decompositions("CH", q0, q1)

    if (matrix == gates.MAGIC).all():  # Magic gate
        return common_decompositions("MAGIC", q0, q1)

    if (matrix == gates.MAGIC.conj().T).all():  # Magic gate (Hermitian conjugate)
        return common_decompositions("MAGIC_DAG", q0, q1)

    return None


def cnot_decomposition(U: NDArray[np.floating]) -> list[QGate]:
    """
    Circuit decomposition of 4 x 4 quantum gates.

    Decompose any two-qubits gate into a circuit of CNOT and single-qubit gates. The function
    determines which decomposition to use based on the Lie group of the input matrix (SO(4), O(4),
    U(4)) or uses a predefined decomposition if the gate is common (SWAP, identity, CNOT, etc.). The output
    is a list of QGate objects.

    Args:
        U (NDArray[float]): 4 x 4 unitary matrix.

    Returns:
        list[QGate]: Circuit decomposition of the input matrix. The output is a list of QGate objects.

    Raises:
        TypeError: If the input matrix is not a numpy array or a QGate object.
        ValueError: If the input matrix is not a 4 x 4 unitary matrix.

    Examples:
        >>> # Use an arbitrary 4 x 4 unitary matrix
        >>> from scipy.stats import unitary_group
        >>> U = unitary_group.rvs(4)
        >>> # Decompose the matrix into a circuit of CNOT and single-qubit gates
        >>> circuit = cnot_decomposition(U)
    """
    # Check the input matrix
    if isinstance(U, QGate):
        matrix = U.matrix
        if matrix.shape != (4, 4) or not is_unitary(matrix):
            raise ValueError("The input matrix must be a 4 x 4 unitary matrix.")

    elif isinstance(U, np.ndarray):
        if U.shape != (4, 4) or not is_unitary(U):
            raise ValueError("The input matrix must be a 4 x 4 unitary matrix.")
        matrix = U

    else:
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Check if the decomposition is known
    known_decomp = known_decomposition(U)
    if known_decomp is not None:
        return known_decomp

    # Check the Lie group of the matrix and return the corresponding decomposition
    if is_orthogonal(matrix):
        if is_special(matrix):
            return so4_decomposition(U)
        return o4_det_minus1_decomposition(U)
    return u4_decomposition(U)
