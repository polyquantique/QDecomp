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

- :func:`kronecker_decomp`: Decompose a 4 x 4 matrix into two 2 x 2 matrices such that their Kronecker product is the closest to the original matrix.
- :func:`so4_decomp`: Decompose a 4 x 4 matrix in SO(4) into a circuit of 2 CNOT gates and 8 single-qubit gates.
- :func:`o4_det_minus1_decomp`: Decompose a 4 x 4 matrix in O(4) with a determinant of -1 into a circuit of 3 CNOT gates and 8 single-qubit gates.
- :func:`canonical_decomp`: Decompose a 4 x 4 unitary matrix into a global phase, two local 4 x 4 matrices, and the three parameters of the canonical gate.
- :func:`u4_decomp`: Decompose a 4 x 4 matrix in U(4) into a circuit of 3 CNOT and 7 single-qubit gates.
- :func:`known_decomp`: Decompose a 4 x 4 matrix into a circuit of CNOT and single-qubit gates using predefined decompositions for common gates.
- :func:`cnot_decomp`: Decompose any two-qubits gate into a circuit of CNOT and single-qubit gates.
- :func:`sqg_decomp`: Decompose any two-qubits gate into a series of Clifford+T gates.

The function :func:`sqg_decomp` is the main function of the module.
It decomposes any 4 x 4 unitary matrix into a circuit of Clifford+T gates by using the :func:`cnot_decomp` and :func:`sqg_decomp` functions.

The function ``cnot_decomp`` is the second most important function of the module.
It decomposes any 4 x 4 unitary matrix into a circuit of CNOT and single-qubit gates.
The function determines which decomposition to use based on the Lie group of the input matrix (SO(4), O(4), U(4)) or uses a predefined decomposition if the gate is common (SWAP, identity, CNOT).
The function returns a list of ``QGate`` objects representing the circuit decomposition.

For more details on the theory, see :cite:`decomp_crooks, decomp_vanloan_2000, decomp_zhang_2003, decomp_vatan_2004`.
"""

from __future__ import annotations

from typing import NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

from qdecomp.decompositions.common_gate_decompositions import common_decomp
from qdecomp.decompositions.sqg import sqg_decomp
from qdecomp.utils import QGate, gates
from qdecomp.utils.gates_utils import is_hermitian, is_orthogonal, is_special, is_unitary

__all__ = [
    "kronecker_decomp",
    "so4_decomp",
    "o4_det_minus1_decomp",
    "canonical_decomp",
    "u4_decomp",
    "cnot_decomp",
    "known_decomp",
    "tqg_decomp",
]

SQRT2 = np.sqrt(2)

# The magic gate is a 4 x 4 matrix used in many decompositions of quantum gates.
MAGIC = gates.MAGIC
MAGIC_DAG = MAGIC.T.conj()


def kronecker_decomp(
    matrix: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the Kronecker decomposition of a 4 x 4 matrix.

    Given a 4 x 4 matrix ``M``, find the two 2 x 2 matrix ``A`` and ``B`` such that their Kronecker
    product is the closest to the matrix ``M`` in the Frobenius norm :cite:`decomp_crooks, decomp_vanloan_2000`.

    Args:
        matrix (NDArray[float]): 4 x 4 matrix.

    Returns:
        tuple[NDArray[float], NDArray[float]]: The two 2 x 2 matrix of the decomposition.

    Raises:
        TypeError: If matrix is not a numpy array.
        ValueError: If matrix is not a 4 x 4 matrix.

    **Example**:

    .. code-block:: python

        >>> import numpy as np
        >>> from qdecomp.decompositions import kronecker_decomp

        # Define two 2 x 2 matrices
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])

        # Compute the Kronecker decomposition
        >>> a, b = kronecker_decomp(np.kron(A, B))
        >>> print(np.allclose(np.kron(A, B), np.kron(a, b)))
        True
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(
            f"The input matrix must be a numpy object, but got {type(matrix).__name__}."
        )
    elif matrix.shape != (4, 4):
        raise ValueError(f"The input matrix must be 4 x 4, but received {matrix.shape}.")

    # Reshape the matrix to a 2 x 2 x 2 x 2 tensor
    matrix = matrix.reshape(2, 2, 2, 2)

    # Transpose the tensor
    matrix = matrix.transpose(0, 2, 1, 3)

    # Reshape the tensor to a 4 x 4 matrix
    matrix = matrix.reshape(4, 4)

    # Compute the singular value decomposition
    u, sv, vh = np.linalg.svd(matrix)

    a_matrix = np.sqrt(sv[0]) * u[:, 0].reshape(2, 2)
    b_matrix = np.sqrt(sv[0]) * vh[0, :].reshape(2, 2)
    return a_matrix, b_matrix


def so4_decomp(U: NDArray[np.floating] | QGate) -> list[QGate]:
    """Circuit decomposition of SO(4) matrices.

    Decompose a 4 x 4 matrix in SO(4) (special orthogonal group) into a circuit of 2 CNOT gates
    and 8 single-qubit gates :cite:`decomp_crooks, decomp_vatan_2004`. The output is a list of QGate objects.

    Args:
        U (NDArray[float]): 4 x 4 matrix in SO(4).

    Returns:
        list[QGate]: Circuit decomposition of the input matrix. The output is a list of QGate objects.

    Raises:
        TypeError: If the input matrix is not a numpy array or a QGate object.
        ValueError: If the input matrix is not in SO(4).
    """
    # Check input type
    if not isinstance(U, (np.ndarray, QGate)):
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Check the input matrix
    matrix = getattr(U, "matrix", U)
    q = getattr(U, "target", (0, 1))

    if matrix.shape != (4, 4) or not is_orthogonal(matrix) or not is_special(matrix):
        raise ValueError("The input matrix must be a 4 x 4 special orthogonal matrix.")

    # Decompose the matrix
    a_tensor_b = MAGIC @ matrix @ MAGIC_DAG

    # Extract A and B
    a, b = kronecker_decomp(a_tensor_b)

    # List of gates to return
    decomposition_circuit = (
        common_decomp("MAGIC", q[0], q[1])
        + [
            QGate.from_matrix(a, name="A", target=(q[0],)),
            QGate.from_matrix(b, name="B", target=(q[1],)),
        ]
        + common_decomp("MAGIC_DAG", q[0], q[1])
    )

    return decomposition_circuit


def o4_det_minus1_decomp(U: NDArray[np.floating] | QGate) -> list[QGate]:
    """Circuit decomposition of O(4) matrices with a determinant of -1.

    Decompose a 4 x 4 matrix in O(4) (orthogonal group) with a determinant of -1 into a circuit of
    3 CNOT and 8 single-qubit gates :cite:`decomp_crooks, decomp_vatan_2004`. The output is a list of QGate objects.

    Args:
        U (NDArray[float]): 4 x 4 matrix in O(4) with a determinant of -1.

    Returns:
        list[QGate]: Circuit decomposition of the input matrix. The output is a list of QGate objects.

    Raises:
        TypeError: If the input matrix is not a numpy array or a QGate object.
        ValueError: If the input matrix is not in O(4) with a determinant of -1.
    """
    # Check input type
    if not isinstance(U, (np.ndarray, QGate)):
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Check the input matrix
    matrix = getattr(U, "matrix", U)
    q = getattr(U, "target", (0, 1))

    if (
        matrix.shape != (4, 4)
        or not is_orthogonal(matrix)
        or not np.isclose(np.linalg.det(matrix), -1)
    ):
        raise ValueError(
            "The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1."
        )

    # Decompose the matrix
    a_tensor_b = MAGIC @ matrix @ MAGIC_DAG @ gates.SWAP

    # Extract A and B
    a, b = kronecker_decomp(a_tensor_b)

    # List of gates to return
    decomposition_circuit = (
        common_decomp("MAGIC", q[0], q[1])[:-1]
        + [
            QGate.from_tuple(("CNOT", (q[0], q[1]), 0)),
            QGate.from_tuple(("CNOT1", (q[0], q[1]), 0)),
            QGate.from_matrix(a, name="A", target=(q[0],)),
            QGate.from_matrix(b, name="B", target=(q[1],)),
        ]
        + common_decomp("MAGIC_DAG", q[0], q[1])
    )

    return decomposition_circuit


class CanonicalDecomposition(NamedTuple):
    """Output of the `canonical_decomp` function.

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


def canonical_decomp(U: NDArray[np.floating]) -> CanonicalDecomposition:
    """Perform the canonical decomposition of a given 4 x 4 unitary matrix.

    Given a 4 x 4 unitary matrix ``U``, find the phase ``alpha``, the two 4 x 4 local unitaries ``A`` and ``B``, and
    the three parameters of the canonical gate to decompose the input matrix ``U`` like

    .. math:: U = e^{i \\alpha} B \\times Can(t_x, t_y, t_z) \\times A.

    ``Can(tx, ty, tz)`` is the canonical gate defined as

    .. math:: Can(t_x, t_y, t_z) = exp(-i\\frac{\\pi}{2} (t_x X\\otimes X + t_y Y\\otimes Y + t_z Z\\otimes Z)),

    where `X`, `Y`, and `Z` are the Pauli matrices :cite:`decomp_crooks, decomp_zhang_2003`.

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

    **Example**:

    .. code-block:: python

        >>> from scipy.stats import unitary_group
        >>> import numpy as np
        >>> from qdecomp.decompositions import canonical_decomp
        >>> from qdecomp.utils import gates

        # Define a 4 x 4 unitary matrix
        >>> U = unitary_group.rvs(4)

        # Perform the canonical decomposition and reconstruct the matrix
        >>> decomp = canonical_decomp(U)
        >>> reconstructed_matrix = np.exp(1.j * decomp.phase) * decomp.B @ gates.canonical_gate(*decomp.t) @ decomp.A

        # Check if the decomposition is correct
        >>> print(np.allclose(U, reconstructed_matrix))
        True
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


def u4_decomp(U: NDArray[np.floating] | QGate) -> list[QGate]:
    """Circuit decomposition of U(4) matrices.

    Decompose a 4 x 4 matrix in U(4) (unitary group) into a circuit of 3 CNOT a 7 single-qubit gates :cite:`decomp_crooks, decomp_vatan_2004`.
    The output is a list of QGate objects.

    Args:
        U (NDArray[float]): 4 x 4 matrix in U(4).

    Returns:
        list[QGate]: Circuit decomposition of the input matrix. The output is a list of QGate objects.

    Raises:
        TypeError: If the input matrix is not a numpy array or a QGate object.
        ValueError: If the input matrix is not in U(4).
    """
    # Check input type
    if not isinstance(U, (np.ndarray, QGate)):
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Check the input matrix
    matrix = getattr(U, "matrix", U)
    q = getattr(U, "target", (0, 1))

    if matrix.shape != (4, 4) or not is_unitary(matrix):
        raise ValueError("The input matrix must be a 4 x 4 unitary matrix.")

    # Decompose the matrix
    canonical_d = canonical_decomp(matrix)
    a_matrix = canonical_d.A
    b_matrix = canonical_d.B
    tx, ty, tz = canonical_d.t

    # Extract A1, A2, B1 and B2
    a1, a2 = kronecker_decomp(a_matrix)
    b1, b2 = kronecker_decomp(b_matrix)
    a2 = gates.S @ a2
    b1 = b1 @ gates.S.conj()

    # List of gates to return
    decomposition_circuit = [
        QGate.from_matrix(a1, name="A1", target=(q[0],)),
        QGate.from_matrix(a2, name="A2", target=(q[1],)),
        QGate.from_tuple(("CNOT1", (q[0], q[1]), 0), name="CNOT1"),
        QGate.from_matrix(gates.power_pauli_z(tz - 0.5), name="PZ", target=(q[0],)),
        QGate.from_matrix(gates.power_pauli_y(tx - 0.5), name="PY", target=(q[1],)),
        QGate.from_tuple(("CNOT", (q[0], q[1]), 0), name="CNOT"),
        QGate.from_matrix(gates.power_pauli_y(0.5 - ty), name="PY", target=(q[1],)),
        QGate.from_tuple(("CNOT1", (q[0], q[1]), 0), name="CNOT1"),
        QGate.from_matrix(b1, name="B1", target=(q[0],)),
        QGate.from_matrix(b2, name="B2", target=(q[1],)),
    ]

    return decomposition_circuit


def known_decomp(U: NDArray[np.floating] | QGate) -> list[QGate] | None:
    """Circuit decompositions of common 4 x 4 matrices.

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
    # Check input type
    if not isinstance(U, (np.ndarray, QGate)):
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Check the input matrix
    matrix = getattr(U, "matrix", U)
    q = getattr(U, "target", (0, 1))

    if matrix.shape != (4, 4) or not is_unitary(matrix):
        raise ValueError("The input matrix must be a 4 x 4 unitary matrix.")

    # Check if the matrix is a known gate
    if (matrix == np.eye(4)).all():  # Identity
        return []

    if (matrix == gates.CNOT).all():  # CNOT
        return [QGate.from_tuple(("CNOT", (q[0], q[1]), 0))]

    if (matrix == gates.CNOT1).all():  # CNOT (flipped)
        return [QGate.from_tuple(("CNOT1", (q[0], q[1]), 0))]

    if (matrix == gates.DCNOT).all():  # DCNOT (CNOT, then CNOT flipped)
        return common_decomp("DCNOT", q[0], q[1])

    if (matrix == gates.INV_DCNOT).all():  # INV_DCNOT (CNOT flipped, then CNOT)
        return common_decomp("INV_DCNOT", q[0], q[1])

    if (matrix == gates.SWAP).all():  # SWAP
        return common_decomp("SWAP", q[0], q[1])

    if (matrix == gates.ISWAP).all():  # ISWAP
        return common_decomp("ISWAP", q[0], q[1])

    if (matrix == gates.CY).all():  # Controlled Y
        return common_decomp("CY", q[0], q[1])

    if (matrix == gates.CZ).all():  # Controlled Z
        return common_decomp("CZ", q[0], q[1])

    if (matrix == gates.CH).all():  # Controlled Hadamard
        return common_decomp("CH", q[0], q[1])

    if (matrix == gates.MAGIC).all():  # Magic gate
        return common_decomp("MAGIC", q[0], q[1])

    if (matrix == gates.MAGIC.conj().T).all():  # Magic gate (Hermitian conjugate)
        return common_decomp("MAGIC_DAG", q[0], q[1])

    return None


def cnot_decomp(U: NDArray[np.floating]) -> list[QGate]:
    """Circuit decomposition of 4 x 4 quantum gates.

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

    **Example**:

    .. code-block:: python

        >>> from qdecomp.decompositions import cnot_decomp
        >>> from scipy.stats import unitary_group

        # Use an arbitrary 4 x 4 unitary matrix
        >>> U = unitary_group.rvs(4)

        # Decompose the matrix into a circuit of CNOT and single-qubit gates
        >>> circuit = cnot_decomp(U)
        >>> for gate in circuit:
        ...     print(f"{gate.target} -> {gate.name}")
        (0,) -> A1
        (1,) -> A2
        (0, 1) -> CNOT1
        (0,) -> PZ
        (1,) -> PY
        (0, 1) -> CNOT
        (1,) -> PY
        (0, 1) -> CNOT1
        (0,) -> B1
        (1,) -> B2
    """
    # Check input type
    if not isinstance(U, (np.ndarray, QGate)):
        raise TypeError(
            f"The input matrix must be a numpy array or a QGate object, but received {type(U).__name__}."
        )

    # Check the input matrix
    matrix = getattr(U, "matrix", U)

    if matrix.shape != (4, 4) or not is_unitary(matrix):
        raise ValueError("The input matrix must be a 4 x 4 unitary matrix.")

    # Check if the decomposition is known
    known_d = known_decomp(U)
    if known_d is not None:
        return known_d

    # Check the Lie group of the matrix and return the corresponding decomposition
    if is_orthogonal(matrix):
        if is_special(matrix):
            return so4_decomp(U)
        return o4_det_minus1_decomp(U)

    return u4_decomp(U)


def tqg_decomp(tqg: Union[np.ndarray, QGate], epsilon: float = 0.01) -> list[QGate]:
    """
    This function decomposes a two-qubit gate (TQG) into a sequence of CNOT and single qubit gates up to a given tolerance :math:`\\varepsilon`.
    It uses and combines the :mod:`qdecomp.decompositions.sqg` and :mod:`qdecomp.decompositions.cnot` decomposition algorithms to achieve this goal.

    Args:
        tqg (Union[np.array, QGate]): The matrix representation of the two-qubit gate to decompose.
        epsilon (float): The tolerance for the decomposition (default: 0.01).

    Returns:
        list[QGate]: A list of QGate objects representing the decomposed gates with their sequences defined.

    Raises:
        TypeError: If the input is not a numpy array or QGate object.
        ValueError: If the input is not a 4x4 matrix or QGate object with a 4x4 matrix.

    **Example**

    .. code-block:: python

        >>> from scipy.stats import unitary_group
        >>> from qdecomp.decompositions import tqg_decomp

        # Decompose a random two qubit gate with tolerance 0.001
        >>> tqg = unitary_group.rvs(4, random_state=42)
        >>> circuit = tqg_decomp(tqg, epsilon=0.001)
        >>> for gates in circuit:
        ...     print(f"{gates.target} -> {gates.sequence}")
        (0,) -> S T H T [...] H Z S T
        (1,) -> S T H T [...] S H S T
        (0, 1) -> CNOT1
        ...
        (1,) -> H T H S [...] T H Z S
    """

    if not isinstance(tqg, (np.ndarray, QGate)):
        raise TypeError(f"Input must be a numpy array or QGate object, got {type(tqg)}.")

    if not isinstance(tqg, QGate):
        tqg_matrix = tqg
        if tqg_matrix.shape != (4, 4):
            raise ValueError(f"Input gate must be a 4x4 matrix, got {tqg.shape}." + str(tqg.shape))
        tqg = QGate.from_matrix(matrix=tqg_matrix, target=(0, 1), epsilon=epsilon)

    if tqg.init_matrix.shape != (4, 4):
        raise ValueError(f"Input gate must be a 4x4 matrix, got {tqg.init_matrix.shape}.")

    cnot_decomp_lists = cnot_decomp(tqg.init_matrix)

    # Decompose each gate in the cnot decomposition list
    for cnot_decomp_qgate in cnot_decomp_lists:
        # if gate sequence is already initialized, skip decomposition
        if cnot_decomp_qgate.sequence is not None:
            continue

        # Else, perform the sqg decomposition
        cnot_qgate_seq, alpha = sqg_decomp(cnot_decomp_qgate.init_matrix, epsilon=epsilon)
        cnot_decomp_qgate.set_decomposition(cnot_qgate_seq, epsilon=epsilon)

    return cnot_decomp_lists
