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

"""Test the cnot module to perform 2-qubit gate decompositions."""

import math

import numpy as np
import pytest
from cliffordplust.circuit import QGate
from cliffordplust.decompositions import (Circuit, Gates,
                                          canonical_decomposition,
                                          cnot_decomposition, is_hermitian,
                                          is_orthogonal, is_special,
                                          is_unitary, known_decomposition,
                                          kronecker_decomposition,
                                          o4_det_minus1_decomposition,
                                          so4_decomposition, u4_decomposition)
from scipy.stats import ortho_group, special_ortho_group, unitary_group


def multiply_circuit(circuit: list[QGate]) -> np.ndarray:
    """
    Multiply a list of QGates objects to get the matrix representation of the circuit.

    Args:
        circuit (list[QGate]): The list of gates in the circuit.

    Returns:
        np.ndarray: The matrix representation of the circuit.
    """
    M = np.eye(4)
    for gate in circuit:
        if gate.matrix.shape == (2, 2):
            if gate.matrix_target == (0,):
                M = np.kron(gate.matrix, np.eye(2)) @ M
            else:
                M = np.kron(np.eye(2), gate.matrix) @ M
        else:
            M = gate.matrix @ M
    return M


@pytest.mark.parametrize(
    "matrix, result",
    [
        (np.eye(2), True),
        (-np.eye(3), False),
        (np.zeros((2, 2)), False),
        (np.array([[1.2, 3.4], [5, -1]]), False),
        (np.array([[0, 1], [1, 0]]), False),
        (np.array([[0, 1], [-1, 0]]), True),
        (special_ortho_group(dim=2, seed=42).rvs(), True),
    ],
)
def test_is_special(matrix, result):
    """Test the is_special function."""
    assert is_special(matrix) == result


@pytest.mark.parametrize(
    "matrix, result",
    [
        (-np.eye(2), True),
        (-np.eye(3), True),
        (np.zeros((2, 2)), False),
        (np.ones((2, 2)), False),
        (np.array([[1.2, 3.4], [5, -1]]), False),
        (np.array([[0, 1], [-1, 0]]), True),
        (ortho_group(dim=2, seed=42).rvs(), True),
    ],
)
def test_is_orthogonal(matrix, result):
    """Test the is_orthogonal function."""
    assert is_orthogonal(matrix) == result


@pytest.mark.parametrize(
    "matrix, result",
    [
        (-np.eye(2), True),
        (-np.eye(3), True),
        (np.zeros((2, 2)), False),
        (np.ones((2, 2)), False),
        (np.array([[1, -2], [-3, 1]]), False),
        (np.array([[0, 1], [-1, 0]]), True),
        (unitary_group(dim=2, seed=42).rvs(), True),
    ],
)
def test_is_unitary(matrix, result):
    """Test the is_unitary function."""
    assert is_unitary(matrix) == result


@pytest.mark.parametrize(
    "A",
    [
        np.zeros((2, 2)),
        np.identity(2),
        np.arange(1, 5).reshape(2, 2),
        np.ones((2, 2)),
        np.array([[1.2, 3.4], [5, -1]]),
        np.array([[np.pi, np.e], [np.log(2), np.sqrt(3)]]),
        np.array([[1, 1], [1, -1]]) / math.sqrt(2),
    ],
)
@pytest.mark.parametrize(
    "B",
    [
        np.zeros((2, 2)),
        np.identity(2),
        np.arange(1, 5).reshape(2, 2),
        np.ones((2, 2)),
        np.array([[1.2, 3.4], [5, -1]]),
        np.array([[np.pi, np.e], [np.log(2), np.sqrt(3)]]),
        np.array([[1, 1], [1, -1]]) / math.sqrt(2),
    ],
)
def test_kronecker_decomposition(A, B):
    """Test the kronecker_decomposition function."""
    M = np.kron(A, B)
    a, b = kronecker_decomposition(M)
    assert np.allclose(M, np.kron(a, b))


def test_kronecker_decomposition_errors():
    """Test the raise of errors when calling kronecker decomposition with wrong arguments."""

    # TypeError: The input matrix must be a numpy object
    for M in [((1, 2), (3, 4)), [[1, 2], [3, 4]], 1, "1", 1.0]:
        with pytest.raises(TypeError, match="The input matrix must be a numpy object, but got"):
            kronecker_decomposition(M)

    # ValueError: The input matrix must be 4 x 4
    for M in [
        np.ones((2, 2)),
        np.ones((3, 3)),
        np.ones((3, 4)),
        np.ones((4, 3)),
        np.ones((5, 5)),
    ]:
        with pytest.raises(ValueError, match="The input matrix must be 4 x 4"):
            kronecker_decomposition(M)


def test_so4_decomposition():
    """Test the decomposition of SO(4) matrices using the so4_decomposition function."""
    for i in range(10):
        # Use a predefined or randomly generated 4x4 matrix
        match i:
            case 0:
                U = np.eye(4)
            case 1:
                U = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            case 2:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
            case 3:
                U = QGate.from_matrix(
                    np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0]]),
                    matrix_target=(0, 1),
                )
            case _:
                U = special_ortho_group(dim=4, seed=i).rvs()

        # Decompose the matrix
        decomposition = so4_decomposition(U)
        reconstructed = multiply_circuit(decomposition)

        # Assert the reconstructed matrix is equal to the original matrix
        if isinstance(U, QGate):
            U = U.matrix
        assert np.allclose(reconstructed, U, rtol=1e-8)


def test_so4_decomposition_errors():
    """Test the raise of errors when calling the so4_decomposition function with wrong arguments."""
    # ValueError: The input matrix is not 4 x 4.
    for U in (np.eye(3), QGate.from_matrix(np.eye(2), matrix_target=(0,))):
        with pytest.raises(
            ValueError, match="The input matrix must be a 4 x 4 special orthogonal matrix."
        ):
            so4_decomposition(U)

    # ValueError: The input matrix is not orthogonal.
    U = np.eye(4) * 1.1
    with pytest.raises(
        ValueError, match="The input matrix must be a 4 x 4 special orthogonal matrix."
    ):
        so4_decomposition(U)

    # ValueError: The input matrix is not special.
    U = np.diag([1, 1, 1, -1])
    with pytest.raises(
        ValueError, match="The input matrix must be a 4 x 4 special orthogonal matrix."
    ):
        so4_decomposition(U)

    # TypeError: The input matrix is not a numpy object or a QGate object.
    U = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    with pytest.raises(
        TypeError, match="The input matrix must be a numpy array or a QGate object, but received"
    ):
        so4_decomposition(U)


def test_o4_det_minus1_decomposition():
    """Test the decomposition of O(4) matrices with a determinant of -1 using the o4_det_minus1_decomposition function."""
    for i in range(20):
        # Use a predefined or randomly generated 4x4 matrix
        match i:
            case 0:
                U = np.diag([1, 1, 1, -1])
            case 1:
                U = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            case 2:
                U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, -1, 0]])
            case 3:
                U = QGate.from_matrix(
                    np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
                    matrix_target=(0, 1),
                )
            case 4:
                U = Gates.CNOT
            case 5:
                U = Gates.CZ
            case 6:
                U = Gates.SWAP
            case _:
                U = ortho_group(dim=4, seed=i).rvs()
                if np.isclose(np.linalg.det(U), 1):
                    U[:, -1] = -U[:, -1]  # To have a determinant of -1

        # Test the decomposition
        decomposition = o4_det_minus1_decomposition(U)
        reconstructed = multiply_circuit(decomposition)

        # Assert the reconstructed matrix is equal to the original matrix
        if isinstance(U, QGate):
            U = U.matrix
        assert np.allclose(reconstructed, U, rtol=1e-8)


def test_o4_det_minus1_decomposition_errors():
    """Test the raise of errors when calling the o4_det_minus1_decomposition function with wrong arguments."""
    # ValueError: The input matrix is not 4 x 4.
    for U in (np.eye(3), QGate.from_matrix(np.eye(2), matrix_target=(0,))):
        with pytest.raises(
            ValueError,
            match="The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1.",
        ):
            o4_det_minus1_decomposition(U)

    # ValueError: The input matrix is not orthogonal.
    U = np.eye(4) * 1.1
    with pytest.raises(
        ValueError,
        match="The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1.",
    ):
        o4_det_minus1_decomposition(U)

    # ValueError: The input matrix does not have a determinant of -1.
    U = np.diag([1, 1, 1, 1])
    with pytest.raises(
        ValueError,
        match="The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1.",
    ):
        o4_det_minus1_decomposition(U)

    # TypeError: The input matrix is not a numpy object or a QGate object.
    U = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    with pytest.raises(
        TypeError, match="The input matrix must be a numpy array or a QGate object, but received"
    ):
        o4_det_minus1_decomposition(U)


@pytest.mark.parametrize(
    "U",
    list(unitary_group(dim=4, seed=42).rvs(10))
    + list(ortho_group(dim=4, seed=42).rvs(10))
    + [
        Gates.SWAP,  # SWAP gate
        Gates.CNOT,  # CNOT gate
        Gates.CZ,  # CZ gate
        Gates.CY,  # CY gate
        Gates.CH,  # CH gate
        Gates.ISWAP,  # iSWAP gate
        Gates.DCNOT,  # DCNOT gate
        np.eye(4),  # Identity gate
        Gates.MAGIC,  # Magic gate
        np.kron(Gates.power_pauli_y(0.39451), Gates.power_pauli_z(-0.22)),  # Parametric gate
        Gates.canonical_gate(0.1, 0.2, 0.3),  # Canonical gate
    ],
)
def test_canonical_decomposition(U):
    """Test the canonical decomposition of 4 x 4 unitary matrix using the canonical_decomposition function."""

    # Perform the decomposition
    A, B, t, alpha = canonical_decomposition(U)

    # Assert the reconstructed matrix is equal to the original matrix
    assert np.allclose(
        U, B @ Gates.canonical_gate(t[0], t[1], t[2]) @ A * np.exp(1.0j * alpha), rtol=1e-8
    )

    # Assert that the matrices A and B have an exact Kroncker decomposition
    a, b = kronecker_decomposition(A)
    alpha, beta = kronecker_decomposition(B)
    assert np.allclose(A, np.kron(a, b)) and np.allclose(B, np.kron(alpha, beta))


def test_canonical_decomposition_errors():
    """Test the raise of errors when calling canonical_decomposition function with wrong arguments."""

    # TypeError: The input matrix must be a numpy object
    for U in [((1, 2), (3, 4)), [[1, 2], [3, 4]]]:
        with pytest.raises(TypeError, match="Matrix U must be a numpy object"):
            canonical_decomposition(U)

    # ValueError: The input matrix must be a 4 x 4 unitary matrix
    for U in [
        np.ones((2, 2)),
        np.ones((3, 3)),
        np.ones((3, 4)),
        np.ones((4, 3)),
    ]:
        with pytest.raises(ValueError, match="U must be a 4 x 4 matrix but has shape"):
            canonical_decomposition(U)

    # ValueError: The input matrix must be a unitary matrix
    with pytest.raises(ValueError, match="U must be a unitary matrix."):
        canonical_decomposition(np.eye(4) * 1.1)


def test_u4_decomposition():
    """Test the decomposition of U(4) matrices using the u4_decomposition function."""
    for i in range(20):
        # Use a predefined or randomly generated 4 x 4 matrix
        match i:
            case 0:
                U = np.diag([1, 1, 1, -1])
            case 1:
                U = np.eye(4)
            case 2:
                U = Gates.CNOT
            case 3:
                U = Gates.INV_DCNOT
            case 4:
                U = Gates.ISWAP
            case 5:
                U = QGate.from_matrix(Gates.MAGIC, matrix_target=(0, 1))
            case 6:
                U = np.kron(Gates.T, Gates.power_pauli_y(0.39451))
            case 7:
                U = Gates.canonical_gate(0.1, 0.2, 0.3)
            case _:  # Randomly generated unitary matrix
                U = unitary_group(dim=4, seed=i).rvs()

        # Test the decomposition
        decomposition = u4_decomposition(U)
        reconstructed = multiply_circuit(decomposition)

        # Assert the reconstructed matrix is equal to the original matrix
        if isinstance(U, QGate):
            U = U.matrix
        phase = reconstructed[0, 0] / U[0, 0]
        assert np.allclose(reconstructed / phase, U, rtol=1e-8)


def test_u4_decomposition_errors():
    """Test the raise of errors when calling the u4_decomposition function with wrong arguments."""

    # ValueError: The input matrix is not 4 x 4.
    for U in (np.eye(3), QGate.from_matrix(np.eye(2), matrix_target=(0,))):
        with pytest.raises(ValueError, match="The input matrix must be a 4 x 4 unitary matrix."):
            u4_decomposition(U)

    # ValueError: The input matrix is not unitary.
    U = np.eye(4) * 1.1
    with pytest.raises(ValueError, match="The input matrix must be a 4 x 4 unitary matrix."):
        u4_decomposition(U)

    # TypeError: The input matrix is not a numpy object or a QGate object.
    U = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    with pytest.raises(
        TypeError, match="The input matrix must be a numpy array or a QGate object, but received"
    ):
        u4_decomposition(U)


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (np.eye(4), []),
        (Gates.CNOT, [QGate.from_tuple(("CNOT", (0, 1), 0))]),
        (Gates.CNOT1, [QGate.from_tuple(("CNOT", (1, 0), 0))]),
        (Gates.DCNOT, Circuit.decompositions("DCNOT", 0, 1)),
        (Gates.INV_DCNOT, Circuit.decompositions("INV_DCNOT", 0, 1)),
        (
            QGate.from_matrix(Gates.ISWAP, matrix_target=(0, 1)),
            Circuit.decompositions("ISWAP", 0, 1),
        ),
        (Gates.MAGIC, Circuit.decompositions("MAGIC", 0, 1)),
        (Gates.SWAP, Circuit.decompositions("SWAP", 0, 1)),
        (Gates.CZ, Circuit.decompositions("CZ", 0, 1)),
        (Gates.CY, Circuit.decompositions("CY", 0, 1)),
        (Gates.CH, Circuit.decompositions("CH", 0, 1)),
        (Gates.MAGIC.conj().T, Circuit.decompositions("MAGIC_DAG", 0, 1)),
        (np.diag([1, 1, -1, -1]), None),
    ],
)
def test_known_decomposition(matrix, expected):
    """Test the known_decomposition function."""
    if expected is not None:
        # Assert the function finds a decomposition
        assert known_decomposition(matrix) is not None

        # Assert the decomposition is correct
        decomposition = known_decomposition(matrix)
        reconstructed = multiply_circuit(decomposition)
        if isinstance(matrix, QGate):
            matrix = matrix.matrix
        assert np.allclose(reconstructed, matrix)
        assert np.allclose(reconstructed, multiply_circuit(expected))

    else:
        assert known_decomposition(matrix) is None


def test_known_decomposition_errors():
    """Test the raise of errors when calling the known_decomposition function with wrong arguments."""

    # ValueError: The input matrix is not 4 x 4.
    for U in (np.eye(3), QGate.from_matrix(np.eye(2), matrix_target=(0,))):
        with pytest.raises(ValueError, match="The input matrix must be a 4 x 4 unitary matrix."):
            known_decomposition(U)

    # ValueError: The input matrix is not unitary.
    U = np.eye(4) * 1.1
    with pytest.raises(ValueError, match="The input matrix must be a 4 x 4 unitary matrix."):
        known_decomposition(U)

    # TypeError: The input matrix is not a numpy object or a QGate object.
    U = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    with pytest.raises(
        TypeError, match="The input matrix must be a numpy array or a QGate object, but received"
    ):
        known_decomposition(U)


def test_cnot_decomposition():
    """Test the two-qubits gate decomposition using the cnot_decomposition function."""
    for i in range(30):
        # Use a predefined or randomly generated 4 x 4 matrix
        match i:
            case 0:
                U = Gates.CZ
            case 1:
                U = Gates.CNOT
            case 2:
                U = Gates.CNOT1
            case 3:
                U = Gates.CH
            case 5:
                U = Gates.canonical_gate(2.1, 1.2, 0.3)
            case 6:
                U = np.kron(Gates.T, Gates.power_pauli_y(0.39451))
            case 7:
                U = np.eye(4)
            case 8:
                U = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            case 9:
                U = QGate.from_matrix(Gates.INV_DCNOT, matrix_target=(0, 1))
            case j if 10 <= j < 20:
                U = unitary_group(dim=4, seed=i).rvs()
            case j if 20 <= j < 25:
                U = ortho_group(dim=4, seed=i).rvs()
            case _:
                U = special_ortho_group(dim=4, seed=i).rvs()

        # Test the decomposition
        decomposition = cnot_decomposition(U)
        reconstructed = multiply_circuit(decomposition)

        # Assert the reconstructed matrix is equal to the original matrix
        if isinstance(U, QGate):
            U = U.matrix
        phase = reconstructed[0, 0] / U[0, 0]
        assert np.allclose(reconstructed / phase, U, rtol=1e-8)


def test_cnot_decomposition_errors():
    """Test the raise of errors when calling the cnot_decomposition function with wrong arguments."""
    # ValueError: The input matrix is not 4 x 4.
    for U in (np.eye(3), QGate.from_matrix(np.eye(2), matrix_target=(0,))):
        with pytest.raises(ValueError, match="The input matrix must be a 4 x 4 unitary matrix."):
            cnot_decomposition(U)

    # ValueError: The input matrix is not unitary.
    U = np.eye(4) * 1.1
    with pytest.raises(ValueError, match="The input matrix must be a 4 x 4 unitary matrix."):
        cnot_decomposition(U)

    # TypeError: The input matrix is not a numpy object or a QGate object.
    U = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    with pytest.raises(
        TypeError, match="The input matrix must be a numpy array or a QGate object, but received"
    ):
        cnot_decomposition(U)
