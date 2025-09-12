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

"""Test the module to perform two-qubit gate decompositions."""

import math

import numpy as np
import pytest
from scipy.stats import ortho_group, special_ortho_group, unitary_group

from qdecomp.decompositions.common_gate_decompositions import common_decomp
from qdecomp.decompositions.tqg import *
from qdecomp.utils import QGate, gates

np.random.seed(42)  # For reproducibility


def multiply_circuit(circuit: list[QGate]) -> np.ndarray:
    """Multiply a list of QGates objects to get the matrix representation of the circuit.

    The function returns a 4 x 4 matrix that represents the circuit formed by the gates in the list.

    Args:
        circuit (list[QGate]): The list of gates in the circuit.

    Returns:
        np.ndarray: The matrix representation of the circuit.
    """
    M = np.eye(4)
    for gate in circuit:
        matrix = gate.matrix
        if matrix.shape == (2, 2):
            if gate.target == (0,):
                M = np.kron(matrix, np.eye(2)) @ M
            else:
                M = np.kron(np.eye(2), matrix) @ M
        else:
            M = matrix @ M
    return M


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
def test_kronecker_decomp(A, B):
    """Test the kronecker_decomp function."""
    M = np.kron(A, B)
    a, b = kronecker_decomp(M)
    assert np.allclose(M, np.kron(a, b))


def test_kronecker_decomp_errors():
    """Test the raise of errors when calling kronecker decomposition with wrong arguments."""

    # TypeError: The input matrix must be a numpy object
    for M in [((1, 2), (3, 4)), [[1, 2], [3, 4]], 1, "1", 1.0]:
        with pytest.raises(TypeError, match="The input matrix must be a numpy object, but got"):
            kronecker_decomp(M)

    # ValueError: The input matrix must be 4 x 4
    for M in [
        np.ones((2, 2)),
        np.ones((3, 3)),
        np.ones((3, 4)),
        np.ones((4, 3)),
        np.ones((5, 5)),
    ]:
        with pytest.raises(ValueError, match="The input matrix must be 4 x 4"):
            kronecker_decomp(M)


@pytest.mark.parametrize(
    "U",
    [
        np.eye(4),
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]),
        QGate.from_matrix(
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0]]),
            target=(0, 1),
        ),
    ]
    + list(special_ortho_group(dim=4, seed=137).rvs(10)),
)
def test_so4_decomp(U):
    """Test the decomposition of SO(4) matrices using the so4_decomp function."""
    # Decompose the matrix
    decomposition = so4_decomp(U)
    reconstructed = multiply_circuit(decomposition)

    # Assert the reconstructed matrix is equal to the original matrix
    if isinstance(U, QGate):
        U = U.matrix
    assert np.allclose(reconstructed, U, rtol=1e-8)


@pytest.mark.parametrize(
    "error",
    [
        (np.eye(3), ValueError, "The input matrix must be a 4 x 4 special orthogonal matrix."),
        (
            QGate.from_matrix(np.eye(2), target=(0,)),
            ValueError,
            "The input matrix must be a 4 x 4 special orthogonal matrix.",
        ),
        (
            np.eye(4) * 1.1,
            ValueError,
            "The input matrix must be a 4 x 4 special orthogonal matrix.",
        ),
        (
            np.diag([1, 1, 1, -1]),
            ValueError,
            "The input matrix must be a 4 x 4 special orthogonal matrix.",
        ),
        (
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            TypeError,
            "The input matrix must be a numpy array or a QGate object, but received",
        ),
    ],
)
def test_so4_decomp_errors(error):
    """Test the raise of errors when calling the so4_decomp function with wrong arguments."""
    U, exc_type, exc_msg = error
    with pytest.raises(exc_type, match=exc_msg):
        so4_decomp(U)


@pytest.mark.parametrize(
    "U",
    [
        np.diag([1, 1, 1, -1]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, -1, 0]]),
        QGate.from_matrix(
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
            target=(0, 1),
        ),
        gates.CNOT,
        gates.CZ,
        gates.SWAP,
    ]
    + list(ortho_group(dim=4, seed=42).rvs(10)),
)
def test_o4_det_minus1_decomp(U):
    """Test the decomposition of O(4) matrices with a determinant of -1 using the o4_det_minus1_decomp function."""
    if isinstance(U, np.ndarray):
        if np.isclose(np.linalg.det(U), 1):
            U[:, -1] = -U[:, -1]  # To have a determinant of -1

    # Test the decomposition
    decomposition = o4_det_minus1_decomp(U)
    reconstructed = multiply_circuit(decomposition)

    # Assert the reconstructed matrix is equal to the original matrix
    if isinstance(U, QGate):
        U = U.matrix
    assert np.allclose(reconstructed, U, rtol=1e-8)


@pytest.mark.parametrize(
    "errors",
    [
        (
            np.eye(3),
            ValueError,
            "The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1.",
        ),
        (
            QGate.from_matrix(np.eye(2), target=(0,)),
            ValueError,
            "The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1.",
        ),
        (
            np.eye(4) * 1.1,
            ValueError,
            "The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1.",
        ),
        (
            np.diag([1, 1, 1, 1]),
            ValueError,
            "The input matrix must be a 4 x 4 orthogonal matrix with a determinant of -1.",
        ),
        (
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            TypeError,
            "The input matrix must be a numpy array or a QGate object, but received",
        ),
    ],
)
def test_o4_det_minus1_decomp_errors(errors):
    """Test the raise of errors when calling the o4_det_minus1_decomp function with wrong arguments."""
    U, exc_type, exc_msg = errors
    with pytest.raises(exc_type, match=exc_msg):
        o4_det_minus1_decomp(U)


@pytest.mark.parametrize(
    "U",
    list(unitary_group(dim=4, seed=42).rvs(10))
    + list(ortho_group(dim=4, seed=42).rvs(10))
    + [
        gates.SWAP,  # SWAP gate
        gates.CNOT,  # CNOT gate
        gates.CZ,  # CZ gate
        gates.CY,  # CY gate
        gates.CH,  # CH gate
        gates.ISWAP,  # iSWAP gate
        gates.DCNOT,  # DCNOT gate
        np.eye(4),  # Identity gate
        gates.MAGIC,  # Magic gate
        np.kron(gates.power_pauli_y(0.39451), gates.power_pauli_z(-0.22)),  # Parametric gate
        gates.canonical_gate(0.1, 0.2, 0.3),  # Canonical gate
    ],
)
def test_canonical_decomp(U):
    """Test the canonical decomposition of 4 x 4 unitary matrix using the canonical_decomp function."""

    # Perform the decomposition
    A, B, t, alpha = canonical_decomp(U)

    # Assert the reconstructed matrix is equal to the original matrix
    assert np.allclose(
        U, B @ gates.canonical_gate(t[0], t[1], t[2]) @ A * np.exp(1.0j * alpha), rtol=1e-8
    )

    # Assert that the matrices A and B have an exact Kroncker decomposition
    a, b = kronecker_decomp(A)
    alpha, beta = kronecker_decomp(B)
    assert np.allclose(A, np.kron(a, b)) and np.allclose(B, np.kron(alpha, beta))


@pytest.mark.parametrize(
    "errors",
    [
        (((1, 2), (3, 4)), TypeError, "Matrix U must be a numpy object"),
        ([[1, 2], [3, 4]], TypeError, "Matrix U must be a numpy object"),
        (np.ones((2, 2)), ValueError, "U must be a 4 x 4 matrix but has shape"),
        (np.ones((3, 3)), ValueError, "U must be a 4 x 4 matrix but has shape"),
        (np.ones((3, 4)), ValueError, "U must be a 4 x 4 matrix but has shape"),
        (np.ones((4, 3)), ValueError, "U must be a 4 x 4 matrix but has shape"),
        (np.eye(4) * 1.1, ValueError, "U must be a unitary matrix."),
    ],
)
def test_canonical_decomp_errors(errors):
    """Test the raise of errors when calling canonical_decomp function with wrong arguments."""

    U, exc_type, exc_msg = errors
    with pytest.raises(exc_type, match=exc_msg):
        canonical_decomp(U)


@pytest.mark.parametrize(
    "U",
    [
        np.diag([1, 1, 1, -1]),
        np.eye(4),
        gates.CNOT,
        gates.INV_DCNOT,
        gates.ISWAP,
        QGate.from_matrix(gates.MAGIC, target=(0, 1)),
        np.kron(gates.T, gates.power_pauli_y(0.39451)),
        gates.canonical_gate(0.1, 0.2, 0.3),
    ]
    + list(unitary_group(dim=4, seed=137).rvs(10)),
)
def test_u4_decomp(U):
    """Test the decomposition of U(4) matrices using the u4_decomp function."""
    # Test the decomposition
    decomposition = u4_decomp(U)
    reconstructed = multiply_circuit(decomposition)

    # Assert the reconstructed matrix is equal to the original matrix
    if isinstance(U, QGate):
        U = U.matrix
    phase = reconstructed[0, 0] / U[0, 0]
    assert np.allclose(reconstructed / phase, U, rtol=1e-8)


@pytest.mark.parametrize(
    "errors",
    [
        (np.eye(3), ValueError, "The input matrix must be a 4 x 4 unitary matrix."),
        (
            QGate.from_matrix(np.eye(2), target=(0,)),
            ValueError,
            "The input matrix must be a 4 x 4 unitary matrix.",
        ),
        (np.eye(4) * 1.1, ValueError, "The input matrix must be a 4 x 4 unitary matrix."),
        (
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            TypeError,
            "The input matrix must be a numpy array or a QGate object, but received",
        ),
    ],
)
def test_u4_decomp_errors(errors):
    """Test the raise of errors when calling the u4_decomp function with wrong arguments."""
    U, exc_type, exc_msg = errors
    with pytest.raises(exc_type, match=exc_msg):
        u4_decomp(U)


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (np.eye(4), []),
        (gates.CNOT, [QGate.from_tuple(("CNOT", (0, 1), 0))]),
        (gates.CNOT1, [QGate.from_tuple(("CNOT1", (0, 1), 0))]),
        (gates.DCNOT, common_decomp("DCNOT", 0, 1)),
        (gates.INV_DCNOT, common_decomp("INV_DCNOT", 0, 1)),
        (
            QGate.from_matrix(gates.ISWAP, target=(0, 1)),
            common_decomp("ISWAP", 0, 1),
        ),
        (gates.MAGIC, common_decomp("MAGIC", 0, 1)),
        (gates.SWAP, common_decomp("SWAP", 0, 1)),
        (gates.CZ, common_decomp("CZ", 0, 1)),
        (gates.CY, common_decomp("CY", 0, 1)),
        (gates.CH, common_decomp("CH", 0, 1)),
        (gates.MAGIC.conj().T, common_decomp("MAGIC_DAG", 0, 1)),
        (np.diag([1, 1, -1, -1]), None),
    ],
)
def test_known_decomp(matrix, expected):
    """Test the known_decomp function."""
    if expected is not None:
        # Assert the function finds a decomposition
        assert known_decomp(matrix) is not None

        # Assert the decomposition is correct
        decomposition = known_decomp(matrix)
        reconstructed = multiply_circuit(decomposition)
        if isinstance(matrix, QGate):
            matrix = matrix.matrix
        assert np.allclose(reconstructed, matrix)
        assert np.allclose(reconstructed, multiply_circuit(expected))

    else:
        assert known_decomp(matrix) is None


@pytest.mark.parametrize(
    "errors",
    [
        (np.eye(3), ValueError, "The input matrix must be a 4 x 4 unitary matrix."),
        (
            QGate.from_matrix(np.eye(2), target=(0,)),
            ValueError,
            "The input matrix must be a 4 x 4 unitary matrix.",
        ),
        (np.eye(4) * 1.1, ValueError, "The input matrix must be a 4 x 4 unitary matrix."),
        (
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            TypeError,
            "The input matrix must be a numpy array or a QGate object, but received",
        ),
    ],
)
def test_known_decomp_errors(errors):
    """Test the raise of errors when calling the known_decomp function with wrong arguments."""

    U, exc_type, exc_msg = errors
    with pytest.raises(exc_type, match=exc_msg):
        known_decomp(U)


@pytest.mark.parametrize(
    "U",
    [
        gates.CZ,
        gates.CNOT,
        gates.CNOT1,
        gates.CH,
        gates.canonical_gate(2.1, 1.2, 0.3),
        np.kron(gates.T, gates.power_pauli_y(0.39451)),
        np.eye(4),
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        QGate.from_matrix(gates.INV_DCNOT, target=(0, 1)),
        *list(unitary_group(dim=4, seed=42).rvs(10)),
        *list(ortho_group(dim=4, seed=42).rvs(10)),
        *list(special_ortho_group(dim=4, seed=42).rvs(10)),
    ],
)
def test_cnot_decomp(U):
    """Test the two-qubits gate decomposition using the cnot_decomp function."""

    # Test the decomposition
    decomposition = cnot_decomp(U)
    reconstructed = multiply_circuit(decomposition)

    # Assert the reconstructed matrix is equal to the original matrix
    if isinstance(U, QGate):
        U = U.matrix
    phase = reconstructed[0, 0] / U[0, 0]
    assert np.allclose(reconstructed / phase, U, rtol=1e-8)


@pytest.mark.parametrize(
    "errors",
    [
        (np.eye(3), ValueError, "The input matrix must be a 4 x 4 unitary matrix."),
        (
            QGate.from_matrix(np.eye(2), target=(0,)),
            ValueError,
            "The input matrix must be a 4 x 4 unitary matrix.",
        ),
        (np.eye(4) * 1.1, ValueError, "The input matrix must be a 4 x 4 unitary matrix."),
        (
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            TypeError,
            "The input matrix must be a numpy array or a QGate object, but received",
        ),
    ],
)
def test_cnot_decomp_errors(errors):
    """Test the raise of errors when calling the cnot_decomp function with wrong arguments."""
    U, exc_type, exc_msg = errors
    with pytest.raises(exc_type, match=exc_msg):
        cnot_decomp(U)


@pytest.mark.parametrize("trial", range(3))
@pytest.mark.parametrize("epsilon", [0.01, 0.001, 0.0001])
def test_tqg_decomp_random_unitary(trial, epsilon):
    """Test the tqg_decomp function with a random unitary matrix."""
    # Test the decomposition
    U = unitary_group.rvs(4, random_state=trial)
    decomposition = tqg_decomp(U, epsilon=epsilon)
    reconstructed = multiply_circuit(decomposition)

    # Assert the reconstructed matrix is equal to the original matrix
    if isinstance(U, QGate):
        U = U.init_matrix
    phase = reconstructed[0, 0] / U[0, 0]
    exact_reconstructed = reconstructed / phase

    # account for error propagation in the decomposition (15*epsilon)
    assert np.allclose(exact_reconstructed, U, atol=15 * epsilon)


def test_tqg_decomp_identity():
    """Test the tqg_decomp function with the identity matrix."""
    # Test with numpy identity matrix
    identity = np.eye(4)
    decomposition = tqg_decomp(identity, epsilon=0.01)
    reconstructed = multiply_circuit(decomposition)

    phase = reconstructed[0, 0] / identity[0, 0]
    exact_reconstructed = reconstructed / phase
    assert np.allclose(exact_reconstructed, identity, atol=0.01)

    # Test with QGate identity matrix
    identity_qgate = QGate.from_matrix(np.eye(4), target=(0, 1))
    decomposition_qgate = tqg_decomp(identity_qgate, epsilon=0.01)
    reconstructed_qgate = multiply_circuit(decomposition_qgate)

    phase_qgate = reconstructed_qgate[0, 0] / identity[0, 0]
    exact_reconstructed_qgate = reconstructed_qgate / phase_qgate
    assert np.allclose(exact_reconstructed_qgate, identity, atol=0.01)


def test_tqg_decomp_invalid_input_type():
    """Test that tqg_decomp raises ValueError for invalid input types."""
    # Test with string
    with pytest.raises(TypeError, match="Input must be a numpy array or QGate object"):
        tqg_decomp("invalid_input", epsilon=0.01)

    # Test with list
    with pytest.raises(TypeError, match="Input must be a numpy array or QGate object"):
        tqg_decomp([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], epsilon=0.01)

    # Test with integer
    with pytest.raises(TypeError, match="Input must be a numpy array or QGate object"):
        tqg_decomp(42, epsilon=0.01)

    # Test with None
    with pytest.raises(TypeError, match="Input must be a numpy array or QGate object"):
        tqg_decomp(None, epsilon=0.01)


def test_tqg_decomp_invalid_matrix_shape():
    """Test that tqg_decomp raises ValueError for invalid matrix shapes."""
    # Test with 2x2 matrix
    matrix_2x2 = np.array([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="Input gate must be a 4x4 matrix"):
        tqg_decomp(matrix_2x2, epsilon=0.01)

    # Test with non-square matrix
    matrix_nonsquare = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    with pytest.raises(ValueError, match="Input gate must be a 4x4 matrix"):
        tqg_decomp(matrix_nonsquare, epsilon=0.01)

    # Test with QGate object
    qgate_invalid = QGate.from_matrix(np.array([[1, 0], [0, 1]]), target=(0,))
    with pytest.raises(ValueError, match="Input gate must be a 4x4 matrix"):
        tqg_decomp(qgate_invalid, epsilon=0.01)
