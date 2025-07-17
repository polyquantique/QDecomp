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
#    limitations under the License.import pytest

"""Test the tqg decomposition function."""

import numpy as np
from scipy.stats import unitary_group
import pytest

from qdecomp.decompositions import tqg_decomp
from qdecomp.utils import QGate


np.random.seed(42)  # For reproducibility


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
        if gate.sequence_matrix.shape == (2, 2):
            if gate.target == (0,):
                M = np.kron(gate.sequence_matrix, np.eye(2)) @ M
            else:
                M = np.kron(np.eye(2), gate.sequence_matrix) @ M
        else:
            M = gate.sequence_matrix @ M
    return M


@pytest.mark.parametrize("trial", range(3))
@pytest.mark.parametrize("epsilon", [0.01, 0.001, 0.0001])
def test_tqg_decomposition_random_unitary(trial, epsilon):
    """Test the tqg_decomposition function with a random unitary matrix."""
    # Test the decomposition
    U = unitary_group.rvs(4, random_state=trial)
    decomposition = tqg_decomp(U, epsilon=epsilon)
    reconstructed = multiply_circuit(decomposition)

    # Assert the reconstructed matrix is equal to the original matrix
    if isinstance(U, QGate):
        U = U.init_matrix
    phase = reconstructed[0, 0] / U[0, 0]
    exact_reconstructed = reconstructed / phase

    # account for error propagation in the decomposition (10*epsilon)
    assert np.allclose(exact_reconstructed, U, atol=10 * epsilon)


def test_tqg_decomposition_identity():
    """Test the tqg_decomposition function with the identity matrix."""
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


def test_tqg_decomposition_invalid_input_type():
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


def test_tqg_decomposition_invalid_matrix_shape():
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
