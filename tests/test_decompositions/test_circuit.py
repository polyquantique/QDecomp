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

"""Test the `circuit_decomposition` function."""

import numpy as np
import pytest
from scipy.stats import unitary_group
from cliffordplust.decompositions.circuit import circuit_decomposition
from cliffordplust.circuit import QGate


def test_circuit_decomposition_single_qubit_gate_output_type():
    # Single-qubit gate (Pauli-X)
    gate = (np.array([[0, 1], [1, 0]]), (0,), 0.01)
    circuit = [gate]
    decomposed_circuit = circuit_decomposition(circuit)

    assert len(decomposed_circuit) == 1
    assert isinstance(decomposed_circuit[0], QGate)


def test_circuit_decomposition_two_qubit_gate_output_type():
    # Two-qubit gate (CNOT)
    gate = (np.eye(4)[[0, 2, 1, 3]], (0, 1), 0.01)
    circuit = [gate]
    decomposed_circuit = circuit_decomposition(circuit)

    assert len(decomposed_circuit) > 0
    for gate in decomposed_circuit:
        assert isinstance(gate, QGate)


def test_circuit_decomposition_random_unitary_output_type():
    # Random single-qubit unitary gate
    gate = (unitary_group.rvs(2), (0,), 0.001)
    circuit = [gate]
    decomposed_circuit = circuit_decomposition(circuit)

    assert len(decomposed_circuit) == 1
    assert isinstance(decomposed_circuit[0], QGate)
    assert decomposed_circuit[0].matrix_target == (0,)


def test_circuit_decomposition_invalid_input_type():
    # Invalid input type
    circuit = "invalid_input"
    with pytest.raises(TypeError, match="Input circuit must be a list"):
        circuit_decomposition(circuit)


def test_circuit_decomposition_invalid_matrix_size():
    # Invalid matrix size
    gate = (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (0,), 0.01)
    circuit = [gate]
    with pytest.raises(ValueError, match="The input matrix must have a size of"):
        circuit_decomposition(circuit)


def test_circuit_decomposition_invalid_target_qubits():
    # Invalid target qubits for single-qubit gate
    gate = (np.array([[0, 1], [1, 0]]), (0, 1), 0.01)
    circuit = [gate]
    with pytest.raises(ValueError, match="The input matrix must have a size of"):
        circuit_decomposition(circuit)


def test_circuit_decomposition_invalid_target_qubits_2():
    # Invalid target qubits for two-qubit gate
    gate = (np.eye(4)[[0, 1, 3, 2]], (0,), 0.01)
    circuit = [gate]
    with pytest.raises(ValueError, match="The input matrix must have a size of"):
        circuit_decomposition(circuit)


@pytest.mark.parametrize(
    "circuit",
    [
        [
            (np.eye(4)[[0, 1, 3, 2]], (0, 1), 0.01),
            (unitary_group.rvs(4), (1, 2), 0.001),
            (
                unitary_group.rvs(2),
                (0,),
                0.05,
            ),
        ]
    ],
)
def test_circuit_decomposition_error_tol(circuit):
    """Test if the result of the decomposition is the same as the input circuit."""
    decomposed_circuit = circuit_decomposition(circuit)
    for gate in decomposed_circuit:
        if gate.approx_matrix is not None:
            assert np.allclose(gate.matrix, gate.approx_matrix, atol=gate.epsilon)
