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

import pytest
import numpy as np

from cliffordplust.circuit import QGate


def test_qgate_from_matrix():
    """Test the QGate.from_matrix() method"""
    # Test a 1 qubit gate
    matrix = np.array([[0, 1], [1, 0]])
    name = "test_gate"
    gate = QGate.from_matrix(matrix=matrix, name=name)

    assert (gate.matrix == matrix).all()
    assert gate.name == name
    assert gate.target == (0,)
    assert gate.control is None
    assert gate.qubit_no == (0, )

    # Test a 2 qubits gate
    matrix = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    name = "test_gate"
    gate = QGate.from_matrix(matrix=matrix, name=name, qubit_no=(2, 5))

    assert (gate.matrix == matrix).all()
    assert gate.name == name
    assert gate.target == (2, 5)
    assert gate.control is None
    assert gate.qubit_no == (2, 5)

def test_qgate_from_matrix_error():
    """Test the QGate.from_matrix() method with errors"""
    # 2D error
    matrix = np.array([0, 1])
    with pytest.raises(ValueError, match="The input matrix must be a 2D matrix. Got "):
        QGate.from_matrix(matrix=matrix)

    # Square matrix error
    matrix = np.array([[0, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError, match="The input matrix must be a square matrix. Got shape "):
        QGate.from_matrix(matrix=matrix)

    # Unitary matrix error
    matrix = np.array([[0, 1], [1, 1]])
    with pytest.raises(ValueError, match="The input matrix must be unitary."):
        QGate.from_matrix(matrix=matrix)

    # The size of the matrix does not match the number of qubits
    matrix = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="The input matrix must have a size of 2"):
        QGate.from_matrix(matrix=matrix, qubit_no=(0, 1))

def test_qgate_from_sequence():
    """Test the QGate.from_sequence() method"""
    # Test a 1 qubit gate
    sequence = "H"
    name = "test_gate"
    gate = QGate.from_sequence(sequence=sequence, name=name)

    assert gate.sequence == sequence
    assert gate.name == name
    print(gate.target)
    assert gate.target == (0, )
    assert gate.control is None
    assert gate.qubit_no == (0, )

    # Test a 2 qubits gate
    sequence = "CNOT"
    name = "test_gate"
    gate = QGate.from_sequence(sequence=sequence, name=name, target=5, control=2)

    assert gate.sequence == sequence
    assert gate.name == name
    assert gate.target == (2, )
    assert gate.control == 5
    assert gate.qubit_no == (5, 2)

test_qgate_from_sequence()