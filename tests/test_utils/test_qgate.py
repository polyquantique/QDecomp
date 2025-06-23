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

import numpy as np
import pytest
from qdecomp.utils import QGate


def test_from_matrix():
    """Test the QGate.from_matrix() method"""
    # Test a 1 qubit gate
    matrix = np.array([[0, 1], [1, 0]])
    name = "test_gate"
    gate = QGate.from_matrix(matrix=matrix, name=name)

    assert (gate.init_matrix == matrix).all()
    assert gate.name == name
    assert gate.target == (0,)

    # Test a 2 qubits gate
    matrix = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    name = "test_gate"
    gate = QGate.from_matrix(matrix=matrix, name=name, target=(2, 5))

    assert (gate.init_matrix == matrix).all()
    assert gate.name == name
    assert gate.target == (2, 5)


def test_from_matrix_error():
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
        QGate.from_matrix(matrix=matrix, target=(0, 1))


def test_from_sequence():
    """Test the QGate.from_sequence() method"""
    # Test a 1 qubit gate
    sequence = "H"
    name = "test_gate"
    gate = QGate.from_sequence(sequence=sequence, name=name)

    assert gate.sequence == sequence
    assert gate.name == name
    assert gate.target == (0,)

    # Test a 2 qubits gate
    sequence = "CNOT"
    name = "test_gate"
    gate = QGate.from_sequence(sequence=sequence, name=name, target=(2, 5))

    assert gate.sequence == sequence
    assert gate.name == name
    assert gate.target == (2, 5)


def test_qgate_init_error():
    """Test the QGate.__init__() errors"""
    # The target qubit is not a tuple
    target_list = [[1, 4], 3, "a"]
    for target in target_list:
        with pytest.raises(TypeError, match="The target qubit must be a tuple. Got "):
            QGate.from_sequence(sequence="H", target=target)

    # If the tuple elements are not integers
    target_list = [(1.0, 4), (3.0, 4), (1, "a"), (None,)]
    for target in target_list:
        with pytest.raises(TypeError, match="The target qubit must be a tuple of integers. Got "):
            QGate.from_sequence(sequence="H", target=target)

    # If the target qubits are not in ascending order
    target_list = [(2, 1), (4, 4), (1, 4, 2)]
    for target in target_list:
        with pytest.raises(ValueError, match="The target qubits must be in ascending order. Got "):
            QGate.from_sequence(sequence="CNOT", target=target)


@pytest.mark.parametrize("name", ["test_gate", None])
def test_name(name):
    """Test the QGate.name property"""
    gate = QGate(name=name)
    assert gate.name == name


@pytest.mark.parametrize("sequence", ["H", "CNOT", "S T"])
def test_sequence(sequence):
    """Test the QGate.sequence property"""
    gate = QGate.from_sequence(sequence=sequence)
    assert gate.sequence == sequence


@pytest.mark.parametrize(
    "gate, target",
    [
        ("H", (1,)),
        ("CNOT", (3,)),
    ],
)
def test_target(gate, target):
    """Test the QGate.target property"""
    gate = QGate.from_sequence(sequence=gate, target=target)
    assert gate.target == target


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(2),
        1.0j * np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[1, 0], [0, -1]]),
        np.array([[1, 0], [0, 1.0j]]),
        np.array([[0, -1.0j], [1.0j, 0]]),
    ],
)
def test_matrix(matrix):
    """Test the QGate.matrix property"""
    gate = QGate.from_matrix(matrix=matrix, target=(1,))

    assert (gate.matrix == gate.init_matrix).all()

    gate.set_decomposition("I", 0.01)
    assert (gate.matrix == gate.sequence_matrix).all()


@pytest.mark.parametrize(
    "sequence, target, matrix",
    [
        ("I", (1,), np.eye(2)),
        ("CNOT", (1, 3), np.eye(4)[[0, 1, 3, 2]]),
        ("H", (1,), np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
        ("S T", (1,), np.diag([1, np.exp(3.0j * np.pi / 4)])),
        (
            "CNOT1",
            (0, 3),
            np.eye(4)[[0, 3, 2, 1]],
        ),
        ("H H", (0,), np.eye(2)),
        ("T Tdag", (0,), np.eye(2)),
    ],
)
def test_seq_matrix__calculate_seq_matrix(sequence, target, matrix):
    """Test the QGate.matrix property and QGate.calculate_matrix() method"""
    gate = QGate.from_sequence(sequence=sequence, target=target)
    assert np.allclose(gate.sequence_matrix, matrix)


def test__calculate_seq_matrix_error():
    """Test the QGate.calculate_matrix() method with errors"""
    gate = QGate.from_sequence(sequence="H")
    with pytest.raises(ValueError, match="The sequence_matrix is already known."):
        gate._calculate_seq_matrix()  # The first call computes the sequence_matrix
        gate._calculate_seq_matrix()  # The second call raises an error

    gate = QGate.from_matrix(matrix=np.eye(2))
    with pytest.raises(ValueError, match="The sequence must be initialized."):
        gate._calculate_seq_matrix()

    gate = QGate.from_sequence(sequence="H CNOT T")
    with pytest.raises(
        ValueError,
        match="The sequence contains a gate that applies on the wrong number of qubits: CNOT",
    ):
        gate._calculate_seq_matrix()


@pytest.mark.parametrize(
    "gate, target, initializer, expected",
    [
        ("H", (1,), "sequence", 1),
        ("CNOT", (3, 5), "sequence", 2),
        (np.eye(2), (1,), "matrix", 1),
        (np.eye(4)[[0, 1, 3, 2]], (3, 4), "matrix", 2),
    ],
)
def test_num_qubits(gate, target, initializer, expected):
    """Test the QGate.num_qubits property"""
    if initializer == "sequence":
        gate = QGate.from_sequence(sequence=gate, target=target)
    else:
        gate = QGate.from_matrix(matrix=gate, target=target)

    assert gate.num_qubits == expected


def test_str():
    """Test the QGate.__str__() method"""
    gate = QGate()
    assert str(gate) == "Target: (0,)\n"

    gate = QGate(name="test_gate")
    assert str(gate) == "Gate: test_gate\nTarget: (0,)\n"

    gate = QGate.from_sequence(sequence="H", target=(1,))
    assert (
        str(gate)
        == """\
Sequence: H
Target: (1,)
"""
    )

    gate = QGate.from_sequence(sequence="CNOT", target=(0, 1))
    assert (
        str(gate)
        == """\
Sequence: CNOT
Target: (0, 1)
"""
    )

    gate = QGate.from_matrix(matrix=np.eye(2), target=(1,), epsilon=0.01)
    assert (
        str(gate)
        == """\
Target: (1,)
Epsilon: 0.01
Init. matrix:
[[1. 0.]
 [0. 1.]]
"""
    )

    gate = QGate.from_matrix(matrix=np.eye(4)[[0, 1, 3, 2]], target=(3, 4))
    assert (
        str(gate)
        == """\
Target: (3, 4)
Init. matrix:
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]]
"""
    )

    gate = QGate.from_sequence(sequence="H", target=(1,), name="H_gate")
    gate._calculate_seq_matrix()
    assert (
        str(gate)
        == """\
Gate: H_gate
Sequence: H
Target: (1,)
Seq. matrix:
[[ 0.70710678+0.j  0.70710678+0.j]
 [ 0.70710678+0.j -0.70710678+0.j]]
"""
    )


@pytest.mark.parametrize(
    "tup",
    [
        ("H", (1,), 0),
        ("CNOT", (1, 3), 0),
        ("CNOT1", (1, 3), 0),
        ("S T", (1,), 0),
        ("CNOT", (0, 1), 0),
        ("H", (0,), 0),
        (np.eye(2), (1,), 0),
        (np.eye(4)[[0, 1, 3, 2]], (3, 4), 0),
        (1, (1,), 0),
        (1, 2),
    ],
)
def test_to_and_from_tuple_with_errors(tup):
    """Test the QGate.to_tuple() and QGate.from_tuple() methods, and the to_tuple() errors"""
    if isinstance(tup[0], str):
        gate = QGate.from_tuple(tup)
        tup_2 = gate.to_tuple()
        assert tup == tup_2

    elif isinstance(tup[0], np.ndarray):
        gate = QGate.from_tuple(tup)
        assert gate.target == tup[1]
        assert (gate.init_matrix == tup[0]).all()

        with pytest.raises(
            ValueError, match="The sequence must be initialized to convert the gate to a tuple."
        ):
            gate.to_tuple()

    elif len(tup) != 3:
        with pytest.raises(ValueError, match="The tuple must contain three elements."):
            QGate.from_tuple(tup)

    else:
        with pytest.raises(
            TypeError, match="The first element of the tuple must be a string or a np.ndarray. Got "
        ):
            QGate.from_tuple(tup)


@pytest.mark.parametrize(
    "gate, decomposition, epsilon",
    [
        (QGate.from_matrix(matrix=np.eye(2)), "I", 0),
        (QGate.from_matrix(matrix=np.eye(4)[[0, 3, 2, 1]], target=(0, 1)), "CNOT1", 0),
        (QGate.from_matrix(np.array([[1, 1], [1, -1]]) / np.sqrt(2)), "H", 1e-10),
    ],
)
def test_set_decomposition(gate, decomposition, epsilon):
    """Test the QGate.set_decomposition() method"""
    gate.set_decomposition(decomposition, epsilon)

    assert gate.sequence == decomposition
    assert np.allclose(gate.sequence_matrix, gate.init_matrix)
    assert gate.epsilon == epsilon

    # Test if the sequence can be specified twice
    gate.set_decomposition(decomposition + " X X", epsilon)

    assert gate.sequence == decomposition + " X X"
    assert np.allclose(gate.sequence_matrix, gate.init_matrix)
    assert gate.epsilon == epsilon


def test_set_decomposition_errors():
    """Test the QGate.set_decomposition() method with errors"""
    # Epsilon not defined
    gate = QGate.from_matrix(np.eye(2))
    with pytest.raises(ValueError, match="The epsilon must be initialized."):
        gate.set_decomposition("I")


@pytest.mark.parametrize(
    "sequence, matrix",
    [
        ("I", np.eye(2)),
        ("", np.eye(2)),
        ("X", np.array([[0, 1], [1, 0]])),
        ("Y", np.array([[0, -1.0j], [1.0j, 0]])),
        ("Z", np.array([[1, 0], [0, -1]])),
        ("H", np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
        ("S", np.array([[1, 0], [0, 1.0j]])),
        ("SDAG", np.diag([1, -1.0j])),
        ("Sdag", np.diag([1, -1.0j])),
        ("T", np.diag([1, np.exp(1.0j * np.pi / 4)])),
        ("Tdag", np.diag([1, np.exp(-1.0j * np.pi / 4)])),
        ("TDAG", np.diag([1, np.exp(-1.0j * np.pi / 4)])),
        ("CNOT", np.eye(4)[[0, 1, 3, 2]]),
        ("CNOT1", np.eye(4)[[0, 3, 2, 1]]),
        ("CX", np.eye(4)[[0, 1, 3, 2]]),
        ("CX1", np.eye(4)[[0, 3, 2, 1]]),
        ("SWAP", np.eye(4)[[0, 2, 1, 3]]),
        ("H H", np.eye(2)),
        ("H H ", np.eye(2)),
        ("H H  ", np.eye(2)),
        ("  H H  ", np.eye(2)),
        ("T S Z", np.diag([1, np.exp(-1.0j * np.pi / 4)])),
        ("X Y", np.diag([-1j, 1j])),
        ("SWAP SWAP", np.eye(4)),
        ("W I", (1 + 1.0j) / np.sqrt(2) * np.eye(2)),
        ("SWAP W_dag SWAP", (1 - 1.0j) / np.sqrt(2) * np.eye(4)),
    ],
)
def test_calculate_seq_matrix(sequence, matrix):
    """Test the QGate.calculate_matrix() and QGate.get_simple_matrix() methods"""
    if matrix.shape[0] == 2:
        gate = QGate.from_sequence(sequence=sequence, target=(0,))
    else:
        assert matrix.shape[0] == 4
        gate = QGate.from_sequence(sequence=sequence, target=(0, 1))

    assert np.allclose(gate.sequence_matrix, matrix)
