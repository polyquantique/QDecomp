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


def test_from_matrix():
    """Test the QGate.from_matrix() method"""
    # Test a 1 qubit gate
    matrix = np.array([[0, 1], [1, 0]])
    name = "test_gate"
    gate = QGate.from_matrix(matrix=matrix, name=name)

    assert (gate.matrix == matrix).all()
    assert gate.name == name
    assert gate.matrix_target == (0, )

    # Test a 2 qubits gate
    matrix = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    name = "test_gate"
    gate = QGate.from_matrix(matrix=matrix, name=name, matrix_target=(2, 5))

    assert (gate.matrix == matrix).all()
    assert gate.name == name
    assert gate.matrix_target == (2, 5)


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
        QGate.from_matrix(matrix=matrix, matrix_target=(0, 1))


def test_from_sequence():
    """Test the QGate.from_sequence() method"""
    # Test a 1 qubit gate
    sequence = "H"
    name = "test_gate"
    gate = QGate.from_sequence(sequence=sequence, name=name)

    assert gate.sequence == sequence
    assert gate.name == name
    assert gate.target == (0, )
    assert gate.control is None

    # Test a 2 qubits gate
    sequence = "CNOT"
    name = "test_gate"
    gate = QGate.from_sequence(sequence=sequence, name=name, target=(2, ), control=5)

    assert gate.sequence == sequence
    assert gate.name == name
    assert gate.target == (2, )
    assert gate.control == 5


def test_from_sequence_error():
    """Test the QGate.from_sequence() errors"""
    with pytest.raises(ValueError, match=
                       "The sequence must start with 'C' if the gate is controlled."):
        QGate.from_sequence(sequence="H", control=1)

    with pytest.raises(ValueError, match=
                       "The sequence must not start with 'C' if the gate is not controlled."):
        QGate.from_sequence(sequence="C")


def test_qgate_init_error():
    """Test the QGate.__init__() errors"""
    # The target qubit is not a tuple of integers
    target_list = [(1.2, ), 3, "a"]
    for target in target_list:
        with pytest.raises(TypeError, match="The target qubit must be a tuple of integers. Got "):
            QGate.from_sequence(sequence="H", target=target)
    
    # If the control qubit is the same as a target qubit
    with pytest.raises(ValueError, match="The control qubit "):
        QGate.from_sequence(sequence="CNOT", target=(2, ), control=2)


@pytest.mark.parametrize("name", ["test_gate", None])
def test_name(name):
    """Test the QGate.name property"""
    gate = QGate(name=name)
    assert gate.name == name


@pytest.mark.parametrize("sequence", ["H", "CNOT", "S T"])
def test_sequence(sequence):
    """Test the QGate.sequence property"""
    ctrl = 1 if sequence.startswith("C") else None
    gate = QGate.from_sequence(sequence=sequence, control=ctrl)
    assert gate.sequence == sequence


@pytest.mark.parametrize("gate, target, control", [
    ("H", (1, ), None),
    ("CNOT", (3, ), 1),
])
def test_target_control(gate, target, control):
    """Test the QGate.target and QGate.control properties"""
    gate = QGate.from_sequence(sequence=gate, control=control, target=target)
    assert gate.target == target
    assert gate.control == control


def test_target_control_error():
    """Test the QGate.target and QGate.control properties with errors"""
    gate = QGate.from_matrix(matrix=np.eye(2))

    with pytest.raises(ValueError, match=
                       "The sequence must be initialized to get the control qubit."):
        gate.control
    with pytest.raises(ValueError, match=
                       "The sequence must be initialized to get the target qubit."):
        gate.target


@pytest.mark.parametrize("gate, qubit_no", [
    (np.eye(2), (1, )),
    (np.eye(4)[[0, 1, 3, 2]], (3, 4)),
])
def test_matrix_target(gate, qubit_no):
    """Test the QGate.matrix_target property"""
    gate = QGate.from_matrix(matrix=gate, matrix_target=qubit_no)
    assert gate.matrix_target == qubit_no


@pytest.mark.parametrize("sequence, target, control, matrix, matrix_target", [
    ("I", (1, ), None, np.eye(2), (1, )),
    ("CNOT", (1, ), 0, np.eye(4)[[0, 1, 3, 2]], (0, 1)),
    ("H", (1, ), None, np.array([[1, 1], [1, -1]]) / np.sqrt(2), (1, )),
    ("S T", (1, ), None, np.diag([1, np.exp(3.j * np.pi / 4)]), (1, )),
    ("CNOT", (0, ), 1, np.eye(4)[[0, 3, 2, 1]], (0, 1)),
    ("H H", (0, ), None, np.eye(2), (0, )),
])
def test_matrix__calculate_matrix(sequence, target, control, matrix, matrix_target):
    """Test the QGate.matrix property and QGate.calculate_matrix() method"""
    gate = QGate.from_sequence(sequence=sequence, target=target, control=control)
    assert np.allclose(gate.matrix, matrix)
    assert gate.matrix_target == matrix_target


@pytest.mark.parametrize("gate", [
    QGate.from_sequence(sequence="H"),
    QGate.from_matrix(np.eye(2)),
])
def test_calculate_matrix_error(gate):
    """Test the QGate.calculate_matrix() method with errors"""
    with pytest.raises(ValueError, match="The matrix is already known."):
        gate.calculate_matrix()
        gate.calculate_matrix()


@pytest.mark.parametrize("gate, target, control, initializer, expected", [
    ("H", (1, ), None, "sequence", 1),
    ("CNOT", (3, ), 1, "sequence", 2),
    (np.eye(2), (1, ), None, "matrix", 1),
    (np.eye(4)[[0, 1, 3, 2]], (3, 4), None, "matrix", 2),
])
def test_nb_qubits(gate, target, control, initializer, expected):
    """Test the QGate.nb_qubits property"""
    if initializer == "sequence":
        gate = QGate.from_sequence(sequence=gate, target=target, control=control)
    else:
        gate = QGate.from_matrix(matrix=gate, matrix_target=target)

    assert gate.nb_qubits == expected


def test_str():
    """Test the QGate.__str__() method"""
    gate = QGate()
    assert str(gate) == ""

    gate = QGate(name="test_gate")
    assert str(gate) == "Gate: test_gate\n"

    gate = QGate.from_sequence(sequence="H", target=(1, ))
    assert str(gate) == """\
Sequence: H
Control: None
Target: (1,)
"""

    gate = QGate.from_sequence(sequence="CNOT", target=(1, ), control=0)
    assert str(gate) == """\
Sequence: CNOT
Control: 0
Target: (1,)
"""

    gate = QGate.from_matrix(matrix=np.eye(2), matrix_target=(1, ))
    assert str(gate) == """\
Matrix:
[[1. 0.]
 [0. 1.]]
Matrix target: (1,)
"""

    gate = QGate.from_matrix(matrix=np.eye(4)[[0, 1, 3, 2]], matrix_target=(3, 4))
    assert str(gate) == """\
Matrix:
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]]
Matrix target: (3, 4)
"""

    gate = QGate.from_sequence(sequence="H", target=(1, ), name="H_gate")
    gate.calculate_matrix()
    assert str(gate) == """\
Gate: H_gate
Sequence: H
Control: None
Target: (1,)
Matrix:
[[ 0.70710678  0.70710678]
 [ 0.70710678 -0.70710678]]
Matrix target: (1,)
"""


@pytest.mark.parametrize("tup", [
    ("H", (1, ), 0),
    ("CNOT", (3, 1), 0),
    ("S T", (1, ), 0),
    ("CNOT", (0, 1), 0),
    ("H", (0, ), 0),
    (np.eye(2), (1, ), 0),
    (np.eye(4)[[0, 1, 3, 2]], (3, 4), 0),
    (1, (1, ), 0),
    (1, 2),
])
def test_to_and_from_tuple_with_errors(tup):
    """Test the QGate.to_tuple() and QGate.from_tuple() methods, and the to_tuple() errors"""
    if isinstance(tup[0], str):
        gate = QGate.from_tuple(tup)
        tup_2 = gate.to_tuple()
        assert tup == tup_2

    elif isinstance(tup[0], np.ndarray):
        gate = QGate.from_tuple(tup)
        assert gate.matrix_target == tup[1]
        assert (gate.matrix == tup[0]).all()
        
        with pytest.raises(ValueError, match=
                           "The sequence must be initialized to convert the gate to a tuple."):
            gate.to_tuple()

    else:
        with pytest.raises(ValueError):
            QGate.from_tuple(tup)


@pytest.mark.parametrize("gate", [
    QGate.from_sequence(sequence="H"),
    QGate.from_sequence(sequence="CNOT", target=(1, ), control=0),
    QGate.from_matrix(matrix=np.eye(2)),
    QGate.from_matrix(matrix=np.eye(4)[[0, 1, 3, 2]], matrix_target=(3, 4)),
])
def test_convert(gate):
    """Test the QGate.convert() method"""
    def to_dict(gate):
        """Convert a gate to a dictionary"""
        dic = dict()
        for attr in ["name", "matrix", "matrix_target", "epsilon"]:
            dic[attr] = getattr(gate, attr)

        return dic

    converted_gate = gate.convert(to_dict)

    assert to_dict(gate) == converted_gate
    assert converted_gate is not None and converted_gate != dict()


@pytest.mark.parametrize("gate, decomposition, epsilon", [
    (QGate.from_matrix(matrix=np.eye(2)), "I", 0),
    (QGate.from_matrix(matrix=np.eye(4)[[0, 3, 2, 1]], matrix_target=(1, 0)), "CNOT", 0),
    (QGate.from_matrix(np.array([[1, 1], [1, -1]]) / np.sqrt(2)), "H", 1e-10),
])
def test_set_decomposition(gate, decomposition, epsilon):
    """Test the QGate.set_decomposition() method"""
    gate.set_decomposition(decomposition, epsilon)

    assert gate.sequence == decomposition
    assert gate.target == (0, )
    assert gate._matrix == None
    assert np.allclose(gate.matrix, gate.approx_matrix)
    assert gate.epsilon == epsilon
    assert isinstance(gate.approx_matrix_target, tuple)


def test_set_decomposition_error():
    """Test the QGate.set_decomposition() method with errors"""
    gate = QGate.from_sequence("H H")

    with pytest.raises(ValueError, match="The sequence is already initialized."):
        gate.set_decomposition("I", epsilon=0)    


@pytest.mark.parametrize("sequence, matrix, control", [
    ("I", np.eye(2), None),
    ("", np.eye(2), None),
    ("X", np.array([[0, 1], [1, 0]]), None),
    ("Y", np.array([[0, -1.j], [1.j, 0]]), None),
    ("Z", np.array([[1, 0], [0, -1]]), None),
    ("H", np.array([[1, 1], [1, -1]]) / np.sqrt(2), None),
    ("S", np.array([[1, 0], [0, 1.j]]), None),
    ("SDAG", np.diag([1, -1.j]), None),
    ("Sdag", np.diag([1, -1.j]), None),
    ("T", np.diag([1, np.exp(1.j * np.pi / 4)]), None),
    ("Tdag", np.diag([1, np.exp(-1.j * np.pi / 4)]), None),
    ("TDAG", np.diag([1, np.exp(-1.j * np.pi / 4)]), None),
    ("CNOT", np.eye(4)[[0, 1, 3, 2]], 0),
    ("CNOT", np.eye(4)[[0, 3, 2, 1]], 1),
    ("CX", np.eye(4)[[0, 1, 3, 2]], 0),
    ("CX", np.eye(4)[[0, 3, 2, 1]], 1),
    ("SWAP", np.eye(4)[[0, 2, 1, 3]], None),
    ("H H", np.eye(2), None),
    ("H H ", np.eye(2), None),
    ("H H  ", np.eye(2), None),
    ("  H H  ", np.eye(2), None),
    ("T S Z", np.diag([1, np.exp(-1.j * np.pi / 4)]), None),
    ("X Y", np.diag([-1j, 1j]), None),
    ("SWAP SWAP", np.eye(4), None),
])
def test_calculate_matrix(sequence, matrix, control):
    """Test the QGate.calculate_matrix() and QGate.get_simple_matrix() methods"""
    if sequence.startswith("SWAP"):
        gate = QGate.from_sequence(sequence=sequence, target=(0, 1))

    elif control is not None:
        target = (1, ) if control == 0 else (0, )
        gate = QGate.from_sequence(sequence=sequence, control=control, target=target)

    else:
        gate = QGate.from_sequence(sequence=sequence)

    assert np.allclose(gate.matrix, matrix)
    assert gate.matrix is not None


@pytest.mark.parametrize("sequence", ["A", "HTH", "cnot"])
def test_calculate_matrix_error(sequence):
    """Test the QGate.calculate_matrix() method with errors"""
    with pytest.raises(ValueError, match=f"Unknown gate {sequence}."):
        gate = QGate.from_sequence(sequence=sequence)
        gate.calculate_matrix()
