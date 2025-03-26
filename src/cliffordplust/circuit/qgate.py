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
This module contains the QGate class, which represents a quantum gate. It provides methods to easily
create a gate from different representations (matrix, sequence, tuple) and to stock its information
in an intuitive way. The class also contains methods to stock the decomposition of the gate, its
initial approximation and its error. Finally, the class provides a method to calculate the matrix
representation of the gate from its sequence.

Classes:
    - QGate: Class representing a quantum gate.
"""

from typing import Any, Callable

import numpy as np


class QGate:
    r"""
    Class representing a quantum gate. It provides the user the ability to easily create and
    manipulate quantum gates. The gates can be defined either by their matrix, their sequence or a
    tuple. A Gate object contains the information about the qubit on which it acts, its control bit,
    if any, and its error if a sequence is used to approximate it. The class also provides a method
    to calculate the matrix representation of the gate from its sequence.

    Attributes:
        name (str or None): Name of the gate.
        nb_qubits (int): Number of qubits on which the gate applies.

        sequence (str or None): Sequence associated with the gate decomposition.
        control (int or None): Number of the control qubit.
        target (int or tuple[int] or None): Number of the target qubit.

        matrix (np.ndarray or None): Matrix representation of the gate.
        matrix_target (tuple[int] or None): Qubits on which the gate applies.

        approx_matrix (np.ndarray or None): Approximated matrix representation of the gate.
        approx_matrix_target (tuple[int] or None): Qubits on which the approximated matrix applies.
        epsilon (float): Tolerance for the gate.

    Class methods:
        from_matrix: Create a QGate object from a matrix.
        from_sequence: Create a QGate object from a sequence.
        from_tuple: Create a QGate object from a tuple.

    Methods:
        __str__: Convert the gate to a string representation.
        to_tuple: Convert the gate to a tuple representation.
        convert: Convert the gate by using a user-defined function.
        set_decomposition: Set the decomposition of the gate. The decomposition might be an
            approximation of the initial gate stored in the approx_matrix attribute.
        calculate_matrix: Calculate the matrix representation of the gate from its sequence.

    Static methods:
        get_simple_matrix: Get the matrix representation of a simple gate.

    **Example**
    .. code-block:: python
        # Import the necessary library and module
        import numpy as np

        from cliffordplust.circuit import QGate

        # Create a gate from a sequence
        g1 = QGate.from_sequence(sequence="CNOT", target=(3, ), control=1)
        print(g1)
        # Sequence: CNOT
        # Control: 1
        # Target: (3,)

        # Create a gate from a matrix
        g2 = QGate.from_matrix(matrix=np.diag([1, -1]), name="My_Z_Gate")
        print(g2)
        # Gate: My_Z_Gate
        # Matrix:
        # [[ 1  0]
        #  [ 0 -1]]
        # Matrix target: (0,)

        # Create a gate from a tuple
        g3 = QGate.from_tuple(("H", (0, ), 0))
        print(g3)
        # Sequence: H
        # Control: None
        # Target: (0,)

        # Get the matrix of a from_sequence() gate
        gate = QGate.from_sequence(sequence="X Z Y", name="my_gate")
        print("Name:", gate.name)          # Name: my_gate
        print("Sequence:", gate.sequence)  # Sequence: X Z Y
        print("Target:", gate.target)      # Target: (0,)
        print("Matrix:\n", gate.matrix)
        # Matrix:
        #  [[ 0.+1.j  0.+0.j]
        #  [-0.+0.j  0.+1.j]]
        print()

        # Set the approximation sequence of a gate
        approx_gate = QGate.from_matrix(matrix=np.diag([1-0.001j, 1.j+0.001]), name="approx_my_gate")
        approx_gate.set_decomposition("S", epsilon=0.01)
        print("Name:", approx_gate.name)          # Name: approx_my_gate
        print("Sequence:", approx_gate.sequence)  # Sequence: S
        print("Initial matrix:\n", approx_gate.approx_matrix)
        # Initial matrix:
        #  [[1.   -0.001j 0.   +0.j   ]
        #  [0.   +0.j    0.001+1.j   ]]
        print("Approximated matrix:\n", approx_gate.matrix)
        # Approximated matrix:
        #  [[1.+0.j 0.+0.j]
        #  [0.+0.j 0.+1.j]]
    """

    def __init__(
        self,
        name: str | None = None,
    ) -> None:
        """
        Initialize the QGate object.

        Args:
            name (str or None): Name of the gate
        """
        # Populate the attributes
        self._name = name

        self._sequence = None
        self._target = None
        self._control = None

        self._matrix = None
        self._matrix_target = None

        self._approx_matrix = None
        self._approx_matrix_target = None
        self._epsilon = None

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        name: str | None = None,
        matrix_target: tuple[int] = (0,),
    ) -> "QGate":
        """
        Create a QGate object from a matrix.

        Args:
            matrix (np.ndarray): Matrix representation of the gate.
            name (str or None): Name of the gate.
            matrix_target (tuple[int]): Qubits on which the gate applies.

        Returns:
            QGate: The QGate object.

        Raises:
            ValueError: If the matrix doesn't have 2 dimensions.
            ValueError: If the matrix is not a square matrix.
            ValueError: If the matrix is not unitary.
            ValueError: If the number of rows of the matrix is not 2^nb_of_qubits.

        **Example**
        .. code-block:: python
            import numpy as np
            from cliffordplust.circuit import QGate

            gate = QGate.from_matrix(np.diag([1, 1j]), name="my_S_gate", matrix_target=(1, ))
            print(gate)
            # Gate: my_S_gate
            # Matrix:
            # [[1.+0.j 0.+0.j]
            #  [0.+0.j 0.+1.j]]
            # Matrix target: (1,)
        """
        # Convert the matrix to a numpy array
        matrix = np.asarray(matrix)

        # 2D matrix
        if matrix.ndim != 2:
            raise ValueError(f"The input matrix must be a 2D matrix. Got {matrix.ndim} dimensions.")

        # Square matrix
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"The input matrix must be a square matrix. Got shape {matrix.shape}.")

        # Unitary matrix
        if not np.allclose(np.eye(matrix.shape[0]), matrix @ matrix.conj().T):
            raise ValueError("The input matrix must be unitary.")

        # Size of the matrix compared to the number of targets and control
        if matrix.shape[0] != 2 ** len(matrix_target):
            raise ValueError(
                "The input matrix must have a size of 2^nb_of_qubit. Got shape "
                + f"{matrix.shape} and {len(matrix_target)} qubit(s)."
            )

        # Create the gate
        gate = cls(name=name)
        gate._matrix = matrix
        gate._matrix_target = matrix_target

        return gate

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        name: str | None = None,
        target: tuple[int] = (0,),
        control: int | None = None,
    ) -> "QGate":
        """
        Create a QGate object from a sequence. If a controlled gate is used, the sequence must start
        with "C".

        Args:
            sequence (str): Sequence associated with the gate decomposition.
            target (tuple[int]): Qubits on which the gate applies.
            control (int or None): Control qubit.

        Returns:
            QGate: The QGate object.

        Raises:
            TypeError: If the target qubit is not a tuple of integers.
            ValueError: If a target qubit is the same as the control qubit.
            ValueError: If the control qubit is not None and the sequence does not start with "C".
            ValueError: If the control qubit is None, but the sequence starts with "C".

        **Example**
        .. code-block:: python
            from cliffordplust.circuit import QGate

            h_gate = QGate.from_sequence("H", name="my_H_gate", target=(2, ))
            print(h_gate)
            # Gate: my_H_gate
            # Sequence: H
            # Control: None
            # Target: (2,)

            cx_gate = QGate.from_sequence("CNOT", name="my_CNOT_gate", control=(3, ), target=(1, ))
            print(cx_gate)
            # Gate: my_CNOT_gate
            # Sequence: CNOT
            # Control: (3,)
            # Target: (1,)
        """
        # Check the input arguments
        # Target qubit is not a tuple of integers
        if not isinstance(target, tuple) or not all(isinstance(q, int) for q in target):
            raise TypeError(f"The target qubit must be a tuple of integers. Got {target}.")

        # Control qubit different from target qubit
        if control in target:
            raise ValueError(
                f"The control qubit ({control}) must be different from the target qubits {target}."
            )

        # Check if the control qubit type (None or int) corresponds to the gate type
        if control is not None and not sequence.startswith("C"):
            raise ValueError("The sequence must start with 'C' if the gate is controlled.")

        if control is None and sequence.startswith("C"):
            raise ValueError("The sequence must not start with 'C' if the gate is not controlled.")

        # Create the gate
        gate = cls(name=name)
        gate._sequence = sequence
        gate._target = target
        gate._control = control

        return gate

    @classmethod
    def from_tuple(cls, tup: tuple, name: str | None = None) -> "QGate":
        """
        Create a QGate object from a tuple.

        Two tuple formats are allowed:
        - (sequence, (control, target), epsilon)
        - (matrix, (matrix_target), epsilon)

        In both cases, the epsilon is discarded.

        Args:
            tup (tuple): Tuple representation of the gate.

        Returns:
            QGate: The QGate object.

        Raises:
            ValueError: If the first elements of the tuple is not a string or a np.ndarray.
            ValueError: If the tuple does not contain three elements.

        **Example**
        .. code-block:: python
            # Import the necessary module
            from cliffordplust.circuit import QGate

            # Create a QGate object from a sequence
            init_gate = QGate.from_sequence("H", name="my_H_gate", target=(2, ))
            print(init_gate)
            # Gate: my_H_gate
            # Sequence: H
            # Control: None
            # Target: (2,)

            # Get the tuple representation of the QGate object
            tup = init_gate.to_tuple()
            print(tup)
            # ('H', (2,), 0)

            # Reconstruct the QGate object from the tuple
            tup_gate = QGate.from_tuple(tup)
            print(tup_gate)
            # Sequence: H
            # Control: None
            # Target: (2,)
        """
        if len(tup) != 3:
            raise ValueError("The tuple must contain three elements.")

        first = tup[0]
        if isinstance(first, str):
            if first.startswith("C"):
                return cls.from_sequence(
                    sequence=first, name=name, target=tup[1][1:], control=tup[1][0]
                )
            else:
                return cls.from_sequence(sequence=first, name=name, target=tup[1])

        elif isinstance(first, np.ndarray):
            return cls.from_matrix(matrix=first, name=name, matrix_target=tup[1])

        else:
            raise ValueError("The first element of the tuple must be a string or a np.ndarray.")

    @property
    def name(self) -> str | None:
        """
        Get the name of the gate.

        Returns:
            str or None: The name of the gate.
        """
        return self._name

    @property
    def sequence(self) -> str | None:
        """
        Get the sequence associated with the gate decomposition.

        Returns:
            str or None: The sequence associated with the gate decomposition.
        """
        return self._sequence

    @property
    def control(self) -> int | None:
        """
        Get the control qubit.

        Returns:
            int or None: The control qubit.

        Raises:
            ValueError: If the sequence is not initialized.
        """
        # Test if the sequence is initialized
        if self.sequence is None:
            raise ValueError("The sequence must be initialized to get the control qubit.")

        return self._control

    @property
    def target(self) -> tuple[int]:
        """
        Get the target qubit.

        Returns:
            tuple[int]: The target qubit.

        Raises:
            ValueError: If the sequence is not initialized.
        """
        # Test if the sequence is initialized
        if self.sequence is None:
            raise ValueError("The sequence must be initialized to get the target qubit.")

        return self._target

    @property
    def matrix_target(self) -> tuple[int] | None:
        """
        Get the qubits on which the gate applies.

        Returns:
            tuple[int]: The qubits on which the gate applies.
        """
        return self._matrix_target

    @property
    def approx_matrix_target(self) -> tuple[int] | None:
        """
        Get the qubits on which the approximated matrix applies.

        Returns:
            tuple[int]: The qubits on which the approximated matrix applies.
        """
        return self._approx_matrix_target

    @property
    def matrix(self) -> np.ndarray:
        """
        Get the matrix representation of the gate.

        Returns:
            np.ndarray: Matrix representation of the gate.
        """
        # Calculate the matrix if it is not already known
        if self._matrix is None:
            self.calculate_matrix()

        return self._matrix

    @property
    def approx_matrix(self) -> np.ndarray | None:
        """
        Get the approximated matrix representation of the gate.

        Returns:
            np.ndarray or None: Approximated matrix representation of the gate.
        """
        return self._approx_matrix

    @property
    def nb_qubits(self) -> int:
        """
        Get the number of qubits on which the gate applies.

        Returns:
            int: The number of qubits on which the gate applies.
        """
        if self.sequence is not None:
            return int(self.control is not None) + len(self.target)
        elif self._matrix is not None:
            return len(self.matrix_target)

    @property
    def epsilon(self) -> float | None:
        """
        Get the tolerance for the gate.

        Returns:
            float or None: The tolerance for the gate.
        """
        return self._epsilon

    def __str__(self) -> str:
        """
        Convert the gate to a string representation.

        Returns:
            str: The string representation of the gate.
        """
        string = ""

        if self.name is not None:
            string += "Gate: " + self.name + "\n"

        if self.sequence is not None:
            string += "Sequence: " + self.sequence + "\n"
            string += "Control: " + str(self.control) + "\n"
            string += "Target: " + str(self.target) + "\n"

        if self._matrix is not None:
            string += "Matrix:\n" + str(self.matrix) + "\n"
            string += "Matrix target: " + str(self.matrix_target) + "\n"

        return string

    def to_tuple(self) -> tuple:
        """
        Convert the gate to a tuple representation.

        Returns:
            tuple: The tuple representation of the gate.

        Raises:
            ValueError: If the sequence is not initialized.
        """
        # Test if the sequence is initialized
        if self.sequence is None:
            raise ValueError("The sequence must be initialized to convert the gate to a tuple.")

        # Convert the gate to a tuple
        target = self.target if self.control is None else (self.control,) + self.target
        epsilon = self.epsilon if self.epsilon is not None else 0

        return (self.sequence, target, epsilon)

    def convert(self, fun: Callable[["QGate"], Any]) -> Any:
        """
        Convert the gate by using a user-defined function.

        Args:
            fun (callable): The user-defined function.

        Returns:
            any: The result of the user-defined function.
        """
        return fun(self)

    def set_decomposition(self, sequence, epsilon) -> None:
        """
        Set the decomposition of the gate. This decomposition doesn't need to be exact. The error
        epsilon specifies the error made by approximating the initial gate by the sequence and the
        exact matrix representation is stored in the approx_matrix attribute.

        Args:
            sequence (str): The decomposition of the gate.
            epsilon (float): The tolerance for the gate.

        Raises:
            ValueError: If the sequence is already initialized.
        """
        if self.sequence is not None:
            raise ValueError("The sequence is already initialized.")

        self._sequence = sequence
        self._epsilon = epsilon

        if sequence.startswith("C"):
            self._control = self.matrix_target[0]
            self._target = self.matrix_target[1:]
        else:
            self._control = None
            self._target = self.matrix_target

        self._approx_matrix = self._matrix
        self._matrix = None

        self._approx_matrix_target = self._matrix_target
        self._matrix_target = None

    def calculate_matrix(self) -> None:
        """
        Calculate the matrix representation of the gate.

        Raises:
            ValueError: If the matrix is already known.
        """
        if self._matrix is not None:
            raise ValueError("The matrix is already known.")

        # Calculate the matrix
        matrix = np.eye(2**self.nb_qubits)
        for name in self.sequence.split(" "):
            simple_matrix, matrix_target = self.get_simple_matrix(name, self.control, self.target)
            matrix = simple_matrix @ matrix

        # Store the matrix and the qubits on which the gate applies
        self._matrix = matrix
        self._matrix_target = matrix_target

    @staticmethod
    def get_simple_matrix(
        name: str, control: int | None = None, target: tuple[int] | None = None
    ) -> tuple[np.ndarray, tuple]:
        """
        Get the matrix representation of a simple gate, i.e. not a sequence of gates.

        Args:
            name (str): The name of the gate.
            control (int or None): The control qubit.
            target (tuple[int] or None): The target qubit.

        Returns:
            tuple[np.ndarray, tuple]:
                The first element of the tuple is the matrix representation of the gate.
                The second element of the tuple is a tuple of the qubits on which the gate applies.

        Raises:
            ValueError: If the sequence of the gate is not recognized.
        """
        match name:
            case "I" | "":
                return np.eye(2 ** len(target)), target

            case "X":
                return np.array([[0, 1], [1, 0]]), target
            case "Y":
                return np.array([[0, -1j], [1j, 0]]), target
            case "Z":
                return np.array([[1, 0], [0, -1]]), target

            case "H":
                return np.array([[1, 1], [1, -1]]) / np.sqrt(2), target

            case "S":
                return np.array([[1, 0], [0, 1j]]), target
            case "Sdag" | "SDAG":
                return np.array([[1, 0], [0, -1j]]), target

            case "T":
                return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]), target
            case "Tdag" | "TDAG":
                return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]]), target

            case "CX" | "CNOT":
                if control < target[0]:
                    return np.eye(4)[[0, 1, 3, 2]], (control,) + target
                else:
                    return np.eye(4)[[0, 3, 2, 1]], target + (control,)

            case "SWAP":
                return np.eye(4)[[0, 2, 1, 3]], target

            case _:
                raise ValueError(f"Unknown gate {name}.")
