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
initial representation and its error. Finally, the class provides a method to calculate the matrix
representation of the gate from its sequence.

Classes:
    - :class:`QGate`: Class representing a quantum gate.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from qdecomp.utils.gates import get_matrix_from_name

__all__ = ["QGate"]


class QGate:
    r"""
    Class representing a quantum gate. It provides the user the ability to easily create and
    manipulate quantum gates. The gates can be defined either by their matrix, their sequence or a
    tuple. A Gate object contains the information about the qubit on which it acts, its control bit,
    if any, and its error if a sequence is used to approximate it. The class also provides a method
    to calculate the matrix representation of the gate from its sequence.

    Class methods to instantiate a QGate object:
        - from_matrix: Create a QGate object from a matrix.
        - from_sequence: Create a QGate object from a sequence.
        - from_tuple: Create a QGate object from a tuple.

    Parameters:
        name (str | None): Name of the gate.
        num_qubits (int): Number of qubits on which the gate applies.

        sequence (str | None): Sequence associated with the gate decomposition.
        target (tuple[int]): Qubits on which the gate acts.

        init_matrix (np.ndarray | None): Matrix used to initialize the gate.
        sequence_matrix (np.ndarray | None): Approximated matrix representation of the gate.

        epsilon (float): Tolerance for the gate.

    Methods to change the QGate representation:
        - to_tuple: Convert the gate to a tuple representation.
        - convert: Convert the gate by using a user-defined function.
        - set_decomposition: Set the decomposition of the gate. The decomposition might be an approximation of the initial gate stored in the init_matrix attribute.
        - calculate_seq_matrix: Calculate the matrix representation of the gate from its sequence.

    **Example**

    .. code-block:: python

        >>> import numpy as np
        >>> from qdecomp.utils import QGate

        # Create a gate from a sequence
        >>> g1 = QGate.from_sequence(sequence="CNOT", target=(1, 3))
        >>> print(g1)
        Sequence: CNOT
        Target: (1, 3)

        # Create a gate from a matrix
        >>> g2 = QGate.from_matrix(matrix=np.diag([1, -1]), name="My_Z_Gate")
        >>> print(g2)
        Gate: My_Z_Gate
        Target: (0,)
        Init. matrix:
        [[ 1  0]
         [ 0 -1]]

        # Create a gate from a tuple
        >>> g3 = QGate.from_tuple(("H", (0, ), 0))
        >>> print(g3)
        Sequence: H
        Target: (0,)

        # Get the matrix of a from_sequence() gate
        >>> gate = QGate.from_sequence(sequence="X Z Y", name="my_gate")
        >>> print("Name:", gate.name)
        Name: my_gate
        >>> print("Sequence:", gate.sequence)
        Sequence: X Z Y
        >>> print("Target:", gate.target)
        Target: (0,)
        >>> print("Matrix:\n", gate.sequence_matrix, "\n")
        Matrix:
         [[ 0.+1.j  0.+0.j]
         [-0.+0.j -0.+1.j]]

        # Set the approximation sequence of a gate
        >>> approx_gate = QGate.from_matrix(matrix=np.diag([1-0.001j, 1.j+0.001]), name="approx_my_gate")
        >>> approx_gate.set_decomposition("S", epsilon=0.01)
        >>> print("Name:", approx_gate.name)
        Name: approx_my_gate
        >>> print("Sequence:", approx_gate.sequence)
        Sequence: S
        >>> print("Initial matrix:\n", approx_gate.init_matrix)
        Initial matrix:
         [[1.   -0.001j 0.   +0.j   ]
         [0.   +0.j    0.001+1.j   ]]
        >>> print("Matrix from sequence:\n", approx_gate.sequence_matrix)
        Matrix from sequence:
         [[1.+0.j 0.+0.j]
         [0.+0.j 0.+1.j]]
    """

    def __init__(
        self,
        name: str | None = None,
        target: tuple[int, ...] = (0,),
    ) -> None:
        """
        Initialize the QGate object.

        Args:
            name (str | None): Name of the gate
            target (tuple[int, ...]): Qubits on which the gate applies.

        Raises:
            ValueError: If the target qubits are not a tuple.
            ValueError: If the target qubits are not integers.
            ValueError: If the target qubits are not in ascending order.
        """
        # Check if the target is a tuple of integers
        if not isinstance(target, tuple):
            raise TypeError(f"The target qubit must be a tuple. Got {target}.")

        # Check if that target qubit(s) are integers
        if not all(isinstance(i, int) for i in target):
            raise TypeError(f"The target qubit must be a tuple of integers. Got {target}.")

        # Check if the target qubits are in ascending order
        if not all(target[i] < target[i + 1] for i in range(len(target) - 1)):
            raise ValueError(f"The target qubits must be in ascending order. Got {target}.")

        # Populate the attributes
        self._name = name

        self._sequence = None
        self._target = target

        self._init_matrix = None
        self._sequence_matrix = None

        self._epsilon = None

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        name: str | None = None,
        target: tuple[int, ...] = (0,),
        epsilon: float | None = None,
    ) -> "QGate":
        """
        Create a QGate object from a matrix.

        Args:
            matrix (np.ndarray): Matrix representation of the gate.
            name (str | None): Name of the gate.
            target (tuple[int, ...]): Qubits on which the gate applies.
            epsilon (float | None): Tolerance for the gate.

        Returns:
            QGate: The QGate object.

        Raises:
            ValueError: If the matrix is not unitary.
            ValueError: If the number of rows of the matrix is not 2^num_of_qubits.

        **Example**

        .. code-block:: python

            >>> import numpy as np
            >>> from qdecomp.utils import QGate

            >>> gate = QGate.from_matrix(np.diag([1, 1j]), name="my_S_gate", target=(1, ))
            >>> print(gate)
            Gate: my_S_gate
            Target: (1,)
            Init. matrix:
            [[1.+0.j 0.+0.j]
             [0.+0.j 0.+1.j]]
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
        if matrix.shape[0] != 2 ** len(target):
            raise ValueError(
                "The input matrix must have a size of 2^num_of_qubit. Got shape "
                + f"{matrix.shape} and {len(target)} qubit(s)."
            )

        # Create the gate
        gate = cls(name=name, target=target)
        gate._init_matrix = matrix
        gate._epsilon = epsilon

        return gate

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        name: str | None = None,
        target: tuple[int, ...] = (0,),
    ) -> "QGate":
        """
        Create a QGate object from a sequence.

        Args:
            sequence (str): Sequence associated with the gate decomposition.
            target (tuple[int, ...]): Qubits on which the gate applies.

        Returns:
            QGate: The QGate object.

        **Example**

        .. code-block:: python

            >>> from qdecomp.utils import QGate

            >>> h_gate = QGate.from_sequence("H", name="my_H_gate", target=(2, ))
            >>> print(h_gate)
            Gate: my_H_gate
            Sequence: H
            Target: (2,)

            >>> cx_gate = QGate.from_sequence("CNOT", name="my_CNOT_gate", target=(1, 3))
            >>> print(cx_gate)
            Gate: my_CNOT_gate
            Sequence: CNOT
            Target: (1, 3)
        """
        # Create the gate
        gate = cls(name=name, target=target)
        gate._sequence = sequence

        return gate

    @classmethod
    def from_tuple(cls, gate_tuple: tuple, name: str | None = None) -> "QGate":
        r"""
        Create a QGate object from a tuple.

        Two tuple formats are allowed:
            - (sequence, target, epsilon)
            - (matrix, target, epsilon)

        In the first case, the epsilon is discarded.

        Args:
            gate_tuple (tuple): Tuple representation of the gate.

        Returns:
            QGate: The QGate object.

        Raises:
            TypeError: If the first elements of the tuple is not a string or a np.ndarray.
            ValueError: If the tuple does not contain three elements.

        **Example**

        .. code-block:: python

            >>> from qdecomp.utils import QGate

            # Create a QGate object from a sequence
            >>> init_gate = QGate.from_sequence("H", name="my_H_gate", target=(2, ))
            >>> print(init_gate)
            Gate: my_H_gate
            Sequence: H
            Target: (2,)

            # Get the tuple representation of the QGate object
            >>> tup = init_gate.to_tuple()
            >>> print(tup, "\n")
            ('H', (2,), 0)

            # Reconstruct the QGate object from the tuple
            >>> tup_gate = QGate.from_tuple(tup)
            >>> print(tup_gate)
            Sequence: H
            Target: (2,)
        """
        if len(gate_tuple) != 3:
            raise ValueError("The tuple must contain three elements.")

        first = gate_tuple[0]
        if isinstance(first, str):
            return cls.from_sequence(sequence=first, name=name, target=gate_tuple[1])

        elif isinstance(first, np.ndarray):
            return cls.from_matrix(
                matrix=first, name=name, target=gate_tuple[1], epsilon=gate_tuple[2]
            )

        else:
            raise TypeError(
                f"The first element of the tuple must be a string or a np.ndarray. Got {type(first)}."
            )

    @property
    def name(self) -> str | None:
        """
        Get the name of the gate.

        Returns:
            str | None: The name of the gate.
        """
        return self._name

    @property
    def sequence(self) -> str | None:
        """
        Get the sequence associated with the gate decomposition.

        Returns:
            str | None: The sequence associated with the gate decomposition.
        """
        return self._sequence

    @property
    def target(self) -> tuple[int]:
        """
        Get the target qubit(s).

        Returns:
            tuple[int]: The target qubit(s).
        """
        return self._target

    @property
    def init_matrix(self) -> np.ndarray:
        """
        Return the matrix used to initialize the gate.

        Returns:
            np.ndarray: Matrix representation of the initialization gate.
        """
        return self._init_matrix

    @property
    def sequence_matrix(self) -> np.ndarray:
        """
        Get the matrix representation of the gate given by its sequence. If the gate is initialized
        with a matrix (obtained with the `init_matrix` property), and then its sequence is
        specified, the sequence matrix represents the matrix obtained by multiplying the gates in
        the sequence.

        Returns:
            np.ndarray: Approximated matrix representation of the gate.
        """
        # Calculate the matrix if it is not already known
        if self._sequence_matrix is None:
            self._calculate_seq_matrix()

        return self._sequence_matrix

    @property
    def matrix(self) -> np.ndarray:
        """
        Get a matrix representation of the gate. If the sequence is known, the matrix associated to
        the sequence is returned, as the `sequence_matrix` property does. If the sequence is not
        known, the matrix used to initialize the gate (accessed via the `init_matrix` property) is
        returned.

        Returns:
            np.ndarray: Matrix representation of the gate.
        """
        if self.sequence is not None:
            return self.sequence_matrix

        else:
            return self.init_matrix

    @property
    def num_qubits(self) -> int:
        """
        Get the number of qubits on which the gate applies.

        Returns:
            int: The number of qubits on which the gate applies.
        """
        return len(self._target)

    @property
    def epsilon(self) -> float | None:
        """
        Get the tolerance for the gate.

        Returns:
            float | None: The tolerance for the gate.
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

        string += "Target: " + str(self.target) + "\n"

        if self.epsilon is not None:
            string += "Epsilon: " + str(self.epsilon) + "\n"

        if self._init_matrix is not None:
            string += "Init. matrix:\n" + str(self.init_matrix) + "\n"

        if self._sequence_matrix is not None:
            string += "Seq. matrix:\n" + str(self.sequence_matrix) + "\n"

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
        epsilon = self.epsilon if self.epsilon is not None else 0

        return (self.sequence, self.target, epsilon)

    def set_decomposition(self, sequence: str, epsilon: float | None = None) -> None:
        """
        Set the decomposition of the gate. This decomposition doesn't need to be exact. The error
        epsilon specifies the error made by approximating the initial gate by the sequence and the
        exact matrix representation is stored in the approx_matrix attribute.

        Args:
            sequence (str): The decomposition of the gate.
            epsilon (float | None): The tolerance for the gate. If None, the value of the epsilon
                attribute is used.

        Raises:
            ValueError: If the sequence is already initialized.
            ValueError: If the epsilon is not already initialized for the gate and not provided as an argument.
        """
        # Reinitialize the _sequence_matrix attribute if the sequence was already specified
        if self.sequence is not None:
            self._sequence_matrix

        # Check if epsilon is defined in the gate or specified as an argument, and set it if necessary
        if epsilon is None:
            if self.epsilon is None:  # pragma: no branch
                raise ValueError("The epsilon must be initialized.")
        else:
            self._epsilon = epsilon

        # Set the sequence
        self._sequence = sequence

    def _calculate_seq_matrix(self) -> None:
        """
        Calculate the matrix representation of the gate from its sequence. The result can be
        accessed using the `sequence_matrix` attribute.

        Raises:
            ValueError: If the sequence_matrix is already known.
            ValueError: If the sequence is not initialized.
            ValueError: If the sequence contains a gate that applies on the wrong number of qubits.
        """
        # Check if the matrix is not already known
        if self._sequence_matrix is not None:
            raise ValueError("The sequence_matrix is already known.")

        # Check if the sequence is initialized
        if self.sequence is None:
            raise ValueError("The sequence must be initialized.")

        # Calculate the matrix
        matrix_shape = 2**self.num_qubits
        matrix = np.eye(matrix_shape, dtype=complex)
        for name in self.sequence.split(" "):
            simple_matrix = get_matrix_from_name(name)

            # If the simple_matrix is a scalar (global phase)
            if not isinstance(simple_matrix, np.ndarray):
                matrix = simple_matrix * matrix
                continue

            # Check if the simple_matrix has the right shape
            if simple_matrix.shape[0] != matrix_shape:
                raise ValueError(
                    f"The sequence contains a gate that applies on the wrong number of qubits: {name}."
                )

            matrix = simple_matrix @ matrix

        # Store the matrix and the qubits on which the gate applies
        self._sequence_matrix = matrix
