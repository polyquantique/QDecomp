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

"""

import numpy as np


class QGate:
    """
    Class representing a quantum gate. The gate can be defined either by its matrix or its name,
    but not both. The Gate object contains the information about the qubit on which it acts and its
    control bit, if any. The gate tolerance is also stored if it needs to be approximated.
    
    Attributes:
        sequence (str or None): Sequence associated with the gate decomposition.
        matrix (np.ndarray or None): Matrix representation of the gate.
        name (str or None): Name of the gate.
        target (int or tuple[int] or None): Number of the target qubit.
        control (int or None): Number of the control qubit.
        approx_matrix (np.ndarray or None): Approximated matrix representation of the gate.
        epsilon (float): Tolerance for the gate.
    """
    def __init__(
        self,
        name: str | None = None,
        target: int | tuple[int] | None = None,
        control: int | None = None,
    ) -> None:
        """
        Initialize the QGate object.

        Args:
            name (str or None): Name of the gate
            target (int or typle[int] or None): Number of the target qubit.
            control (int or None): Number of the control qubit.

        Raises:
            ValueError: If a target qubit is the same as the control qubit.
        """
        # Check the input arguments
        # Control qubit different from target qubit
        if control in target:
            raise ValueError(
                f"The control qubit ({control}) must be different from the target qubits {target}."
            )

        # Populate the attributes
        self._name = name

        self._matrix = None
        self._qubit_no = target if control is None else (control, ) + target
        self._sequence = None
        
        self._approx_matrix = None
        self._epsilon = None

    @classmethod
    def from_matrix(
        cls, 
        matrix: np.ndarray,
        name: str | None = None,
        qubit_no: tuple[int] = (0, ),
    ) -> "QGate":
        """
        Create a QGate object from a matrix.

        Args:
            matrix (np.ndarray): Matrix representation of the gate.
            name (str or None): Name of the gate.
            qubit_no (tuple[int]): Qubits on which the gate applies.

        Returns:
            QGate: The QGate object.

        Raises:
            ValueError: If the matrix doesn't have 2 dimensions.
            ValueError: If the matrix is not a square matrix.
            ValueError: If the matrix is not unitary.
            ValueError: If the number of rows of the matrix is not 2^nb_of_qubits.
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
        if matrix.shape[0] != 2**len(qubit_no):
            raise ValueError("The input matrix must have a size of 2^nb_of_qubit. Got shape " +
                             f"{matrix.shape} and {len(qubit_no)} qubit(s).")

        # Create the gate
        gate = cls(name=name, target=qubit_no[1:], control=qubit_no[0])
        gate._matrix = matrix
        gate._qubit_no = qubit_no

        return gate

    @classmethod
    def from_sequence(
        cls, 
        sequence: str,
        name: str | None = None,
        target: tuple[int] = (0, ),
        control: int | None = None,
    ) -> "QGate":
        """
        Create a QGate object from a sequence.

        Args:
            sequence (str): Sequence associated with the gate decomposition.
            target (tuple[int]): Qubits on which the gate applies.
            control (int or None): Control qubit.

        Returns:
            QGate: The QGate object.
        """
        # Create the gate
        gate = cls(name=name, target=target, control=control)
        gate._sequence = sequence
        gate._qubit_no = target if control is None else (control, ) + target

        return gate

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
        """
        if self.sequence is not None and self.sequence.startswith("C"):
            return self.qubit_no[0]
        else:
            return None
    
    @property
    def target(self) -> tuple[int]:
        """
        Get the target qubit.

        Returns:
            tuple[int]: The target qubit.
        """
        if self.control is None:
            return self.qubit_no
        else:
            return self.qubit_no[1:]
    
    @property
    def qubit_no(self) -> tuple[int]:
        """
        Get the qubits on which the gate applies.

        Returns:
            tuple[int]: The qubits on which the gate applies.
        """
        return self._qubit_no

    def __str__(self) -> str:
        """
        Convert the gate to a string representation.

        Returns:
            str: The string representation of the gate.
        """
        return str(self.to_tuple())

    def to_tuple(self) -> tuple:
        """
        Convert the gate to a tuple representation.

        Returns:
            tuple: The tuple representation of the gate.
        """
        if self.sequence is not None:
            sequence = self.sequence
        else:
            sequence = self.matrix

        if self.control is not None:
            target = (self.control, self.target)
        else:
            target = (self.target,)

        return (sequence, target, self.epsilon)

    def convert(self, fun: callable[["Gate"], any]) -> any:
        """
        Convert the gate by using a user-defined function.

        Args:
            fun (callable): The user-defined function.

        Returns:
            any: The result of the user-defined function.
        """
        return fun(self)

    def _calculate_matrix(self) -> None:
        """
        Calculate the matrix representation of the gate.
        """
        if self._matrix is not None:
            return
        
        # Calculate the matrix
        matrix = np.eye(2**self.n_qubits)
        for name in self.sequence.split(" "):
            simple_matrix = self.get_simple_matrix(name, self.control, self.target)
            matrix = simple_matrix @ matrix
        
        # Store the matrix and the qubits on which the gate applies
        self._matrix = matrix
        if self.control is not None:
            self._qubit_no = (self.control, ) + self.target
        else:
            self._qubit_no = self.target

    @staticmethod
    def get_simple_matrix(
        name: str,
        control: int | None = None,
        target: tuple[int] | None = None) -> np.ndarray:
        """
        Get the matrix representation of a simple gate, i.e. not a sequence of gates.

        Args:
            name (str): The name of the gate.
            control (int or None): The control qubit.
            target (tuple[int] or None): The target qubit.
        
        Returns:
            np.ndarray: The matrix representation of the gate.

        Raises:
            ValueError: If the sequence of the gate is not recognized.
        """
        match name:
            case "I" | "":
                return np.eye(2)

            case "X":
                self.matrix = np.array([[0, 1], [1, 0]])
            case "Y":
                self.matrix = np.array([[0, -1j], [1j, 0]])
            case "Z":
                self.matrix = np.array([[1, 0], [0, -1]])

            case "H":
                self.matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

            case "S":
                self.matrix = np.array([[1, 0], [0, 1j]])
            case ["Sdag", "SDAG"]:
                self.matrix = np.array([[1, 0], [0, -1j]])

            case "T":
                self.matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
            case ["Tdag", "TDAG"]:
                self.matrix = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])

            case ["CX", "CNOT"]:
                if self.control < self.target:
                    self.matrix = np.eye(4)[[0, 1, 3, 2]]
                else:
                    self.matrix = np.eye(4)[[0, 3, 2, 1]]

            case "SWAP":
                self.matrix = np.eye(4)[[0, 2, 1, 3]]

            case _:
                raise ValueError(f"Unknown gate {self.sequence}.")
