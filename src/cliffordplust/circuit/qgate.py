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
        matrix (np.ndarray or None): Matrix representation of the gate.
        name (str or None): Name of the gate.
        target (int or tuple[int] or None): Number of the target qubit.
        control (int or None): Number of the control qubit.
        epsilon (float): Tolerance for the gate.
    """
    def __init__(
        self,
        matrix: np.ndarray | None = None,
        name: str | None = None,
        target: int | tuple[int] | None = None,
        control: int | None = None,
        epsilon: float = 0,
        sequence: str | None = None,
    ) -> None:
        """

        Args:
            matrix (np.ndarray or None): Matrix representation of the gate.
            name (str or None): Name of the gate
            target (int or typle[int] or None): Number of the target qubit.
            control (int or None): Number of the control qubit.
            epsilon (float): Tolerance for the gate.
            sequence (str or None): Clifford+T sequence associated with the gate decomposition.

        Raises:
            ValueError: If the matrix is not a square matrix.
            ValueError: If the matrix is not unitary.
            ValueError: If the number of rows of the matrix is not 2^(number of targets + number of control).
            ValueError: If the target qubit is equal to the control qubit.
            ValueError: When both the sequence and the matrix are specified or unspecified.
            TypeError: If the target qubit is not specified (None).
            ValueError: If the tolerance is negative.
        """
        # Check the input arguments
        # Square matrix
        if matrix is not None and matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"The matrix must be a square matrix. Got shape {matrix.shape}.")

        # Unitary matrix
        if matrix is not None and not np.allclose(
            np.eye(matrix.shape[0]), matrix @ matrix.conj().T
        ):
            raise ValueError("The matrix must be unitary.")

        # Size of the matrix compared to the number of targets and control
        nb_tgt_ctrl = 0
        if isinstance(target, int):
            nb_tgt_ctrl += 1
        elif isinstance(target, tuple):
            nb_tgt_ctrl += len(target)
        if control is not None:
            nb_tgt_ctrl += 1
        if matrix.shape[0] != 2**nb_tgt_ctrl:
            raise ValueError(
                f"The matrix must have a size of 2^(number of targets + number of control). Got shape {matrix.shape}."
            )

        # Control qubit different from target qubit
        if isinstance(target, int) and control == control:
            raise ValueError("The target qubit must be different from the control qubit.")
        else:
            if control in target:
                raise ValueError("The control qubit must be different from the target qubit.")
        
        # Either the sequence or the matrix is specified
        if (sequence is not None) == (matrix is not None):
            raise ValueError("Either the sequence or the matrix must be specified, and not both.")
        
        # The target qubit is specified
        if target is None:
            raise TypeError("The target qubit must be specified.")

        # The tolerance is positive
        if epsilon < 0:
            raise ValueError("The tolerance must be positive.")

        # Populate the attributes
        self.matrix = matrix
        self.name = name
        self.target = target
        self.control = control
        self.epsilon = epsilon
        self.sequence = sequence

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

        Raises:
            ValueError: If the sequence of the gate is not recognized.
        """
        if self.matrix is not None:
            return

        match self.sequence:
            case "I":
                self.matrix = np.eye(2)

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
