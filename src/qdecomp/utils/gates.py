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
Definition of common quantum gates.

This module contains the matrix definition of common single and two qubit quantum gates.

The module also contains functions to generate parametric gates:
    - :func:`power_pauli_y`: Return the Pauli Y power gate.
    - :func:`power_pauli_z`: Return the Pauli Z power gate.
    - :func:`canonical_gate`: Return the canonical gate.
    - :func:`get_matrix_from_name`: Return the matrix of a gate from its name.
"""

import sys

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

SQRT2 = np.sqrt(2)


# Single qubit gates
I = np.array([[1, 0], [0, 1]])
"""NDArray[float]: Identity gate."""

X = np.array([[0, 1], [1, 0]])
"""NDArray[float]: Pauli X gate. Alias: 'NOT'."""
NOT = X

Y = np.array([[0, -1j], [1j, 0]])
"""NDArray[float]: Pauli Y gate."""

Z = np.array([[1, 0], [0, -1]])
"""NDArray[float]: Pauli Z gate."""

H = 1 / SQRT2 * np.array([[1, 1], [1, -1]])
"""NDArray[float]: Hadamard gate."""

S = np.array([[1, 0], [0, 1.0j]])
"""NDArray[float]: Phase gate. S is the square root of the Pauli Z gate."""

V = 1 / 2 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
"""NDArray[float]: V gate. V is the square root of the Pauli X gate."""

T = np.array([[1, 0], [0, np.exp(1.0j * np.pi / 4)]])
"""NDArray[float]: T gate. T is the fourth root of the Pauli Z gate."""

W = np.exp(1.0j * np.pi / 4)
"""Complex: W gate. W is the global omega phase gate"""

# Two qubit gates
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
"""NDArray[float]: CNOT | CX gate. Controlled-Not with the control on the first qubit. Alias 'CX'."""
CX = CNOT

CNOT1 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
"""NDArray[float]: CNOT gate with the control on the second qubit. Alias 'CX1'."""
CX1 = CNOT1

DCNOT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
"""NDArray[float]: DCNOT (Double CNOT) gate. CNOT gate followed by an inverted CNOT gate. Alias 'DCX'."""
DCX = DCNOT

INV_DCNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
"""NDArray[float]: Inverted DCNOT gate. Inverted CNOT gate followed by a CNOT gate. Alias 'INV_DCX'."""
INV_DCX = INV_DCNOT

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
"""NDArray[float]: SWAP gate."""

ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
"""NDArray[float]: iSWAP gate."""

CY = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])
"""NDArray[float]: Controlled Y gate with the control on the first qubit."""

CY1 = SWAP @ CY @ SWAP
"""NDArray[float]: Controlled Y gate with the control on the second qubit."""

CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
"""NDArray[float]: Controlled Z gate with the control on the first qubit."""

CZ1 = SWAP @ CZ @ SWAP
"""NDArray[float]: Controlled Z gate with the control on the second qubit."""

CH = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1 / SQRT2, 1 / SQRT2],
        [0, 0, 1 / SQRT2, -1 / SQRT2],
    ]
)
"""NDArray[float]: Controlled Hadamard gate with the control on the first qubit."""

CH1 = SWAP @ CH @ SWAP
"""NDArray[float]: Controlled Hadamard gate with the control on the second qubit."""

MAGIC = 1 / SQRT2 * np.array([[1, 1.0j, 0, 0], [0, 0, 1.0j, 1], [0, 0, 1.0j, -1], [1, -1.0j, 0, 0]])
"""NDArray[float]: Magic gate. The magic gate is used in various decompositions algorithms."""


# Parametric gates


def power_pauli_y(p: float) -> NDArray[np.floating]:
    """
    Return the Pauli Y power gate.

    Args:
        p (float): Power of the Pauli Y gate.

    Returns:
        NDArray[float]: Pauli Y power gate.
    """
    angle = np.pi / 2 * p
    phase = np.exp(1.0j * angle)

    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return phase * matrix


def power_pauli_z(p: float) -> NDArray[np.floating]:
    """
    Return the Pauli Z power gate.

    Args:
        p (float): Power of the Pauli Z matrix.

    Returns:
        NDArray[float]: Pauli Z power gate.
    """
    return np.diag([1, np.exp(1.0j * np.pi * p)])


def canonical_gate(tx: float, ty: float, tz: float) -> NDArray[np.floating]:
    """
    Return the matrix form of the canonical gate for the given parameters.

    Args:
        tx, ty, tz (float): Parameters of the canonical gates.

    Returns:
        NDArray[float]: Matrix form of the canonical gate.
    """
    XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    YY = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
    ZZ = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    exponent = -1.0j * np.pi / 2 * (tx * XX + ty * YY + tz * ZZ)
    return expm(exponent)


def get_matrix_from_name(name: str) -> NDArray[np.floating]:
    """
    Get the matrix of a gate by its name. If the name ends with "dag", "dagger", "_dag" or
    "_dagger", the dagger of the gate is returned.

    Args:
        name (str): Name of the gate.

    Returns:
        NDArray[float]: Matrix of the gate.

    Raises:
        ValueError: If the gate name is not recognized.
    """
    # Error message for unrecognized gate names
    error_msg = f"The gate {name} is not recognized. Please check the name."

    # Dagger of the gate
    dag_suffix = ["dag", "dagger"]
    for suffix in dag_suffix:
        n_char = len(suffix)  # Number of characters in the suffix

        # Check if the name doesn't start with "{suffix}" of "_{suffix}".
        if name.lower().startswith(suffix) or name.lower().startswith(f"_{suffix}"):
            raise ValueError(error_msg)

        if name[-n_char:].lower() == suffix:
            if name[-n_char - 1] == "_":  # Handle the case where the name ends with "_{suffix}"
                n_char += 1

            matrix = get_matrix_from_name(name[:-n_char])
            return matrix.T.conj()

    # Check if the name is empty (alias for identity gate)
    if name == "":
        return I

    # Get the gate from its string name
    gate = getattr(sys.modules[__name__], name.upper(), None)
    if gate is not None:  # Check if the gate is a matrix
        return gate

    raise ValueError(error_msg)
