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
Circuit decomposition of common quantum gates.

This module contains the circuit decomposition of many common quantum gates in terms of Clifford+T gates.

The decompositions are returned as a list of :class:QGate objects.
"""

from qdecomp.utils import QGate


def dcnot_decomp(q0: int, q1: int) -> list[QGate]:
    """Circuit implementation of the double CNOT (DCNOT) gate (CNOT, then CNOT inverted).

    Decompose the DCNOT gate into a circuit of CNOT and inverted CNOT gates.

    Args:
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the DCNOT gate.
    """
    dcnot_circuit = [
        QGate.from_tuple(("CNOT", (q0, q1), 0)),
        QGate.from_tuple(("CNOT1", (q0, q1), 0)),
    ]
    return dcnot_circuit


def inv_dcnot_decomp(q0: int, q1: int) -> list[QGate]:
    """Circuit implementation of the inverted DCNOT gate (CNOT inverted, then CNOT).

    Decompose the inverted DCNOT gate into a circuit of inverted CNOT and CNOT gates.

    Args:
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the inverted DCNOT gate.
    """
    inv_dcnot_circuit = [
        QGate.from_tuple(("CNOT1", (q0, q1), 0)),
        QGate.from_tuple(("CNOT", (q0, q1), 0)),
    ]
    return inv_dcnot_circuit


def magic_decomp(q0: int, q1: int) -> list[QGate]:
    """Circuit implementation of the magic gate.

    Decompose the magic gate into a circuit of S, H, and CNOT gates.

    Args:
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the magic gate.
    """
    magic_circuit = [
        QGate.from_tuple(("S", (q0,), 0)),
        QGate.from_tuple(("S", (q1,), 0)),
        QGate.from_tuple(("H", (q1,), 0)),
        QGate.from_tuple(("CNOT1", (q0, q1), 0)),
    ]
    return magic_circuit


def magic_dag_decomp(q0: int, q1: int) -> list[QGate]:
    """Circuit implementation of the hermitian conjugate of the magic gate.

    Decompose the hermitian conjugate of the magic gate into a circuit of SDAG, H, and CNOT gates.

    Args:
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the gate.
    """
    magic_dag_circuit = [
        QGate.from_tuple(("CNOT1", (q0, q1), 0)),
        QGate.from_tuple(("H", (q1,), 0)),
        QGate.from_tuple(("SDAG", (q1,), 0)),
        QGate.from_tuple(("SDAG", (q0,), 0)),
    ]
    return magic_dag_circuit


def swap_decomp(q0: int, q1: int) -> list[QGate]:
    """Circuit implementation of the SWAP gate.

    Decompose the SWAP gate into a circuit of CNOT gates.

    Args:
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the SWAP gate.
    """
    swap_circuit = [
        QGate.from_tuple(("CNOT", (q0, q1), 0)),
        QGate.from_tuple(("CNOT1", (q0, q1), 0)),
        QGate.from_tuple(("CNOT", (q0, q1), 0)),
    ]
    return swap_circuit


def cy_decomp(q0: int, q1: int) -> list[QGate]:
    """Circuit implementation of the controlled Y (CY) gate.

    Decompose the CY gate into a circuit of SDAG, CNOT and S gates.

    Args:
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the controlled Y gate.
    """
    cy_circuit = [
        QGate.from_tuple(("SDAG", (q1,), 0)),
        QGate.from_tuple(("CNOT", (q0, q1), 0)),
        QGate.from_tuple(("S", (q1,), 0)),
    ]
    return cy_circuit


def cz_decomp(q0: int, q1: int) -> list[QGate]:
    """Circuit implementation of the controlled Z (CZ) gate.

    Decompose the CZ gate into a circuit of H, CNOT, and H gates.

    Args:
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the controlled Z gate.
    """
    cz_circuit = [
        QGate.from_tuple(("H", (q1,), 0)),
        QGate.from_tuple(("CNOT", (q0, q1), 0)),
        QGate.from_tuple(("H", (q1,), 0)),
    ]
    return cz_circuit


def ch_decomp(q0: int, q1: int):
    """Circuit implementation of the controlled Hadamard (CH) gate.

    Decompose the CH gate into a circuit of Clifford+T gates.

    Args:
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the controlled Hadamard gate.
    """
    ch_circuit = [
        QGate.from_tuple(("S", (q1,), 0)),
        QGate.from_tuple(("H", (q1,), 0)),
        QGate.from_tuple(("T", (q1,), 0)),
        QGate.from_tuple(("CNOT", (q0, q1), 0)),
        QGate.from_tuple(("TDAG", (q1,), 0)),
        QGate.from_tuple(("H", (q1,), 0)),
        QGate.from_tuple(("SDAG", (q1,), 0)),
    ]
    return ch_circuit


def iswap_decomp(q0: int, q1: int) -> list[QGate]:
    """Circuit implementation of the iSWAP gate.

    Decompose the iSWAP gate into a circuit of Clifford+T gates.

    Args:
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the iSWAP gate.
    """
    iswap_circuit = (
        swap_decomp(q0, q1)
        + cz_decomp(q0, q1)
        + [
            QGate.from_tuple(("S", (q0,), 0)),
            QGate.from_tuple(("S", (q1,), 0)),
        ]
    )
    return iswap_circuit


def common_decomp(name: str, q0: int, q1: int) -> list[QGate]:
    """Return the Clifford+T decomposition of a 4 x 4 quantum gate.

    Given the name of a 2 qubit quantum gate, return the Clifford+T decomposition of the gate.
    The decomposition is returned as a list of QGate objects.
    If the decomposition of the gate is not implemented, an error is raised.

    Args:
        name (str): Name of the quantum gate.
        q0 (int): First target qubit of the gate.
        q1 (int): Second target qubit of the gate.

    Returns:
        list[QGate]: List of QGate objects representing the decomposition of the gate.

    Raises:
        ValueError: If the decomposition of the gate is not implemented
    """
    if name == "DCNOT":
        return dcnot_decomp(q0, q1)
    elif name == "INV_DCNOT":
        return inv_dcnot_decomp(q0, q1)
    elif name == "MAGIC":
        return magic_decomp(q0, q1)
    elif name == "MAGIC_DAG":
        return magic_dag_decomp(q0, q1)
    elif name == "SWAP":
        return swap_decomp(q0, q1)
    elif name == "CY":
        return cy_decomp(q0, q1)
    elif name == "CZ":
        return cz_decomp(q0, q1)
    elif name == "CH":
        return ch_decomp(q0, q1)
    elif name == "ISWAP":
        return iswap_decomp(q0, q1)
    raise ValueError(f"Decomposition of gate {name} not implemented.")
