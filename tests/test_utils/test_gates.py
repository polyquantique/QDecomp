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
from qdecomp.utils.gates import *


@pytest.mark.parametrize("power", range(-3, 5))
def test_power_pauli_y(power):
    """Test the power of the Pauli Y gate."""
    result = power_pauli_y(power)
    complement = power_pauli_y(2 - power)
    inverse = power_pauli_y(-power)

    expected = np.array([[0, -1j], [1j, 0]]) if power & 1 else np.eye(2)

    assert np.allclose(result, expected)
    assert np.allclose(complement @ result, np.eye(2))
    assert np.allclose(inverse @ result, np.eye(2))


@pytest.mark.parametrize("power", range(-3, 5))
def test_power_pauli_z(power):
    """Test the power of the Pauli Z gate."""
    result = power_pauli_z(power)
    complement = power_pauli_z(2 - power)
    inverse = power_pauli_z(-power)

    expected = np.diag([1, -1]) if power & 1 else np.eye(2)

    assert np.allclose(result, expected)
    assert np.allclose(complement @ result, np.eye(2))
    assert np.allclose(inverse @ result, np.eye(2))


@pytest.mark.parametrize("tx", np.linspace(-3, 10, 7))
@pytest.mark.parametrize("ty", np.linspace(-3, 10, 7))
@pytest.mark.parametrize("tz", np.linspace(-3, 10, 7))
def test_canonical_gate(tx, ty, tz):
    """
    Test the canonical gate.

    Note: this function might need a more robust test.

    Refer to https://threeplusone.com/pubs/on_gates.pdf, Section 5, for the properties of the canonical gate.
    """
    can = canonical_gate(tx, ty, tz)

    # CAN_dag = CAN(-tx, -ty, -tz)
    assert np.allclose(can.T.conj(), canonical_gate(-tx, -ty, -tz))


@pytest.mark.parametrize(
    "name",
    [
        "I",
        "_dag",
        "dagger",
        "_DAGdagger_dagger",
        "X",
        "Y",
        "Z",
        "H",
        "S",
        "V",
        "T",
        "CX",
        "CX1",
        "DCX",
        "INV_DCX",
        "SWAP",
        "ISWAP",
        "CY",
        "CY1",
        "CZ",
        "CZ1",
        "CH",
        "CH1",
        "MAGIC",
        "W",
        "WDAGGER",
    ],
)
def test_get_matrix_from_name(name):
    """Test the single qubit gates."""
    gate = get_matrix_from_name(name)

    dag_list = ["dag", "_dag", "dagger", "_dagger", "DAG", "_DAG", "DAGGER", "_DAGGER"]
    dag_choice = dag_list[
        sum(ord(letter) for letter in name) % len(dag_list)
    ]  # Randomly choose a suffix
    gate_dag = get_matrix_from_name(name + dag_choice)

    phase_gate = name.startswith("W")
    if phase_gate:
        assert not isinstance(gate, np.ndarray)  # Phase gate is a scalar
        assert np.isclose(gate * gate_dag, 1.0)
        return

    tqg = "C" in name or "SWAP" in name  # True if the gate is a two-qubit gate
    if tqg:
        shape = (4, 4)
    else:
        shape = (2, 2)

    assert gate.shape == shape
    assert np.allclose(gate_dag @ gate, np.eye(shape[0]))


@pytest.mark.parametrize("name", ["NoT", "HADAMARD", "INVALID_GATE", "H_dag_"])
def test_get_matrix_from_name_error(name):
    """Test the error when the gate name is not recognized."""
    with pytest.raises(
        ValueError, match=f"The gate {name} is not recognized. Please check the name."
    ):
        get_matrix_from_name(name)
