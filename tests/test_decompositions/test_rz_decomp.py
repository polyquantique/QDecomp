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
from qdecomp.decompositions.rz import rz_decomposition, optimize_sequence
from qdecomp.utils import QGate

"""
Tests for the rz_decomposition function to ensure correct decomposition of RZ gates into Clifford+T sequences.
"""
np.random.seed(42)  # For reproducibility


@pytest.mark.parametrize("epsilon", [1e-2, 1e-3, 1e-4])
@pytest.mark.parametrize("angle", [np.pi / 2, np.pi / 4, np.pi / 6, np.pi, np.pi / 8, np.pi / 12])
def test_rz_decomposition_precision(angle, epsilon):
    """Test if rz_decomposition returns a sequence within the desired error."""
    sequence = rz_decomposition(epsilon=epsilon, angle=angle, add_global_phase=True)
    gate = QGate.from_matrix(
        np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]]),
        target=(0,),
        epsilon=epsilon,
    )
    gate.set_decomposition(sequence, epsilon)
    sequence_matrix = gate.sequence_matrix
    error = max(np.linalg.svd(sequence_matrix - gate.init_matrix, compute_uv=False))
    assert error < epsilon


@pytest.mark.parametrize("angle", np.random.uniform(0, 2 * np.pi, 10))
@pytest.mark.parametrize("epsilon", [1e-2, 1e-3, 1e-4])
def test_rz_decomposition_random_angle(angle, epsilon):
    """Test if the decomposition of a random angle is correct."""
    sequence = rz_decomposition(epsilon, angle)
    gate = QGate.from_matrix(
        np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]]),
        target=(0,),
        epsilon=epsilon,
    )
    gate.set_decomposition(sequence, epsilon)
    sequence_matrix = gate.sequence_matrix
    error = max(np.linalg.svd(sequence_matrix - gate.init_matrix, compute_uv=False))
    assert error < epsilon


def test_rz_decomposition_identity():
    """Test decomposition of an RZ gate with a zero angle."""
    epsilon = 1e-4
    angle = 0.0
    sequence = rz_decomposition(epsilon, angle)
    assert sequence == ""


# def test_rz_decomposition_invalid_epsilon():
#     """Test that an invalid (negative) epsilon raises a ValueError."""
#     with pytest.raises(ValueError):
#         rz_decomposition(-1e-5, 1.0)  # Negative epsilon should raise an error


def test_rz_decomposition_invalid_angle():
    """Test that a non-numeric angle raises a TypeError."""
    with pytest.raises(TypeError):
        rz_decomposition(1e-5, "invalid_angle")


@pytest.mark.parametrize(
    "sequence", ["HTHTHTHTHT", "TTTTTT", "TTTTTTHTTHTTTT", "HTTTTHTTHTHTTTTTTT"]
)
def test_optimize_sequence_reptition(sequence):
    """Test if optimize_sequence returns the correct optimized sequence."""
    optimized_sequence = optimize_sequence(sequence)
    assert "T" * 2 not in optimized_sequence
    assert "H" * 2 not in optimized_sequence
    assert "Z" * 2 not in optimized_sequence
    assert "S" * 2 not in optimized_sequence


@pytest.mark.parametrize(
    "sequence, expected",
    [
        ("HTHTHTHTHT", "HTHTHTHTHT"),
        ("TTTTTT", "ZS"),
        ("TTTTTTHTTHTTTT", "ZSHSHZ"),
        ("HTTTTHTTHTHTTTTTTT", "HZHSHTHZST"),
        ("HTTTTTTTTHTTTTTTHTT", "ZSHS"),
    ],
)
def test_optimize_sequence_validity(sequence, expected):
    """Test if optimize_sequence returns the expected answer."""
    optimized_sequence = optimize_sequence(sequence)
    assert optimized_sequence == expected


def test_optimize_sequence_str():
    """Test if optimize_sequence raises a TypeError for not string input type."""
    sequence = 3
    with pytest.raises(TypeError, match="Input sequence must be a string"):
        optimize_sequence(sequence)
