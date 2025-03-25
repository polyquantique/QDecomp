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

"""Tests for the gates_utils module."""

import numpy as np
import pytest
from scipy.stats import ortho_group, special_ortho_group, unitary_group

from cliffordplust.utils.gates_utils import is_hermitian, is_orthogonal, is_special, is_unitary


@pytest.mark.parametrize(
    "matrix, result",
    [
        (np.eye(2), True),
        (-np.eye(3), False),
        (np.zeros((2, 2)), False),
        (np.array([[1.2, 3.4], [5, -1]]), False),
        (np.array([[0, 1], [1, 0]]), False),
        (np.array([[0, 1], [-1, 0]]), True),
        (special_ortho_group(dim=2, seed=42).rvs(), True),
    ],
)
def test_is_special(matrix, result):
    """Test the is_special function."""
    assert is_special(matrix) == result


@pytest.mark.parametrize(
    "matrix, result",
    [
        (-np.eye(2), True),
        (-np.eye(3), True),
        (np.zeros((2, 2)), False),
        (np.ones((2, 2)), False),
        (np.array([[1.2, 3.4], [5, -1]]), False),
        (np.array([[0, 1], [-1, 0]]), True),
        (ortho_group(dim=2, seed=42).rvs(), True),
    ],
)
def test_is_orthogonal(matrix, result):
    """Test the is_orthogonal function."""
    assert is_orthogonal(matrix) == result


@pytest.mark.parametrize(
    "matrix, result",
    [
        (-np.eye(2), True),
        (-np.eye(3), True),
        (np.zeros((2, 2)), False),
        (np.ones((2, 2)), False),
        (np.array([[1, -2], [-3, 1]]), False),
        (np.array([[0, 1], [-1, 0]]), True),
        (unitary_group(dim=2, seed=42).rvs(), True),
    ],
)
def test_is_unitary(matrix, result):
    """Test the is_unitary function."""
    assert is_unitary(matrix) == result


@pytest.mark.parametrize(
    "matrix, result",
    [
        (-np.eye(2), True),
        (-np.eye(3), True),
        (np.zeros((2, 2)), True),
        (np.ones((2, 2)), True),
        (np.array([[1.0j, 1], [1.0j, 1]]), False),
        (np.array([[0, 1], [-1, 0]]), False),
        (np.array([[1, 1j], [-1j, 1]]), True),
    ],
)
def test_is_hermitian(matrix, result):
    """Test the is_hermitian function."""
    assert is_hermitian(matrix) == result
