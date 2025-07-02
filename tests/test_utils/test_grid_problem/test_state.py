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

import mpmath as mp
import numpy as np
import pytest
from qdecomp.rings.rings import *
from qdecomp.utils.grid_problem.grid_operator import *
from qdecomp.utils.grid_problem.state import *


@pytest.fixture
def valid_matrices():
    """Returns valid symmetric 2x2 matrices with determinant 1."""
    A = np.array([[mp.mpf(2), mp.mpf(1)], [mp.mpf(1), mp.mpf(2)]]) / mp.sqrt(mp.mpf(3))
    B = np.array([[mp.mpf(3), mp.mpf(1)], [mp.mpf(1), mp.mpf(3)]]) / mp.sqrt(mp.mpf(8))
    return A, B


@pytest.fixture
def invalid_matrix():
    """Returns an invalid non-symmetric 2x2 matrix."""
    return np.array([[mp.mpf(1), mp.mpf(2)], [mp.mpf(3), mp.mpf(4)]])


@pytest.fixture
def invalid_size_matrix():
    """Returns an invalid non-2x2 matrix."""
    return np.array([[mp.mpf(1), mp.mpf(2), mp.mpf(3)], [mp.mpf(4), mp.mpf(5), mp.mpf(6)]])


@pytest.fixture
def valid_state(valid_matrices):
    """Returns a valid State instance."""
    A, B = valid_matrices
    return State(A, B)


# --- Initialization Tests ---


def test_state_initialization():
    A = np.array([[mp.mpf(2), mp.mpf(1)], [mp.mpf(1), mp.mpf(2)]])
    B = np.array([[mp.mpf(3), mp.mpf(0.5)], [mp.mpf(0.5), mp.mpf(3)]])
    state = State(A, B)
    assert isinstance(state, State)
    assert np.all(state.A == A / mp.sqrt(mp.det(A)))
    assert np.all(state.B == B / mp.sqrt(mp.det(B)))


def test_invalid_initialization():
    with pytest.raises(ValueError):
        State(
            np.array([[mp.mpf(1), mp.mpf(2)], [mp.mpf(2), mp.mpf(1)]]),
            np.array([[mp.mpf(1), mp.mpf(2)], [mp.mpf(2), mp.mpf(1)]]),
        )
    with pytest.raises(ValueError):
        State(np.array([[mp.mpf(1), mp.mpf(2)]]), np.array([[mp.mpf(1), mp.mpf(1)]]))


def test_invalid_matrix_type():
    """Test that initializing with non-numpy arrays raises a TypeError."""
    with pytest.raises(TypeError):
        State([[1, 2], [3, 4]], [[5, 6], [7, 8]])


def test_invalid_matrix_elements():
    """Test that a matrix with non-mp.mpf elements raises a TypeError."""
    with pytest.raises(TypeError):
        State(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))


def test_invalid_matrix_size(invalid_size_matrix, valid_matrices):
    """Test that non-2x2 matrices raise a ValueError."""
    A, B = valid_matrices
    with pytest.raises(ValueError):
        State(invalid_size_matrix, B)
    with pytest.raises(ValueError):
        State(A, invalid_size_matrix)


def test_non_symmetric_matrix(invalid_matrix, valid_matrices):
    """Test that non-symmetric matrices raise a ValueError."""
    A, B = valid_matrices
    with pytest.raises(ValueError):
        State(invalid_matrix, B)
    with pytest.raises(ValueError):
        State(A, invalid_matrix)


# --- Property Tests ---


def test_properties(valid_state):
    """Test the computed properties of the state."""
    state = valid_state
    assert isinstance(state.z, mp.mpf)
    assert isinstance(state.zeta, mp.mpf)
    assert isinstance(state.e, mp.mpf)
    assert isinstance(state.epsilon, mp.mpf)
    assert isinstance(state.b, mp.mpf)
    assert isinstance(state.beta, mp.mpf)
    assert isinstance(state.skew, mp.mpf)
    assert isinstance(state.bias, mp.mpf)


# --- Transformation Tests ---


@pytest.mark.parametrize("G", [I, K, X, A, B, R])
def test_transform(G):
    A = np.array([[mp.mpf(7.5), mp.mpf(1)], [mp.mpf(1), mp.mpf(2)]])
    B = np.array([[mp.mpf(3), mp.mpf(0.5)], [mp.mpf(0.5), mp.mpf(21.33)]])
    state = State(A, B)
    new_state = state.transform(G)
    assert isinstance(new_state, State)
    # For I, the state should be unchanged; for others, just check type and shape
    if G is I:
        assert np.all(np.vectorize(mp.almosteq)(new_state.A, state.A))
        assert np.all(np.vectorize(mp.almosteq)(new_state.B, state.B))


def test_transform_invalid_type(valid_state):
    """Test that transforming with a non-grid operator raises TypeError."""
    state = valid_state
    with pytest.raises(TypeError):
        state.transform(42)


# --- Shift Tests ---


def test_shift(valid_state):
    """Test the shift method with a valid integer."""
    state = valid_state
    shifted_state = state.shift(2)
    assert isinstance(shifted_state, State)


def test_shift_invalid_type(valid_state):
    """Test that shifting with a non-integer raises ValueError."""
    state = valid_state
    with pytest.raises(ValueError):
        state.shift(2.5)
    with pytest.raises(ValueError):
        state.shift("two")
