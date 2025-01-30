import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from Domega import Domega, H, I, T, T_inv
from exact_synthesis import apply_sequence, exact_synthesis, is_unitary, random_sequence


def test_random_sequence_length():
    """Test if the random sequence has the correct number of 'H' gates."""
    length = 10
    sequence = random_sequence(length)
    h_count = sequence.count("H")
    assert h_count == length


def test_random_sequence_characters():
    """Test if the random sequence contains only 'H' and 'T' characters."""
    sequence = random_sequence(10)
    assert all(char in "HT" for char in sequence)


def test_apply_sequence_identity():
    """Test if apply_sequence correctly applies a sequence of gates to the identity matrix."""
    sequence = "HTHTTHTHTH"
    result = apply_sequence(sequence)
    expected = H @ T @ H @ T @ T @ H @ T @ H @ T @ H
    assert np.array_equal(result, expected)


def test_apply_sequence_invalid_character():
    """Test if apply_sequence raises a ValueError for invalid characters in the sequence."""
    sequence = "HTX"
    with pytest.raises(ValueError, match="Invalid character in sequence"):
        apply_sequence(sequence)


def test_is_unitary_true():
    """Test if is_unitary correctly identifies unitary matrices."""
    matrix = I  # Identity matrix
    assert is_unitary(matrix)


def test_is_unitary_false():
    """Test if is_unitary correctly identifies non-unitary matrices."""
    matrix = 2 * H  # Non-unitary matrix
    assert not is_unitary(matrix)


def test_exact_synthesis_invalid_elements():
    """Test if exact_synthesis raises TypeError for invalid matrix elements."""
    matrix = np.array([[1, 0], [0, 1]])  # Not Domega elements
    with pytest.raises(TypeError, match="Matrix elements must be of class"):
        exact_synthesis(matrix)


def test_exact_synthesis_non_2x2():
    """Test if exact_synthesis raises TypeError for non-2x2 matrices."""
    D = Domega((1, 0), (0, 1), (0, 0), (0, 0))
    matrix = np.array([[D, D, D], [D, D, D], [D, D, D]])  # 3x3 matrix
    with pytest.raises(TypeError, match="Matrix must be of size 2x2"):
        exact_synthesis(matrix)


def test_exact_synthesis_non_unitary():
    """Test if exact_synthesis raises ValueError for non-unitary matrices."""
    matrix = 2 * H  # Non-unitary matrix
    with pytest.raises(ValueError, match="Matrix must be unitary"):
        exact_synthesis(matrix)


# Needs to be changed once S3 sequence is added
def test_exact_synthesis_valid():
    """Test if exact_synthesis returns the correct sequence and final matrix for a valid input."""
    initial_sequence = random_sequence(20)
    initial_matrix = apply_sequence(initial_sequence)
    sequence, temp_matrix = exact_synthesis(initial_matrix)
    final_matrix = apply_sequence(sequence, temp_matrix)
    assert isinstance(sequence, str)
    assert final_matrix.shape == (2, 2)
    assert is_unitary(final_matrix)
    assert initial_matrix.all() == final_matrix.all()
