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
Test of exact_synthesis
"""

import numpy as np
import pytest

from cliffordplust.exact_synthesis import *


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
    sequence = "HTHTTHTHTHW"
    result = apply_sequence(sequence)
    expected = H @ T @ H @ T @ T @ H @ T @ H @ T @ H @ W
    assert np.array_equal(result, expected)


def test_apply_sequence_empty_sequence():
    """Test if apply_sequence returns the identity matrix for an empty sequence."""
    sequence = ""
    result = apply_sequence(sequence)
    assert np.array_equal(result, I)


def test_apply_sequence_invalid_character():
    """Test if apply_sequence raises a ValueError for invalid characters in the sequence."""
    sequence = "HTX"
    with pytest.raises(ValueError, match="Invalid character in sequence"):
        apply_sequence(sequence)


def test_is_unitary_true():
    """Test if is_unitary correctly identifies unitary matrices."""
    matrix = H  # Identity matrix
    assert is_unitary(matrix)


def test_is_unitary_false():
    """Test if is_unitary correctly identifies non-unitary matrices."""
    matrix = 2 * H  # Non-unitary matrix
    assert not is_unitary(matrix)


def test_convert_to_tuple_invalid_elements():
    """Test if convert_to_tuple raises TypeError for invalid matrix elements."""
    matrix = np.array([[1, 0], [0, 1]])  # Not Domega elements
    with pytest.raises(TypeError, match="Matrix elements must be of class"):
        convert_to_tuple(matrix)


def test_convert_to_tuple_non_2x2():
    """Test if convert_to_tuple raises TypeError for non-2x2 matrices."""
    D = Domega((1, 0), (0, 1), (0, 0), (0, 0))
    matrix = np.array([[D, D, D], [D, D, D], [D, D, D]])  # 3x3 matrix
    with pytest.raises(TypeError, match="Matrix must be of size 2x2"):
        convert_to_tuple(matrix)


def test_exact_synthesis_reduc_invalid_elements():
    """Test if exact_synthesis_reduc raises TypeError for invalid matrix elements."""
    matrix = np.array([[1, 0], [0, 1]])  # Not Domega elements
    with pytest.raises(TypeError, match="Matrix elements must be of class"):
        exact_synthesis_reduc(matrix)


def test_exact_synthesis_reduc_non_2x2():
    """Test if exact_synthesis_reduc raises TypeError for non-2x2 matrices."""
    D = Domega((1, 0), (0, 1), (0, 0), (0, 0))
    matrix = np.array([[D, D, D], [D, D, D], [D, D, D]])  # 3x3 matrix
    with pytest.raises(TypeError, match="Matrix must be of size 2x2"):
        exact_synthesis_reduc(matrix)


def test_exact_synthesis_reduc_non_unitary():
    """Test if exact_synthesis_reduc raises ValueError for non-unitary matrices."""
    matrix = 2 * H  # Non-unitary matrix
    with pytest.raises(ValueError, match="Matrix must be unitary"):
        exact_synthesis_reduc(matrix)


def test_lookup_sequence_invalid_elements():
    """Test if lookup_sequence raises TypeError for invalid matrix elements."""
    matrix = np.array([[1, 0], [0, 1]])  # Not Domega elements
    with pytest.raises(TypeError, match="Matrix elements must be of class"):
        lookup_sequence(matrix)


def test_lookup_sequence_non_2x2():
    """Test if lookup_sequence raises TypeError for non-2x2 matrices."""
    D = Domega((1, 0), (0, 1), (0, 0), (0, 0))
    matrix = np.array([[D, D, D], [D, D, D], [D, D, D]])  # 3x3 matrix
    with pytest.raises(TypeError, match="Matrix must be of size 2x2"):
        lookup_sequence(matrix)


def test_lookup_sequence_non_unitary():
    """Test if lookup_sequence raises ValueError for non-unitary matrices."""
    matrix = 2 * H  # Non-unitary matrix
    with pytest.raises(ValueError, match="Matrix must be unitary"):
        lookup_sequence(matrix)


def test_exact_synthesis_invalid_elements():
    """Test if exact_synthesis_alg raises TypeError for invalid matrix elements."""
    matrix = np.array([[1, 0], [0, 1]])  # Not Domega elements
    with pytest.raises(TypeError, match="Matrix elements must be of class"):
        exact_synthesis_alg(matrix)


def test_exact_synthesis_non_2x2():
    """Test if exact_synthesis_alg raises TypeError for non-2x2 matrices."""
    D = Domega((1, 0), (0, 1), (0, 0), (0, 0))
    matrix = np.array([[D, D, D], [D, D, D], [D, D, D]])  # 3x3 matrix
    with pytest.raises(TypeError, match="Matrix must be of size 2x2"):
        exact_synthesis_alg(matrix)


def test_exact_synthesis_non_unitary():
    """Test if exact_synthesis_alg raises ValueError for non-unitary matrices."""
    matrix = 2 * H  # Non-unitary matrix
    with pytest.raises(ValueError, match="Matrix must be unitary"):
        exact_synthesis_alg(matrix)


def test_evaluate_omega_exponent():
    """Test if evaluate_omega_exponent returns the correct exponent."""
    omega = Domega((0, 0), (0, 0), (1, 0), (0, 0))
    z_1 = Domega((1, 0), (0, 0), (0, 0), (0, 0))
    z_2 = Domega((0, 0), (0, 0), (1, 0), (0, 0))
    result = evaluate_omega_exponent(z_1, z_2)
    assert result == 2
    assert omega**2 * z_2 == z_1


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 200])
def test_exact_synthesis_reduc_sde(length):
    """Test if the sde of the initial matrix is greater than 3, algorithm reduces to smaller than 3, otherwise remains the same."""
    initial_sequence = random_sequence(length)
    U_i = apply_sequence(initial_sequence)
    _, U_f = exact_synthesis_reduc(U_i)
    if (U_i[0, 0] * U_i[0, 0].complex_conjugate()).sde() > 3:
        assert (U_f[0, 0] * U_f[0, 0].complex_conjugate()).sde() <= 3
    else:
        assert U_i.all() == U_f.all()
        assert (U_i[0, 0] * U_i[0, 0].complex_conjugate()).sde() == (
            U_f[0, 0] * U_f[0, 0].complex_conjugate()
        ).sde()


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 200])
def test_exact_synthesis_reduc_valid(length):
    """Test if exact_synthesis_reduc returns the correct sequence and final matrix for a valid input."""
    initial_sequence = random_sequence(length)
    initial_matrix = apply_sequence(initial_sequence)
    sequence, temp_matrix = exact_synthesis_reduc(initial_matrix)
    final_matrix = apply_sequence(sequence, temp_matrix)
    assert isinstance(sequence, str)
    assert final_matrix.shape == (2, 2)
    assert is_unitary(final_matrix)
    assert initial_matrix.all() == final_matrix.all()


def test_gen_seq_no_ending_T():
    """Test if the generated sequence does not end with a T gate."""
    sequences = generate_sequences()
    assert all(not seq.endswith("T") for seq in sequences)


@pytest.mark.parametrize("initial_sequence", generate_sequences())
def test_lookup_table_find_entries(initial_sequence):
    """Test if the lookup table is generated correctly."""
    U_3 = apply_sequence(initial_sequence)
    U_3_first_column = convert_to_tuple(U_3)
    found_sequence = lookup_sequence(U_3)
    found_U3 = apply_sequence(found_sequence)
    found_U3_first_column = convert_to_tuple(found_U3)
    assert found_sequence != None
    assert U_3_first_column == found_U3_first_column


def test_exact_synthesis_alg_maxWTH():
    """Test if the exact synthesis algorithm returns a sequence with no more than 8 following W and T gates."""
    initial_sequence = random_sequence(100)
    initial_matrix = apply_sequence(initial_sequence)
    final_sequence = exact_synthesis_alg(initial_matrix)
    assert "W" * 8 not in final_sequence
    assert "T" * 8 not in final_sequence
    assert "HH" not in final_sequence


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 200])
def test_full_exact_synthesis_alg(length):
    """Test if the full exact synthesis algorithm returns the correct sequence and final matrix."""
    initial_sequence = random_sequence(length)
    initial_matrix = apply_sequence(initial_sequence)
    final_sequence = exact_synthesis_alg(initial_matrix)
    final_matrix = apply_sequence(final_sequence)
    assert final_matrix.all() == initial_matrix.all()
