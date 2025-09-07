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

import os

import numpy as np
import pytest

from qdecomp.rings import Domega  # type: ignore[import]
from qdecomp.utils.exact_synthesis import *  # type: ignore[import]
from qdecomp.utils.exact_synthesis.exact_synthesis import (
    OMEGA,
    ONE_DOMEGA,
    ZERO_DOMEGA,
    H,
    I,
    T,
    W,
    apply_sequence,
    domega_matrix_to_tuple,
    exact_synthesis_reduc,
    generate_sequences,
    get_omega_exponent,
    is_unitary_domega,
    s3_decomposition,
)

# Set a fixed random seed for reproducibility
np.random.seed(42)


def random_sequence(n: int) -> str:
    """Generate a random sequence of H and T gates of length n
    Args:
        n (int): number of H gates in the sequence
    Returns:
        str: Random sequence of H and T gates
    """
    sequence = ""
    for _ in range(n):
        sequence += np.random.choice(
            ["H", "HT", "HTT", "HTTT", "HTTTT", "HTTTTT", "HTTTTTT", "HTTTTTTT"]
        )
    return sequence


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


@pytest.mark.parametrize(
    "sequence, expected",
    [
        ("HTHTTHTHTHW", H @ T @ H @ T @ T @ H @ T @ H @ T @ H @ W),
        ("", I),
        ("H", H),
        ("T", T),
        ("W", W),
        ("HT", H @ T),
        ("TH", T @ H),
        ("HTW", H @ T @ W),
        ("TWH", T @ W @ H),
        ("HTHT", H @ T @ H @ T),
        ("THTTHTTTTTTH", T @ H @ T @ T @ H @ T @ T @ T @ T @ T @ T @ H),
    ],
)
def test_apply_sequence_identity(sequence, expected):
    """Test if apply_sequence correctly applies a sequence of gates to the identity matrix with various sequences."""
    result = apply_sequence(sequence)
    assert np.array_equal(result, expected)


def test_apply_sequence_invalid_character():
    """Test if apply_sequence raises a ValueError for invalid characters in the sequence."""
    sequence = "HTX"
    with pytest.raises(ValueError, match="Invalid character in sequence"):
        apply_sequence(sequence)


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (H, True),
        (T, True),
        (W, True),
        (I, True),
        (H @ T, True),
        (H @ W, True),
        (T @ H, True),
        (T @ W, True),
        (2 * H, False),
        (2 * T, False),
        (1 * H, True),
        (3 * H @ T @ T @ H, False),
        (1 * H @ T @ H @ T @ T, True),
    ],
)
def test_is_unitary_output(matrix, expected):
    """Test if is_unitary correctly identifies unitary matrices."""
    assert is_unitary_domega(matrix) == expected


def test_convert_to_tuple_invalid_elements():
    """Test if convert_to_tuple raises TypeError for invalid matrix elements."""
    matrix = np.array([[1, 0], [0, 1]])  # Not Domega elements
    with pytest.raises(TypeError, match="Matrix elements must be of class"):
        domega_matrix_to_tuple(matrix)


D = Domega((1, 0), (0, 1), (0, 0), (0, 0))


@pytest.mark.parametrize(
    "matrix", [np.array([[D, D, D], [D, D, D], [D, D, D]]), np.array([D]), np.array([D, D])]
)
def test_convert_to_tuple_non_2x2(matrix):
    """Test if convert_to_tuple raises TypeError for non-2x2 matrices."""
    with pytest.raises(TypeError, match="Matrix must be of shape 2x2"):
        domega_matrix_to_tuple(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([["a", "b"], ["c", "d"]]),
        np.array([[1, 2], [3, 4]]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]),
        np.array([[1, 0], [0, 1]]),
    ],
)
def test_exact_synthesis_invalid_elements(matrix):
    """Test if exact_synthesis_alg raises TypeError for invalid matrix elements."""
    with pytest.raises(TypeError, match="The matrix elements must be of class"):
        exact_synthesis_alg(matrix)


@pytest.mark.parametrize(
    "matrix", [np.array([[D, D, D], [D, D, D], [D, D, D]]), np.array([D]), np.array([D, D])]
)
def test_exact_synthesis_non_2x2(matrix):
    """Test if exact_synthesis_alg raises TypeError for non-2x2 matrices."""
    with pytest.raises(TypeError, match="The matrix must be of shape 2x2"):
        exact_synthesis_alg(matrix)


@pytest.mark.parametrize(
    "matrix",
    [2 * H, 2 * T, 2 * W, 3 * H, 3 * T, 19 * H, 19 * W],
)
def test_exact_synthesis_non_unitary(matrix):
    """Test if exact_synthesis_alg raises ValueError for non-unitary matrices."""
    with pytest.raises(ValueError, match="The matrix must be unitary"):
        exact_synthesis_alg(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([["a", "b"], ["c", "d"]]),
        np.array([[1, 2], [3, 4]]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]),
        np.array([[1, 0], [0, 1]]),
    ],
)
def test_lookup_sequence_invalid_elements(matrix):
    """Test if lookup_sequence raises TypeError for invalid matrix elements."""
    with pytest.raises(TypeError, match="The matrix elements must be of class"):
        s3_decomposition(matrix)


@pytest.mark.parametrize(
    "matrix", [np.array([[D, D, D], [D, D, D], [D, D, D]]), np.array([D]), np.array([D, D])]
)
def test_lookup_sequence_non_2x2(matrix):
    """Test if lookup_sequence raises TypeError for non-2x2 matrices."""
    with pytest.raises(TypeError, match="The matrix must be of shape 2x2"):
        s3_decomposition(matrix)


@pytest.mark.parametrize(
    "matrix",
    [2 * H, 2 * T, 2 * W, 3 * H, 3 * T, 19 * H, 19 * W],
)
def test_lookup_sequence_non_unitary(matrix):
    """Test if lookup_sequence raises ValueError for non-unitary matrices."""
    with pytest.raises(ValueError, match="The matrix must be unitary"):
        s3_decomposition(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        H @ T @ H @ T @ H @ T @ H,
        H @ T @ H @ T @ H @ T @ H @ T @ H,
        H @ T @ H @ T @ H @ T @ H @ T @ H @ T @ H,
    ],
)
def test_lookup_sequence_invalid_sde(matrix):
    """Test if lookup_sequence raises ValueError for matrices with sde > 3."""
    with pytest.raises(ValueError, match="The matrix must have a sde < 4"):
        s3_decomposition(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([["a", "b"], ["c", "d"]]),
        np.array([[1, 2], [3, 4]]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]),
        np.array([[1, 0], [0, 1]]),
    ],
)
def test_exact_synthesis_reduc_invalid_elements(matrix):
    """Test if exact_synthesis_reduc raises TypeError for invalid matrix elements."""
    with pytest.raises(TypeError, match="The matrix elements must be of class"):
        exact_synthesis_reduc(matrix)


@pytest.mark.parametrize(
    "matrix", [np.array([[D, D, D], [D, D, D], [D, D, D]]), np.array([D]), np.array([D, D])]
)
def test_exact_synthesis_reduc_non_2x2(matrix):
    """Test if exact_synthesis_reduc raises TypeError for non-2x2 matrices."""
    with pytest.raises(TypeError, match="The matrix must be of shape 2x2"):
        exact_synthesis_reduc(matrix)


@pytest.mark.parametrize(
    "matrix",
    [2 * H, 2 * T, 2 * W, 3 * H, 3 * T, 19 * H, 19 * W],
)
def test_exact_synthesis_reduc_non_unitary(matrix):
    """Test if exact_synthesis_reduc raises ValueError for non-unitary matrices."""
    with pytest.raises(ValueError, match="The matrix must be unitary"):
        exact_synthesis_reduc(matrix)


@pytest.mark.parametrize(
    "z_1, z_2, expected",
    [
        (Domega((1, 0), (0, 0), (0, 0), (0, 0)), Domega((0, 0), (0, 0), (1, 0), (0, 0)), 2),
        (Domega((0, 0), (0, 0), (1, 0), (0, 0)), Domega((1, 0), (0, 0), (0, 0), (0, 0)), 6),
        (Domega((1, 0), (0, 0), (0, 0), (0, 0)), Domega((0, 0), (1, 0), (0, 0), (0, 0)), 1),
    ],
)
def test_evaluate_omega_exponent(z_1, z_2, expected):
    """Test if evaluate_omega_exponent returns the correct exponent."""
    omega = Domega((0, 0), (0, 0), (1, 0), (0, 0))
    result = get_omega_exponent(z_1, z_2)
    assert result == expected
    assert omega**expected * z_2 == z_1


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
def test_exact_synthesis_reduc_valid_random(length):
    """Test if exact_synthesis_reduc returns the correct sequence and final matrix for a random input."""
    initial_sequence = random_sequence(length)
    initial_matrix = apply_sequence(initial_sequence)
    sequence, temp_matrix = exact_synthesis_reduc(initial_matrix)
    final_matrix = apply_sequence(sequence, temp_matrix)
    assert isinstance(sequence, str)
    assert final_matrix.shape == (2, 2)
    assert is_unitary_domega(final_matrix)
    assert initial_matrix.all() == final_matrix.all()


def test_gen_seq_no_ending_T():
    """Test if the generated sequence does not end with a T gate."""
    sequences = generate_sequences()
    assert all(not seq.endswith("T") for seq in sequences)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 10, 20])
def test_exact_synthesis_alg_maxWTH(length):
    """Test if the exact synthesis algorithm returns a sequence with no more than 8 following W and T gates."""
    initial_sequence = random_sequence(length)
    initial_matrix = apply_sequence(initial_sequence)
    final_sequence = exact_synthesis_alg(initial_matrix)
    assert "W" * 8 not in final_sequence
    assert "T" * 8 not in final_sequence
    assert "HH" not in final_sequence


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 200])
def test_full_exact_synthesis_alg_random(length):
    """Test if the full exact synthesis algorithm returns the correct sequence and final matrix for random input."""
    initial_sequence = random_sequence(length)
    initial_matrix = apply_sequence(initial_sequence)
    final_sequence = exact_synthesis_alg(initial_matrix, insert_global_phase=True)
    final_matrix = apply_sequence(final_sequence)
    assert (final_matrix == initial_matrix).all()


@pytest.mark.parametrize(
    "matrix, sequence",
    [
        (np.array([[ZERO_DOMEGA, ONE_DOMEGA], [ONE_DOMEGA, ZERO_DOMEGA]]), "HTTTTH"),  # X
        (np.array([[ONE_DOMEGA, ZERO_DOMEGA], [ZERO_DOMEGA, -ONE_DOMEGA]]), "TTTT"),  # Z
        (
            np.array([[ZERO_DOMEGA, -1 * OMEGA**2], [OMEGA**2, ZERO_DOMEGA]]),
            "TTHTTTTHTTTTTT",
        ),  # Y
    ],
)
def test_exact_synthesis_alg_valid_known(matrix, sequence):
    """Test if the exact synthesis algorithm returns the correct sequence and final matrix for known input."""
    final_sequence = exact_synthesis_alg(matrix, insert_global_phase=True)
    assert final_sequence == sequence
    final_matrix = apply_sequence(final_sequence)
    assert (final_matrix == matrix).all()


def test_generate_s3_creates_file():
    """Test if generate_s3 creates the s3_table.json file."""
    output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "qdecomp",
        "utils",
        "exact_synthesis",
        "s3_table.json",
    )
    # Remove the file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Run the function
    generate_s3()

    # Check if the file was created
    assert os.path.exists(output_file)


@pytest.mark.parametrize(
    "sequence", ["HTHTHTHTHT", "TTTTTT", "TTTTTTHTTHTTTT", "HTTTTHTTHTHTTTTTTT", "HZHHZH", "HZZH"]
)
def test_optimize_sequence_repetition(sequence):
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
        ("HZHHZH", ""),
        ("ZTZ", "T"),
        ("STS", "ZT"),
        ("TST", "Z"),
        ("ZSZ", "S"),
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
