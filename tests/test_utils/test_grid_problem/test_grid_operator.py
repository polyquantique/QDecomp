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
import mpmath as mp
from cliffordplust.rings.rings import D, Zsqrt2, Dsqrt2
from cliffordplust.grid_problem.grid_operator import Grid_Operator, I, R, K, X, Z, A, B

# Valid entries for testing
valid_entries = [
    1,
    2,
    Zsqrt2(1, 1),
    Zsqrt2(10, 8),
    Zsqrt2(4, 4),
    Dsqrt2(D(1, 1), D(1, 3)),
    Dsqrt2(D(2, 2), D(1, 0)),
]

# Invalid entries for testing
invalid_entries = [1.0, 3.5, 5.111, "invalid", "not_valid"]


# Parametrize the test to run 20 times
@pytest.mark.parametrize(
    "grid_op_list",
    [
        [
            np.random.choice(valid_entries),
            np.random.choice(valid_entries),
            np.random.choice(valid_entries),
            np.random.choice(valid_entries),
        ]
        for _ in range(20)
    ],
)
def test_grid_operator_list_1(grid_op_list):
    # Test Grid_Operator initialization
    try:
        grid_op = Grid_Operator(grid_op_list)
        assert grid_op is not None  # Ensuring the object was initialized correctly
    except Exception as e:
        pytest.fail(f"Grid_Operator initialization failed: {e}")


# Parametrize the test to run 20 times
@pytest.mark.parametrize(
    "grid_op_list",
    [
        [
            [np.random.choice(valid_entries), np.random.choice(valid_entries)],
            [np.random.choice(valid_entries), np.random.choice(valid_entries)],
        ]
        for _ in range(20)
    ],
)
def test_grid_operator_list2(grid_op_list):
    # Test Grid_Operator initialization
    try:
        grid_op = Grid_Operator(grid_op_list)
        assert grid_op is not None  # Ensuring the object was initialized correctly
    except Exception as e:
        pytest.fail(f"Grid_Operator initialization failed: {e}")


# Invalid inputs that should raise a ValueError
invalid_lists = [
    [1, 2, 3],  # Not a 4-element list or 2x2 list
    [1, 2, 3, "invalid", 6],  # One invalid element in a 4-element list
    [[1, 2], [3, "invalid", 7]],  # Invalid element in a 2x2 nested list
    [[1, 2], [3, 4, 5]],  # Invalid shape in 2x2 nested list
    [[1, 2], [3]],  # Incomplete 2x2 nested list
]


@pytest.mark.parametrize("invalid_list", invalid_lists)
def test_grid_operator_list_error(invalid_list):
    with pytest.raises(
        ValueError,
        match="G must be a 4-element flat list or a 2x2 nested list with valid elements.",
    ):
        Grid_Operator(invalid_list)

# Parametrize the test to run 20 times
@pytest.mark.parametrize(
    "grid_op_array",
    [
        np.array(
            [
                [np.random.choice(valid_entries), np.random.choice(valid_entries)],
                [np.random.choice(valid_entries), np.random.choice(valid_entries)],
            ],
            dtype=object,
        )
        for _ in range(20)
    ],
)
def test_grid_operator_array(grid_op_array):
    # Test Grid_Operator initialization
    try:
        grid_op = Grid_Operator(grid_op_array)
        assert grid_op is not None  # Ensuring the object was initialized correctly
    except Exception as e:
        pytest.fail(f"Grid_Operator initialization failed: {e}")


# Parametrize the test to run 20 times
@pytest.mark.parametrize(
    "grid_op_array",
    [
        np.array(
            [
                [np.random.choice(invalid_entries), np.random.choice(valid_entries)],
                [np.random.choice(valid_entries), np.random.choice(valid_entries)],
            ],
            dtype=object,
        )
        for _ in range(20)
    ],
)
def test_grid_operator_array_error(grid_op_array):
    element = grid_op_array[0, 0]
    with pytest.raises(TypeError, match=f"Element {element} must be an int, D, Zsqrt2, or Dsqrt2."):
        Grid_Operator(grid_op_array)


grid_ops = [I, R, K, X, Z, A, B]


@pytest.mark.parametrize("grid_op", [G for G in grid_ops])
def test_repr(grid_op):
    # Expected representation as a single-line string
    expected_repr = f"[[{grid_op.a} {grid_op.b}] [{grid_op.c} {grid_op.d}]]"

    # Normalize the actual repr to a single line (remove newlines and extra spaces)
    actual_repr = repr(grid_op).replace("\n", "").replace("  ", " ").strip()

    # Assert that the normalized actual repr matches the expected repr
    assert actual_repr == expected_repr, f"Expected {expected_repr}, got {actual_repr}"


@pytest.mark.parametrize("grid_op", grid_ops)
def test_neg(grid_op):
    # Negate the grid operator
    neg_op = -grid_op

    # Verify that all elements are negated
    assert neg_op.a == -grid_op.a
    assert neg_op.b == -grid_op.b
    assert neg_op.c == -grid_op.c
    assert neg_op.d == -grid_op.d


@pytest.mark.parametrize(
    "grid_op, expected_det", [(I, 1), (R, 1), (K, 1), (X, -1), (Z, -1), (A, 1), (B, 1)]
)
def test_det(grid_op, expected_det):
    # Compute the determinant
    det = grid_op.det()

    # Verify the determinant matches the expected value
    assert det == expected_det, f"Expected {expected_det}, got {det}"


@pytest.mark.parametrize("grid_op", grid_ops)
def test_dag(grid_op):
    # Compute the dag (transpose)
    dag_op = grid_op.dag()

    # Expected transpose of the grid operator
    expected_dag = Grid_Operator([grid_op.a, grid_op.c, grid_op.b, grid_op.d])

    # Verify that the resulting dag matches the expected transpose
    assert dag_op.a == expected_dag.a
    assert dag_op.b == expected_dag.b
    assert dag_op.c == expected_dag.c
    assert dag_op.d == expected_dag.d


@pytest.mark.parametrize("grid_op", grid_ops)
def test_conjugate(grid_op):
    # Apply conjugation to the grid operator
    conjugated_op = grid_op.conjugate()

    # Function to compute the expected conjugated value
    def compute_conjugated(element):
        if isinstance(element, (Zsqrt2, Dsqrt2)):
            return element.sqrt2_conjugate()
        return element  # No change for int or D

    # Verify that each element in the conjugated grid is correct
    assert conjugated_op.a == compute_conjugated(grid_op.a)
    assert conjugated_op.b == compute_conjugated(grid_op.b)
    assert conjugated_op.c == compute_conjugated(grid_op.c)
    assert conjugated_op.d == compute_conjugated(grid_op.d)


@pytest.mark.parametrize("grid_op", grid_ops)
def test_inv_valid(grid_op):
    # Compute the inverse
    inv_op = grid_op.inv()

    # Compute the expected inverse based on the determinant
    det = grid_op.det()
    if det == 1:
        expected_inv = Grid_Operator([grid_op.d, -grid_op.b, -grid_op.c, grid_op.a])
    else:  # determinant == -1
        expected_inv = Grid_Operator([-grid_op.d, grid_op.b, grid_op.c, -grid_op.a])

    # Verify that each element of the computed inverse matches the expected result
    assert inv_op.a == expected_inv.a
    assert inv_op.b == expected_inv.b
    assert inv_op.c == expected_inv.c
    assert inv_op.d == expected_inv.d


@pytest.mark.parametrize(
    "grid_op",
    [
        # Determinant = 0 (should raise ValueError)
        Grid_Operator([1, 2, 2, 4]),
        # Determinant not equal to ±1 (should raise ValueError)
        Grid_Operator([2, 0, 0, 2]),
    ],
)
def test_inv_invalid(grid_op):
    # Verify that invalid cases raise ValueError
    with pytest.raises(
        ValueError, match="The inversion is not defined|Determinant must be non-zero"
    ):
        grid_op.inv()


@pytest.mark.parametrize("grid_op", grid_ops)
def test_as_float(grid_op):
    # Convert the grid operator to a float array
    float_array = grid_op.as_float()

    # Verify that each element in the array is the float representation of the original element
    assert float_array[0, 0] == float(grid_op.a)
    assert float_array[0, 1] == float(grid_op.b)
    assert float_array[1, 0] == float(grid_op.c)
    assert float_array[1, 1] == float(grid_op.d)

@pytest.mark.parametrize("grid_op", grid_ops)
def test_as_mpfloat(grid_op):
    # Convert the grid operator to a mpfloat array
    float_array = grid_op.as_mpmath()

    # Verify that each element in the array is the mpfloat representation of the original element
    assert float_array[0, 0] == grid_op.a.mpfloat()
    assert float_array[0, 1] == grid_op.b.mpfloat()
    assert float_array[1, 0] == grid_op.c.mpfloat()
    assert float_array[1, 1] == grid_op.d.mpfloat()


# Test cases
add_sub_test = [X + Z, R + B, K - X, B - A]
add_sub_expected = [
    Grid_Operator([1, 1, 1, -1]),  # Expected result of X + Z
    Grid_Operator(
        [
            Dsqrt2(D(1, 0), D(1, 1)),
            Dsqrt2(D(0, 0), D(1, 1)),
            Dsqrt2(D(0, 0), D(1, 1)),
            Dsqrt2(D(1, 0), D(1, 1)),
        ]
    ),  # R + B
    Grid_Operator(
        [
            Dsqrt2(D(-1, 0), D(1, 1)),
            Dsqrt2(D(-1, 0), D(-1, 1)),
            Dsqrt2(D(0, 0), D(1, 1)),
            Dsqrt2(D(0, 0), D(1, 1)),
        ]
    ),  # K - X
    Grid_Operator([0, Zsqrt2(2, 1), 0, 0]),  # B - A
]


@pytest.mark.parametrize("operation, expected", zip(add_sub_test, add_sub_expected))
def test_add_sub(operation, expected):
    """Test addition and subtraction of grid operators element by element."""
    # Compare each element one by one
    assert operation.a == expected.a, f"Expected a: {expected.a}, got: {operation.a}"
    assert operation.b == expected.b, f"Expected b: {expected.b}, got: {operation.b}"
    assert operation.c == expected.c, f"Expected c: {expected.c}, got: {operation.c}"
    assert operation.d == expected.d, f"Expected d: {expected.d}, got: {operation.d}"

@pytest.mark.parametrize(
    "scalar, grid_op",
    list(zip(valid_entries, grid_ops))  # Pairing elements from both lists
)
def test_mul_scal(scalar, grid_op):
    """Test multiplication of grid operators with a scalar."""
    result = grid_op * scalar
    assert result.a == Dsqrt2.from_ring(scalar) * grid_op.a
    assert result.b == Dsqrt2.from_ring(scalar) * grid_op.b
    assert result.c == Dsqrt2.from_ring(scalar) * grid_op.c
    assert result.d == Dsqrt2.from_ring(scalar) * grid_op.d


G = np.random.choice(grid_ops)


@pytest.mark.parametrize(
    "grid_op, exponent, expected",
    [
        (G, 0, I),  # Anything raised to the power 0 is identity
        (G, 1, G),  # Anything raised to the power 1 is itself
        (X, 2, I),  # X squared is identity
        (K, 3, K * K * K),  # K cubed
        (G, -1, G.inv()),  # Inverse of A
        (R, -2, R.inv() * R.inv()),  # R to the power of -2
        (I, 27, I),  # Identity to any power is the identity
    ],
)
def test_pow(grid_op, exponent, expected):
    """Test exponentiation of grid operators."""
    result = grid_op**exponent
    # Verify the results element-wise
    assert result.a == expected.a, f"Expected {expected.a}, got {result.a}"
    assert result.b == expected.b, f"Expected {expected.b}, got {result.b}"
    assert result.c == expected.c, f"Expected {expected.c}, got {result.c}"
    assert result.d == expected.d, f"Expected {expected.d}, got {result.d}"


@pytest.mark.parametrize(
    "grid_op, invalid_exponent",
    [
        (X, 1.5),  # Exponent is not an integer
        (B, "2"),  # Exponent is a string
    ],
)
def test_pow_invalid_exponent(grid_op, invalid_exponent):
    """Test that invalid exponents raise a TypeError."""
    with pytest.raises(TypeError, match="Exponent must be an integer."):
        _ = grid_op**invalid_exponent
