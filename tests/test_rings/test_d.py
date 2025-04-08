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

"""Test the D class."""

import math

import mpmath as mp
import pytest

from qdecomp.rings import D

# Set a high precision for mpmath
mp.mp.dps = 75


@pytest.mark.parametrize("num", [0, 1, 5, 100, 73, -5, -1])
@pytest.mark.parametrize("denom", [0, 1, 5, 10, 19, 50])
def test_float(num: int, denom: int) -> None:
    """Test the float value of the D class."""
    assert math.isclose(num / 2**denom, float(D(num, denom)))

@pytest.mark.parametrize("num", [0, 1, 5, 100, 73, -5, -1])
@pytest.mark.parametrize("denom", [0, 1, 5, 10, 19, 50])
def test_mpfloat(num: int, denom: int) -> None:
    """Test the mpfloat value of the D class."""
    assert math.isclose(mp.mpf(num) / 2 ** mp.mpf(denom), D(num, denom).mpfloat())
    assert math.isclose(num / 2**denom, D(num, denom).mpfloat())


@pytest.mark.parametrize(
    "n1", [D(1, 1), D(0, 0), D(-1, 1), D(-55, 14), D(-759, 30), D(57, 71), 0, 5, -20]
)
@pytest.mark.parametrize("n2", [D(1, 1), D(0, 0), D(-1, 1), D(-55, 14), D(1, 2), D(57, 71)])
def test_summation(n1, n2):
    """Test the addition of two D numbers."""
    n = n1
    n += n2
    assert math.isclose(float(n1 + n2), float(n1) + float(n2)) and math.isclose(
        float(n), float(n1) + float(n2)
    )


@pytest.mark.parametrize(
    "n1", [D(1, 1), D(0, 0), D(-1, 1), D(-55, 14), D(-759, 30), D(1, 2), D(57, 71), 0, 5, -20]
)
@pytest.mark.parametrize("n2", [D(1, 1), D(0, 0), D(-1, 1), D(-55, 14), D(1, 2), D(57, 71)])
def test_subtraction(n1, n2):
    """Test the subtraction of two D numbers."""
    n = n1
    n -= n2
    assert math.isclose(float(n1 - n2), float(n1) - float(n2)) and math.isclose(
        float(n), float(n1) - float(n2)
    )


@pytest.mark.parametrize(
    "n1", [D(1, 1), D(0, 0), D(-1, 1), D(-55, 14), D(-759, 30), D(1, 2), D(57, 71), 0, 5, -20]
)
@pytest.mark.parametrize("n2", [D(1, 1), D(0, 0), D(-1, 1), D(-55, 14), D(1, 2), D(57, 71)])
def test_multiplication(n1, n2):
    """Test the multiplication of two D numbers."""
    n = n1
    n *= n2
    assert math.isclose(float(n1 * n2), float(n1) * float(n2)) and math.isclose(
        float(n), float(n1) * float(n2)
    )


@pytest.mark.parametrize(
    "base", [D(1, 1), D(0, 0), D(-1, 1), D(-55, 14), D(-759, 30), D(1, 2), D(57, 71)]
)
@pytest.mark.parametrize("exponent", [0, 3, 5, 7, 10, 20])
def test_power(base, exponent):
    """Test the power of a D number."""
    n = base
    n **= exponent
    assert math.isclose(float(base**exponent), float(base) ** exponent) and math.isclose(
        float(n), float(base) ** exponent
    )


@pytest.mark.parametrize(
    "n",
    [
        [D(1, 1), "1/2^1"],
        [D(0, 0), "0/2^0"],
        [D(-1, 1), "-1/2^1"],
        [D(-55, 14), "-55/2^14"],
        [D(1, 2), "1/2^2"],
        [D(57, 71), "57/2^71"],
    ],
)
def test_repr(n):
    """Test the string representation of a D number."""
    assert str(n[0]) == n[1]


@pytest.mark.parametrize("not_int", ["1", 1.0, [1], 1.0j])
def test_init_type_error(not_int):
    """Test the raise of a TypeError when the D class is initialized with a non-integer."""
    with pytest.raises(TypeError, match="Class arguments must be of type int, but received"):
        D(not_int, 1)
    with pytest.raises(TypeError, match="Class arguments must be of type int, but received"):
        D(1, not_int)


def test_init_value_error():
    """Test the raise of a ValueError when the D class is initialized with a negative denominator."""
    with pytest.raises(ValueError, match="Denominator exponent must be positive"):
        D(1, -1)


def test_abs():
    """Test the absolute value of a D number."""
    assert abs(D(5, 1)) == D(5, 1) and abs(D(-5, 1)) == D(5, 1)


def test_equality():
    """Test the equality of two D numbers."""
    assert (
        D(1, 1) == D(1, 1)
        and D(1, 1) != D(1, 2)
        and D(1, 1) != D(2, 1)
        and D(0, 0) == 0
        and D(-10, 0) == -10
        and D(1, 1) != 1.0j
    )


def test_inequality():
    """Test the inequality of two D numbers."""
    assert (
        D(1, 1) < D(2, 1)
        and D(1, 2) < D(1, 1)
        and D(1, 1) > D(1, 2)
        and D(2, 1) > D(1, 1)
        and D(1, 1) <= D(1, 1)
        and D(1, 1) >= D(1, 1)
        and D(1, 1) <= D(2, 1)
    )


@pytest.mark.parametrize("nb", [1.0, 1.0j, (1, 1), "1"])
def test_summation_error(nb):
    """Test the raise of a TypeError when the addition of D numbers is not possible."""
    with pytest.raises(TypeError, match="Summation is not defined between D and"):
        D(1, 1) + nb
    with pytest.raises(TypeError, match="Summation is not defined between D and"):
        nb + D(1, 1)


@pytest.mark.parametrize("nb", [1.0, 1.0j, (1, 1), "1"])
def test_subtraction_error(nb):
    """Test the raise of a TypeError when the subtraction of D numbers is not possible."""
    with pytest.raises(TypeError, match="Subtraction is not defined between D and"):
        D(1, 1) - nb


@pytest.mark.parametrize("nb", [1.0, 1.0j, (1, 1), "1"])
def test_multiplication_error(nb):
    """Test the raise of a TypeError when the multiplication of D numbers is not possible."""
    with pytest.raises(TypeError, match="Product is not defined between D and"):
        D(1, 1) * nb
    with pytest.raises(TypeError, match="Product is not defined between D and"):
        nb * D(1, 1)


def test_power_type_error():
    """Test the raise of a TypeError when the power of a D number is not an integer."""
    with pytest.raises(TypeError, match="Expected power to be an integer"):
        D(1, 1) ** 1.0


def test_power_value_error():
    """Test the raise of a ValueError when the power of a D number is negative."""
    with pytest.raises(ValueError, match="Expected power to be a positive integer"):
        D(1, 1) ** -1
