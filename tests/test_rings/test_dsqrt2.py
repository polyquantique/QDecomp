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

"""Test the Dsqrt2 class."""

import math

import mpmath as mp
import pytest

import qdecomp.rings as r
from qdecomp.rings import D, Dsqrt2

# Set the precision of mpmath to 75 decimal places
mp.mp.dps = 75

SQRT2 = math.sqrt(2)


@pytest.mark.parametrize(
    "n",
    [
        Dsqrt2((1, 1), (1, 1)),
        Dsqrt2((0, 0), (0, 0)),
        Dsqrt2((-1, 1), (1, 1)),
        Dsqrt2((1, 1), (-1, 1)),
        Dsqrt2((-55, 14), (29, 15)),
        Dsqrt2((36, 71), (41, 1)),
        Dsqrt2((302, 85), (711, 66)),
    ],
)
def test_float(n):
    """Test the float value of the Dsqrt2 class."""
    assert math.isclose(float(n), float(n.a) + float(n.b) * SQRT2)


@pytest.mark.parametrize(
    "n",
    [
        Dsqrt2((1, 1), (1, 1)),
        Dsqrt2((0, 0), (0, 0)),
        Dsqrt2((-1, 1), (1, 1)),
        Dsqrt2((1, 1), (-1, 1)),
        Dsqrt2((-55, 14), (29, 15)),
        Dsqrt2((36, 71), (41, 1)),
        Dsqrt2((302, 85), (711, 66)),
    ],
)
def test_mpfloat(n):
    """Test the mpfloat value of the Dsqrt2 class."""
    assert math.isclose(n.a.mpfloat() + n.b.mpfloat() * mp.sqrt(2), n.mpfloat())
    assert math.isclose(float(n), n.mpfloat())


def test_float_small_number():
    """Test the float value of the Dsqrt2 class with small numbers."""
    n = Dsqrt2((-1, 2), (3, 4)) ** 15
    assert math.isclose(float(n), float(Dsqrt2((-1, 2), (3, 4))) ** 15)


@pytest.mark.parametrize(
    "n1",
    [
        Dsqrt2((1, 1), (1, 1)),
        Dsqrt2((0, 0), (0, 0)),
        Dsqrt2((-1, 1), (1, 1)),
        Dsqrt2((1, 1), (-1, 1)),
        Dsqrt2((-55, 14), (29, 15)),
        Dsqrt2((36, 71), (41, 1)),
        Dsqrt2((302, 85), (711, 66)),
        0,
        5,
        -10,
    ],
)
@pytest.mark.parametrize(
    "n2",
    [
        Dsqrt2((1, 1), (1, 1)),
        Dsqrt2((0, 0), (0, 0)),
        Dsqrt2((-1, 1), (1, 1)),
        Dsqrt2((1, 1), (-1, 1)),
        Dsqrt2((-55, 14), (29, 15)),
        Dsqrt2((36, 71), (41, 1)),
        Dsqrt2((302, 85), (711, 66)),
        D(1, 1),
        D(-55, 14),
        D(302, 85),
    ],
)
def test_summation(n1, n2):
    """Test the summation of two Dsqrt2 instances."""
    n = n1
    n += n2
    assert math.isclose(float(n1 + n2), float(n1) + float(n2)) and math.isclose(
        float(n1) + float(n2), float(n)
    )


@pytest.mark.parametrize(
    "n1",
    [
        Dsqrt2((1, 1), (1, 1)),
        Dsqrt2((0, 0), (0, 0)),
        Dsqrt2((-1, 1), (1, 1)),
        Dsqrt2((1, 1), (-1, 1)),
        Dsqrt2((-55, 14), (29, 15)),
        Dsqrt2((36, 71), (41, 1)),
        Dsqrt2((302, 85), (711, 66)),
        0,
        5,
        -10,
    ],
)
@pytest.mark.parametrize(
    "n2",
    [
        Dsqrt2((1, 1), (1, 1)),
        Dsqrt2((0, 0), (0, 0)),
        Dsqrt2((-1, 1), (1, 1)),
        Dsqrt2((1, 1), (-1, 1)),
        Dsqrt2((-55, 14), (29, 15)),
        Dsqrt2((36, 71), (41, 1)),
        Dsqrt2((302, 85), (711, 66)),
        D(1, 1),
        D(-55, 14),
        D(302, 85),
    ],
)
def test_subtraction(n1, n2):
    """Test the subtraction of two Dsqrt2 instances."""
    n = n1
    n -= n2
    assert math.isclose(float(n1 - n2), float(n1) - float(n2)) and math.isclose(
        float(n1) - float(n2), float(n)
    )


@pytest.mark.parametrize(
    "n1",
    [
        Dsqrt2((1, 1), (1, 1)),
        Dsqrt2((0, 0), (0, 0)),
        Dsqrt2((-1, 1), (1, 1)),
        Dsqrt2((1, 1), (-1, 1)),
        Dsqrt2((-55, 14), (29, 15)),
        Dsqrt2((36, 71), (41, 1)),
        Dsqrt2((302, 85), (711, 66)),
        0,
        5,
        -10,
    ],
)
@pytest.mark.parametrize(
    "n2",
    [
        Dsqrt2((1, 1), (1, 1)),
        Dsqrt2((0, 0), (0, 0)),
        Dsqrt2((-1, 1), (1, 1)),
        Dsqrt2((1, 1), (-1, 1)),
        Dsqrt2((-55, 14), (29, 15)),
        Dsqrt2((36, 71), (41, 1)),
        Dsqrt2((302, 85), (711, 66)),
        D(1, 1),
        D(-55, 14),
        D(302, 85),
    ],
)
def test_multiplication(n1, n2):
    """Test the multiplication of two Dsqrt2 instances."""
    n = n1
    n *= n2
    assert math.isclose(float(n1 * n2), float(n1) * float(n2)) and math.isclose(
        float(n1) * float(n2), float(n)
    )


@pytest.mark.parametrize(
    "base",
    [
        Dsqrt2((1, 1), (1, 1)),
        Dsqrt2((0, 0), (0, 0)),
        Dsqrt2((-1, 1), (1, 1)),
        Dsqrt2((1, 1), (-1, 1)),
        Dsqrt2((-55, 14), (29, 15)),
        Dsqrt2((36, 71), (41, 1)),
        Dsqrt2((302, 85), (711, 66)),
        Dsqrt2((-57, 3), (39, 3)),
        Dsqrt2((-1, 2), (3, 4)),
    ],
)
@pytest.mark.parametrize("exp", [0, 1, 3, 6, 10])
def test_power(base, exp):
    """Test the power of a Dsqrt2 instance."""
    n = base
    n **= exp
    assert math.isclose(float(base**exp), float(base) ** exp) and math.isclose(
        float(base) ** exp, float(n)
    )


def test_equality():
    """Test the equality of two Dsqrt2 instances."""
    assert Dsqrt2((-1, 2), (3, 4)) == Dsqrt2((-1, 2), (3, 4))
    assert Dsqrt2((1, 1), (1, 1)) != Dsqrt2((1, 1), (1, 2))
    assert Dsqrt2((1, 1), (0, 0)) == D(1, 1)
    assert Dsqrt2((1, 1), (0, 0)) != D(1, 2)
    assert Dsqrt2((0, 0), (5, 2)) != D(5, 2)
    assert Dsqrt2((10, 0), (0, 0)) == 10
    assert Dsqrt2((0, 0), (0, 0)) != (0,)


def test_inequalities():
    """Test the inequalities of two Dsqrt2 instances."""
    n1 = Dsqrt2((-1, 2), (3, 4))
    n2 = Dsqrt2((3, 2), (1, 1))
    assert n2 > n1 and n1 < n2 and n1 >= n1 and n2 <= n2 and n1 > 0 and n2 < 1.5


def test_sqrt2_conjugate():
    """Test the √2-conjugate of a Dsqrt2 instance."""
    assert Dsqrt2((1, 1), (1, 1)).sqrt2_conjugate() == Dsqrt2((1, 1), (-1, 1))
    assert Dsqrt2((1, 1), (-1, 1)).sqrt2_conjugate() == Dsqrt2((1, 1), (1, 1))


def test_init_type_error_tuple():
    """Test the raise of a TypeError when initializing a Dsqrt2 instance with a tuple that has wrong entries."""
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Dsqrt2((1, 1, 1), (1, 1))
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Dsqrt2((1, 1), (1, 1, 1))
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Dsqrt2((1, 1.0), (1, 1))
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Dsqrt2((1, 1), (1.0j, 1))


def test_init_type_error_wrong_type():
    """Test the raise of a TypeError when the arguments are not tuple or D objects."""
    with pytest.raises(TypeError, match="Class arguments must be of type tuple"):
        Dsqrt2(1, 1)
    with pytest.raises(TypeError, match="Class arguments must be of type tuple"):
        Dsqrt2((1, 1), [1, 1])
    with pytest.raises(TypeError, match="Class arguments must be of type tuple"):
        Dsqrt2("11", (1, 1))


def test_init_value_error():
    """Test the raise of a ValueError when initializing a Dsqrt2 instance with negative denominator exponent."""
    with pytest.raises(ValueError, match="Denominator exponent must be positive, but"):
        Dsqrt2((1, -1), (1, 1))
    with pytest.raises(ValueError, match="Denominator exponent must be positive, but"):
        Dsqrt2((1, 1), (1, -1))


def test_get_item():
    """Test the __getitem__ method of the Dsqrt2 class."""
    n = Dsqrt2((1, 2), (3, 4))
    assert n[0] == D(1, 2)
    assert n[1] == D(3, 4)


@pytest.mark.parametrize(
    "nb",
    [
        (Dsqrt2((0, 0), (0, 0)), "0/2^0+0/2^0√2"),
        (Dsqrt2((1, 1), (1, 1)), "1/2^1+1/2^1√2"),
        (Dsqrt2((-1, 2), (-3, 1)), "-1/2^2-3/2^1√2"),
    ],
)
def test_repr(nb):
    """Test the string representation of the Dsqrt2 class."""
    assert str(nb[0]) == nb[1]


def test_summation_type_error():
    """Test the raise of a TypeError when the Dsqrt2 class is summed with the wrong type."""
    n = Dsqrt2((1, 1), (1, 1))
    with pytest.raises(TypeError, match="Summation is not defined between Dsqrt2 and"):
        n + "1"
    with pytest.raises(TypeError, match="Summation is not defined between Dsqrt2 and"):
        n + 1.0


def test_subtraction_type_error():
    """Test the raise of a TypeError when the Dsqrt2 class is subtracted with the wrong type."""
    n = Dsqrt2((1, 1), (1, 1))
    with pytest.raises(TypeError, match="Subtraction is not defined between Dsqrt2 and"):
        n - "1"
    with pytest.raises(TypeError, match="Subtraction is not defined between Dsqrt2 and"):
        n - 1.0


def test_multiplication_type_error():
    """Test the raise of a TypeError when the Dsqrt2 class is multiplied with the wrong type."""
    n = Dsqrt2((1, 1), (1, 1))
    with pytest.raises(TypeError, match="Multiplication is not defined between Dsqrt2 and"):
        n * "1"
    with pytest.raises(TypeError, match="Multiplication is not defined between Dsqrt2 and"):
        n * 1.0


def test_power_type_error():
    """Test the raise of a TypeError when the Dsqrt2 class is powered with a non-integer."""
    n = Dsqrt2((1, 1), (1, 1))
    with pytest.raises(TypeError, match="Expected power to be an integer"):
        n ** "1"
    with pytest.raises(TypeError, match="Expected power to be an integer"):
        n**1.0


def test_power_value_error():
    """Test the raise of a ValueError when the Dsqrt2 class is powered with a negative integer."""
    n = Dsqrt2((1, 1), (1, 1))
    with pytest.raises(ValueError, match="Expected power to be a positive integer"):
        n**-1


@pytest.mark.parametrize(
    "n",
    [
        (Dsqrt2((1, 1), (1, 1)), Dsqrt2((1, 1), (1, 1))),
        (-25, Dsqrt2((-25, 0), (0, 0))),
        (D(3, 7), Dsqrt2((3, 7), (0, 0))),
        (r.Zsqrt2(-11, 17), Dsqrt2((-11, 0), (17, 0))),
        (r.Zomega(5, 0, -5, -71), Dsqrt2((-71, 0), (-5, 0))),
        (r.Domega((-17, 41), (0, 0), (17, 41), (13, 5)), Dsqrt2((13, 5), (17, 41))),
    ],
)
def test_from_ring(n):
    """Test the from_ring method of the Dsqrt2 class."""
    assert Dsqrt2.from_ring(n[0]) == n[1]


@pytest.mark.parametrize(
    "n", [1.0, 1.0j, "1", [2], r.Zomega(1, 0, 1, 0), r.Domega((1, 1), (1, 1), (1, 1), (1, 1))]
)
def test_from_ring_value_error(n):
    """Test the raise of a ValueError in the from_ring method when the ring value is not a Dsqrt2 instance."""
    with pytest.raises(ValueError, match="Cannot convert"):
        Dsqrt2.from_ring(n)


def test_is_zomega():
    """Test the is_zomega method of the Dsqrt2 class."""
    assert Dsqrt2((1, 1), (1, 1)).is_zomega == False
    assert Dsqrt2((1, 0), (1, 0)).is_zomega == True
    assert Dsqrt2((1, 0), (0, 0)).is_zomega == True


def test_is_zsqrt2():
    """Test the is_zsqrt2 method of the Dsqrt2 class."""
    assert Dsqrt2((1, 1), (1, 1)).is_zsqrt2 == False
    assert Dsqrt2((1, 0), (1, 0)).is_zsqrt2 == True
    assert Dsqrt2((1, 0), (0, 0)).is_zsqrt2 == True


def test_is_integer():
    """Test the is_integer method of the Dsqrt2 class."""
    assert Dsqrt2((1, 1), (0, 0)).is_integer == False
    assert Dsqrt2((1, 0), (1, 0)).is_integer == False
    assert Dsqrt2((1, 0), (0, 0)).is_integer == True


def test_is_d():
    """Test the is_d method of the Dsqrt2 class."""
    assert Dsqrt2((1, 1), (0, 0)).is_d == True
    assert Dsqrt2((1, 0), (1, 0)).is_d == False
    assert Dsqrt2((1, 0), (0, 0)).is_d == True
