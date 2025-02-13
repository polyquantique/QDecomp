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

import math

import cliffordplust.rings as r
import numpy as np
import pytest
from cliffordplust.rings import Zomega

OMEGA = (1 + 1.0j) / math.sqrt(2)


@pytest.mark.parametrize(
    "n",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
    ],
)
def test_real(n):
    """Test the real value of the Zomega class."""
    real = (n.a * OMEGA**3 + n.b * OMEGA**2 + n.c * OMEGA + n.d).real
    assert math.isclose(n.real(), real)


def test_real_small_numbers():
    """Test the real value of small Zomega numbers."""
    n = Zomega(-5, 0, 5, -7) ** 10
    assert math.isclose((complex(Zomega(-5, 0, 5, -7)) ** 10).real, n.real())


@pytest.mark.parametrize(
    "n",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
    ],
)
def test_imag(n):
    """Test the imaginary value of the Zomega class."""
    imag = (n.a * OMEGA**3 + n.b * OMEGA**2 + n.c * OMEGA + n.d).imag
    assert math.isclose(n.imag(), imag)


def test_imag_small_numbers():
    """Test the imaginary value of small Zomega numbers."""
    n = Zomega(5, -7, 5, 0) ** 11
    assert math.isclose((complex(Zomega(5, -7, 5, 0)) ** 11).imag, n.imag())


@pytest.mark.parametrize(
    "n",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
    ],
)
def test_complex(n):
    """Test the complex value of the Zomega class."""
    complex_value = n.a * OMEGA**3 + n.b * OMEGA**2 + n.c * OMEGA + n.d
    assert np.isclose(complex(n), complex_value)


@pytest.mark.parametrize(
    "n1",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
        0,
        5,
        -10,
        100,
    ],
)
@pytest.mark.parametrize(
    "n2",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
    ],
)
def test_addition(n1, n2):
    """Test the addition of two Zomega numbers."""
    n = n1
    n += n2
    assert np.isclose(complex(n1 + n2), complex(n1) + complex(n2)) and np.isclose(
        complex(n), complex(n1) + complex(n2)
    )


@pytest.mark.parametrize(
    "n1",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
        0,
        5,
        -10,
        100,
    ],
)
@pytest.mark.parametrize(
    "n2",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
    ],
)
def test_subtraction(n1, n2):
    """Test the subtraction of two Zomega numbers."""
    n = n1
    n -= n2
    assert np.isclose(complex(n1 - n2), complex(n1) - complex(n2)) and np.isclose(
        complex(n), complex(n1) - complex(n2)
    )


@pytest.mark.parametrize(
    "n1",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
        0,
        5,
        -10,
        100,
    ],
)
@pytest.mark.parametrize(
    "n2",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
    ],
)
def test_multiplication(n1, n2):
    """Test the multiplication of two Zomega numbers."""
    n = n1
    n *= n2
    assert np.isclose(complex(n1 * n2), complex(n1) * complex(n2)) and np.isclose(
        complex(n), complex(n1) * complex(n2)
    )


@pytest.mark.parametrize(
    "base",
    [
        Zomega(1, 1, 1, 1),
        Zomega(0, 0, 0, 0),
        Zomega(-1, 1, 1, -1),
        Zomega(1, 2, -5, 4),
        Zomega(-55, 14, 77, 23),
        Zomega(36, 71, 41, -94),
        Zomega(-5, 0, 5, -7),
    ],
)
@pytest.mark.parametrize("exp", [0, 1, 3, 6, 10])
def test_power(base, exp):
    """Test the power of a Zomega number."""
    n = base
    n **= exp
    assert np.isclose(complex(base**exp), complex(base) ** exp) and np.isclose(
        complex(n), complex(base) ** exp
    )


def test_sqrt2_conjugate():
    """Test the √2-conjugate method of the class Zomega."""
    n = Zomega(1, -2, -4, 5)
    assert (
        n.sqrt2_conjugate() == Zomega(-1, -2, 4, 5) and n.sqrt2_conjugate().sqrt2_conjugate() == n
    )


def test_complex_conjugate():
    """Test the complex_conjugate method of the class Zomega."""
    n = Zomega(1, -2, -4, 5)
    assert n.complex_conjugate() == Zomega(4, 2, -1, 5) and np.isclose(
        (n * n.complex_conjugate()).imag(), 0
    )


def test_init_type_errors():
    """Test the raise of a TypeError of when initializing Zomega with non-integer."""
    with pytest.raises(TypeError, match="Class arguments must be of type int but received"):
        Zomega(1, 1, 1, 1.0)
    with pytest.raises(TypeError, match="Class arguments must be of type int but received"):
        Zomega(1, "1", 1, 1)


def test_equality():
    """Test the equality of two Zomega instances."""
    assert Zomega(1, 2, 3, 4) == Zomega(1, 2, 3, 4)
    assert Zomega(1, 1, 1, 1) != Zomega(1, 1, 1, 2)
    assert Zomega(0, 0, 0, -1) == -1
    assert Zomega(0, -1, 0, 1) == 1 - 1.0j
    assert Zomega(-2, 0, 2, 5) == 5 + 2 * math.sqrt(2)
    assert Zomega(1, 1, 1, 1) != [1]


@pytest.mark.parametrize(
    "n",
    [
        (Zomega(1, 1, 1, 1), "1ω3 + 1ω2 + 1ω1 + 1"),
        (Zomega(0, 0, 0, 0), "0ω3 + 0ω2 + 0ω1 + 0"),
        (Zomega(-1, -1, 1, -1), "-1ω3 - 1ω2 + 1ω1 - 1"),
        (Zomega(1, 2, -5, 4), "1ω3 + 2ω2 - 5ω1 + 4"),
        (Zomega(-55, 14, 77, 23), "-55ω3 + 14ω2 + 77ω1 + 23"),
    ],
)
def test_str(n):
    """Test the string representation of a Zomega instance."""
    assert str(n[0]) == n[1]


def test_getitem():
    """Test the getitem method of a Zomega instance."""
    n = Zomega(1, 2, -5, 4)
    assert n[0] == 1 and n[1] == 2 and n[2] == -5 and n[3] == 4


def test_iter():
    """Test the iteration through a Zomega instance."""
    n = Zomega(1, 2, -5, 4)
    assert list(n) == [1, 2, -5, 4]


def test_addition_type_errors():
    """Test the raise of a TypeError when doing an addition with the wrong type."""
    with pytest.raises(TypeError, match="Summation is not defined between Zomega"):
        Zomega(1, 1, 1, 1) + 1.0j
    with pytest.raises(TypeError, match="Summation is not defined between Zomega"):
        Zomega(1, 1, 1, 1) + 1.0


def test_subtraction_type_errors():
    """Test the raise of a TypeError when doing a subtraction with the wrong type."""
    with pytest.raises(TypeError, match="Subtraction is not defined between Zomega"):
        Zomega(1, 1, 1, 1) - 1.0j
    with pytest.raises(TypeError, match="Subtraction is not defined between Zomega"):
        Zomega(1, 1, 1, 1) - 1.0


def test_multiplication_type_errors():
    """Test the raise of a TypeError when doing a multiplication with the wrong type."""
    with pytest.raises(TypeError, match="Product is not defined between Zomega"):
        Zomega(1, 1, 1, 1) * 1.0j
    with pytest.raises(TypeError, match="Product is not defined between Zomega"):
        Zomega(1, 1, 1, 1) * 1.0


def test_power_type_errors():
    """Test the raise of a TypeError when when the exponent is not an integer."""
    with pytest.raises(TypeError, match="Exponent must be an integer, but received"):
        Zomega(1, 1, 1, 1) ** 1.0


def test_power_value_errors():
    """Test the raise of a ValueError when the exponent is negative."""
    with pytest.raises(ValueError, match="Exponent must be a positive integer, but got"):
        Zomega(1, 1, 1, 1) ** -1


@pytest.mark.parametrize(
    "n",
    [
        (-25, Zomega(0, 0, 0, -25)),
        (12 - 5.0j, Zomega(0, -5, 0, 12)),
        (r.D(5, 0), Zomega(0, 0, 0, 5)),
        (r.Zsqrt2(-33, 17), Zomega(-17, 0, 17, -33)),
        (Zomega(1, 2, 3, 4), Zomega(1, 2, 3, 4)),
        (r.Dsqrt2((-65, 0), (21, 0)), Zomega(-21, 0, 21, -65)),
        (r.Domega((33, 0), (12, 0), (-55, 0), (23, 0)), Zomega(33, 12, -55, 23)),
    ],
)
def test_from_ring(n):
    """Test the from_ring method of the class Zomega."""
    assert Zomega.from_ring(n[0]) == n[1]


def test_from_ring_value_errors():
    """Test the raise of a ValueError if the from_ring method cannot perform the conversion."""
    with pytest.raises(ValueError, match="Cannot convert"):
        Zomega.from_ring(1.0)
    with pytest.raises(ValueError, match="Cannot convert"):
        Zomega.from_ring(1 + 1.5j)
    with pytest.raises(ValueError, match="Cannot convert"):
        Zomega.from_ring(r.D(1, 1))
    with pytest.raises(ValueError, match="Cannot convert"):
        Zomega.from_ring(r.Dsqrt2((1, 1), (1, 1)))
    with pytest.raises(ValueError, match="Cannot convert"):
        Zomega.from_ring(r.Domega((1, 1), (1, 1), (1, 1), (1, 1)))


def test_is_dsqrt2():
    """Test the is_dsqrt2 method of the class Zomega."""
    assert Zomega(1, 1, 1, 1).is_dsqrt2 == False
    assert Zomega(-1, 0, 1, -1).is_dsqrt2 == True
    assert Zomega(0, 0, 0, 4).is_dsqrt2 == True


def test_is_zsqrt2():
    """Test the is_zsqrt2 method of the class Zomega."""
    assert Zomega(1, 1, 1, 1).is_zsqrt2 == False
    assert Zomega(-1, 0, 1, -1).is_zsqrt2 == True
    assert Zomega(0, 0, 0, 4).is_zsqrt2 == True


def test_is_d():
    """Test the is_d method of the class Zomega."""
    assert Zomega(1, 1, 1, 1).is_d == False
    assert Zomega(-1, 0, 1, -1).is_d == False
    assert Zomega(0, 0, 0, 4).is_d == True


def test_is_integer():
    """Test the is_integer method of the class Zomega."""
    assert Zomega(1, 1, 1, 1).is_integer == False
    assert Zomega(-1, 0, 1, -1).is_integer == False
    assert Zomega(0, 0, 0, 4).is_integer == True
