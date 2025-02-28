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
import numbers as num

import numpy as np
import pytest
from numpy.random import randint

from cliffordplust.rings import Zsqrt2


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.0, 2),  # float
        (1, 2.0),  # float
        (1.0, 2.0),  # float
        (1 + 1.0j, 2),  # complex
        ("a", 2),  # string
        ([1], 2),  # list
        ((1,), 2),  # tuple
        (range(5), 2),  # range
        ({1: 1}, 2),  # dict
        (
            {
                1,
            },
            2,
        ),  # set
        (None, 2),  # None
    ],
)
def test_init_exceptions(a, b):
    """Test the raise of type errors when giving wrong argument type in Zsqrt2 class"""
    with pytest.raises(TypeError, match="Expected inputs to be of type int, but got"):
        Zsqrt2(a, b)


def test_repr():
    """Test the string representation of the Zsqrt2 class"""
    nb = [
        (Zsqrt2(1, 1), "1+1√2"),
        (Zsqrt2(1, -1), "1-1√2"),
        (Zsqrt2(1, 2), "1+2√2"),
        (Zsqrt2(1, -2), "1-2√2"),
        (Zsqrt2(0, 1), "0+1√2"),
        (Zsqrt2(0, -1), "0-1√2"),
        (Zsqrt2(0, 2), "0+2√2"),
        (Zsqrt2(0, -2), "0-2√2"),
        (Zsqrt2(1, 0), "1+0√2"),
        (Zsqrt2(-1, 0), "-1+0√2"),
        (Zsqrt2(0, 0), "0+0√2"),
    ]
    assert all([str(ni[0]) == ni[1] for ni in nb])


def test_get_item():
    """Test the __get_item__ dunder method of the Zsqrt2 class"""
    a, b = (1, 2)
    ring_element = Zsqrt2(a, b)
    assert ring_element[0] == a and ring_element[1] == b


@pytest.mark.parametrize(
    ("a", "b"),
    [randint(-100, 101, size=2) for _ in range(10)]
    + [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, 1)],
)
def test_float_repr(a, b):
    """Test the float representation of the Zsqrt2 class."""
    ring_element = Zsqrt2(a, b)
    assert math.isclose(float(ring_element), a + b * math.sqrt(2))


@pytest.mark.parametrize(
    "n1", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
@pytest.mark.parametrize(
    "n2", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
def test_addition(n1, n2):
    """Test the addition definition of Zsqrt2 class"""
    if not (isinstance(n1, num.Integral) and isinstance(n2, num.Integral)):
        sum = n1 + n2
        n = n1
        n += n2
        assert (
            isinstance(sum, Zsqrt2)
            and math.isclose(float(sum), float(n1) + float(n2))
            and math.isclose(float(sum), float(n))
        )


@pytest.mark.parametrize(
    "nb", [1.0, 1 + 1.0j, "1", float(Zsqrt2(1, 1)), [1], (1,), {1}, {1: 1}, None, range(5)]
)
def test_addition_type_exceptions(nb):
    """Test the raise of a type error when doing a summation of a Z[sqrt2] with an element that is not of type int or Z[sqrt2]."""
    ring_element = Zsqrt2(1, 2)
    with pytest.raises(TypeError):
        ring_element + nb
    with pytest.raises(TypeError):
        nb + ring_element


@pytest.mark.parametrize(
    "n1", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
@pytest.mark.parametrize(
    "n2", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
def test_subtraction(n1, n2):
    """Test the subtraction definition of Zsqrt2 class"""
    if not (isinstance(n1, num.Integral) and isinstance(n2, num.Integral)):
        sub = n1 - n2
        n = n1
        n -= n2
        assert (
            isinstance(sub, Zsqrt2)
            and math.isclose(float(sub), float(n1) - float(n2))
            and math.isclose(float(sub), float(n))
        )


@pytest.mark.parametrize(
    "nb", [1.0, 1 + 1.0j, "1", float(Zsqrt2(1, 1)), [1], (1,), {1}, {1: 1}, None, range(5)]
)
def test_subtraction_type_exceptions(nb):
    """Test the raise of a type error when doing a subtraction of a Z[sqrt2] with an element that is not of type int or Z[sqrt2]."""
    ring_element = Zsqrt2(1, 2)
    with pytest.raises(TypeError):
        ring_element - nb
    with pytest.raises(TypeError):
        nb - ring_element


@pytest.mark.parametrize(
    "n1", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
@pytest.mark.parametrize(
    "n2", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
def test_product(n1, n2):
    """Test the product definition of Zsqrt2 class"""
    if not (isinstance(n1, num.Integral) and isinstance(n2, num.Integral)):
        prod = n1 * n2
        n = n1
        n *= n2
        assert (
            isinstance(prod, Zsqrt2)
            and math.isclose(float(prod), float(n1) * float(n2))
            and math.isclose(float(prod), float(n))
        )


@pytest.mark.parametrize(
    "nb", [1.0, 1 + 1.0j, "1", float(Zsqrt2(1, 1)), [1], (1,), {1}, {1: 1}, None, range(5)]
)
def test_product_type_exceptions(nb):
    """Test the raise of a type error when doing a multiplication of a Z[sqrt2] with an element that is not of type int or Z[sqrt2]."""
    ring_element = Zsqrt2(1, 2)
    with pytest.raises(TypeError):
        ring_element * nb
    with pytest.raises(TypeError):
        nb * ring_element


@pytest.mark.parametrize(
    "base",
    [Zsqrt2(randint(-20, 20), randint(-20, 20)) for _ in range(5)]
    + [
        Zsqrt2(0, 0),
        Zsqrt2(1, 0),
        Zsqrt2(0, 1),
        Zsqrt2(-1, 0),
        Zsqrt2(0, -1),
        Zsqrt2(-1, -1),
        Zsqrt2(1, 1),
    ],
)
@pytest.mark.parametrize("power", np.arange(0, 10, 1))
def test_power(base, power):
    """Test the power definition of Zsqrt2 class."""
    result = base**power
    n = base
    n **= power
    assert (
        isinstance(result, Zsqrt2)
        and math.isclose(float(result), float(base) ** power)
        and math.isclose(float(result), float(n))
    )


@pytest.mark.parametrize(
    "nb", [1.0, 1 + 1.0j, "1", float(Zsqrt2(1, 1)), [1], (1,), {1}, {1: 1}, None, range(5)]
)
def test_power_type_exceptions(nb):
    """Test the raise of a type error when raising a Z[sqrt2] element to a non-integer power."""
    ring_element = Zsqrt2(1, 2)
    with pytest.raises(TypeError):
        ring_element**nb
    with pytest.raises(TypeError):
        nb**ring_element


def test_power_value_exception():
    """Test the raise of a value error when the exponent is negative"""
    ring_element = Zsqrt2(1, 2)
    with pytest.raises(ValueError, match="Expected power to be a positive integer, but got"):
        ring_element**-1


@pytest.mark.parametrize(
    "nb", [Zsqrt2(1, 1), Zsqrt2(1, -1), Zsqrt2(-1, 1), Zsqrt2(-1, -1), Zsqrt2(0, 0)]
)
def test_negation(nb):
    """Test the negation of a Zsqrt2 element."""
    assert float(-nb) == -float(nb) and isinstance(-nb, Zsqrt2)


@pytest.mark.parametrize(
    "nb",
    [Zsqrt2(randint(-100, 101), randint(-100, 101)) for _ in range(5)]
    + [
        Zsqrt2(0, 0),
        Zsqrt2(1, 0),
        Zsqrt2(0, 1),
        Zsqrt2(-1, 0),
        Zsqrt2(0, -1),
        Zsqrt2(-1, -1),
        Zsqrt2(1, 1),
    ],
)
@pytest.mark.parametrize("precision", [-5, -3, -1, 1, 1, 3, 5, None])
def test_rounding(nb, precision):
    """Test the rounding of a Zsqrt2 element."""
    assert round(nb, precision) == round(float(nb), precision)


@pytest.mark.parametrize(
    "nb",
    [Zsqrt2(randint(-100, 101), randint(-100, 101)) for _ in range(5)]
    + [
        Zsqrt2(0, 0),
        Zsqrt2(1, 0),
        Zsqrt2(0, 1),
        Zsqrt2(-1, 0),
        Zsqrt2(0, -1),
        Zsqrt2(-1, -1),
        Zsqrt2(1, 1),
    ],
)
def test_floor(nb):
    """Test the floor rounding of a Zsqrt2 element."""
    assert math.floor(nb) == math.floor(float(nb))


@pytest.mark.parametrize(
    "nb",
    [Zsqrt2(randint(-100, 101), randint(-100, 101)) for _ in range(5)]
    + [
        Zsqrt2(0, 0),
        Zsqrt2(1, 0),
        Zsqrt2(0, 1),
        Zsqrt2(-1, 0),
        Zsqrt2(0, -1),
        Zsqrt2(-1, -1),
        Zsqrt2(1, 1),
    ],
)
def test_ceil(nb):
    """Test the ceil rounding of a Zsqrt2 element."""
    assert math.ceil(nb) == math.ceil(float(nb))


@pytest.mark.parametrize(
    ("a", "b"),
    [randint(-100, 101, size=2) for _ in range(5)]
    + [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, 1)],
)
def test_conjugate(a, b):
    """Test the √2-conjugation of a ring element."""
    ring_element = Zsqrt2(a, b)
    assert ring_element.conjugate() == Zsqrt2(a, -b)


@pytest.mark.parametrize(
    "n1", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
@pytest.mark.parametrize(
    "n2",
    [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0), 1.5],
)
def test__eq__(n1, n2):
    """Test the equality of Zsqrt2 class."""
    if math.isclose(float(n1), float(n2)):
        assert n1 == n2
    else:
        assert n1 != n2
