import math
import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import pytest
from grid_algorithm_1D.Zsqrt2 import Zsqrt2
from numpy.random import randint


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.0, 2),       # float
        (1, 2.0),       # float
        (1.0, 2.0),     # float
        (1 + 1.0j, 2),  # complex
        ("a", 2),       # string
        ([1], 2),       # list
        ((1,), 2),      # tuple
        (range(5), 2),  # range
        ({1: 1}, 2),    # dict
        ({1,}, 2,),     # set
        (None, 2),      # None
    ],
)
def test_init_exceptions(a: Any, b: Any) -> None:
    """Test the raise of type errors when giving wrong argument type in Zsqrt2 class"""
    with pytest.raises(TypeError):
        Zsqrt2(a, b)


@pytest.mark.parametrize(("a", "b"), [randint(-100, 101, size=2) for i in range(15)] + [(0, 0)])
def test_repr(a: int, b: int) -> None:
    """Test the string representation of the Zsqrt2 class"""
    ring_element = Zsqrt2(a, b)
    assert str(ring_element) == (
        lambda input: (
            f"{input[0]} + {input[1]}√2" if input[1] >= 0 else f"{input[0]} - {-input[1]}√2"
        )
    )((a, b))


def test_get_item() -> None:
    """Test the __get_item__ dunder method of the Zsqrt2 class"""
    a, b = randint(-100, 101, 2)
    ring_element = Zsqrt2(a, b)
    assert ring_element[0] == a and ring_element[1] == b


@pytest.mark.parametrize(("a", "b"), [randint(-100, 101, size=2) for i in range(15)] + [(0, 0)])
def test_float_repr(a: int, b: int) -> None:
    """Test the float representation of the Zsqrt2 class."""
    ring_element = Zsqrt2(a, b)
    assert math.isclose(float(ring_element), a + b * math.sqrt(2))


@pytest.mark.parametrize(
    "n1", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
@pytest.mark.parametrize(
    "n2", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
def test_addition(n1: Zsqrt2 | int, n2: Zsqrt2 | int) -> None:
    """Test the addition definition of Zsqrt2 class"""
    if not (isinstance(n1, int) and isinstance(n2, int)):
        sum = n1 + n2
        assert isinstance(sum, Zsqrt2) and math.isclose(float(sum), float(n1) + float(n2))


@pytest.mark.parametrize(
    "nb", [1.0, 1 + 1.0j, "1", float(Zsqrt2(1, 1)), [1], (1,), {1}, {1: 1}, None, range(5)]
)
def test_addition_type_exceptions(nb: Any) -> None:
    """Test the raise of a type error when doing a summation of a Z[sqrt2] with an element that is not of type int or Z[sqrt2]."""
    ring_element = Zsqrt2(randint(-100, 101), randint(-100, 101))
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
def test_subtraction(n1: Zsqrt2 | int, n2: Zsqrt2 | int) -> None:
    """Test the subtraction definition of Zsqrt2 class"""
    if not (isinstance(n1, int) and isinstance(n2, int)):
        sub = n1 - n2
        assert isinstance(sub, Zsqrt2) and math.isclose(float(sub), float(n1) - float(n2))


@pytest.mark.parametrize(
    "nb", [1.0, 1 + 1.0j, "1", float(Zsqrt2(1, 1)), [1], (1,), {1}, {1: 1}, None, range(5)]
)
def test_subtraction_type_exceptions(nb: Any) -> None:
    """Test the raise of a type error when doing a subtraction of a Z[sqrt2] with an element that is not of type int or Z[sqrt2]."""
    ring_element = Zsqrt2(randint(-100, 101), randint(-100, 101))
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
def test_product(n1: Zsqrt2 | int, n2: Zsqrt2 | int) -> None:
    """Test the product definition of Zsqrt2 class"""
    if not (isinstance(n1, int) and isinstance(n2, int)):
        prod = n1 * n2
        assert isinstance(prod, Zsqrt2) and math.isclose(float(prod), float(n1) * float(n2))


@pytest.mark.parametrize(
    "nb", [1.0, 1 + 1.0j, "1", float(Zsqrt2(1, 1)), [1], (1,), {1}, {1: 1}, None, range(5)]
)
def test_product_type_exceptions(nb: Any) -> None:
    """Test the raise of a type error when doing a multiplication of a Z[sqrt2] with an element that is not of type int or Z[sqrt2]."""
    ring_element = Zsqrt2(randint(-100, 101), randint(-100, 101))
    with pytest.raises(TypeError):
        ring_element * nb
    with pytest.raises(TypeError):
        nb * ring_element


@pytest.mark.parametrize(
    "base", [Zsqrt2(randint(-20, 20), randint(-20, 20)) for i in range(15)] + [Zsqrt2(0, 0)]
)
@pytest.mark.parametrize("power", np.arange(0, 15, 1))
def test_power(base: Zsqrt2, power: int) -> None:
    """Test the power definition of Zsqrt2 class."""
    result = base**power
    assert isinstance(result, Zsqrt2) and math.isclose(float(result), float(base) ** power)

@pytest.mark.parametrize(
    "nb", [1.0, 1 + 1.0j, "1", float(Zsqrt2(1, 1)), [1], (1,), {1}, {1: 1}, None, range(5)]
)
def test_power_type_exceptions(nb: Any) -> None:
    """Test the raise of a type error when raising a Z[sqrt2] element to a non-integer power."""
    ring_element = Zsqrt2(randint(-100, 101), randint(-100, 101))
    with pytest.raises(TypeError):
        ring_element ** nb
    with pytest.raises(TypeError):
        nb ** ring_element


@pytest.mark.parametrize(
    "nb", [Zsqrt2(1, 1), Zsqrt2(1, -1), Zsqrt2(-1, 1), Zsqrt2(-1, -1), Zsqrt2(0, 0)]
)
def test_negation(nb: Zsqrt2) -> None:
    """Test the negation of a Zsqrt2 element."""
    assert float(-nb) == -float(nb) and isinstance(-nb, Zsqrt2)


@pytest.mark.parametrize(
    "nb", [Zsqrt2(randint(-100, 101), randint(-100, 101)) for i in range(15)] + [Zsqrt2(0, 0)]
)
@pytest.mark.parametrize("precision", [-5, -3, -1, 1, 1, 3, 5])
def test_rounding(nb: Zsqrt2, precision: int) -> None:
    """Test the rounding of a Zsqrt2 element."""
    assert round(nb, precision) == round(float(nb), precision)


@pytest.mark.parametrize(
    "nb", [Zsqrt2(randint(-100, 101), randint(-100, 101)) for i in range(15)] + [Zsqrt2(0, 0)]
)
def test_floor(nb: Zsqrt2) -> None:
    """Test the floor rounding of a Zsqrt2 element."""
    assert math.floor(nb) == math.floor(float(nb))


@pytest.mark.parametrize(
    "nb", [Zsqrt2(randint(-100, 101), randint(-100, 101)) for i in range(15)] + [Zsqrt2(0, 0)]
)
def test_ceil(nb: Zsqrt2) -> None:
    """Test the ceil rounding of a Zsqrt2 element."""
    assert math.ceil(nb) == math.ceil(float(nb))


@pytest.mark.parametrize(("a", "b"), [randint(-100, 101, size=2) for i in range(15)] + [(0, 0)])
def test_conjugate(a: int, b: int) -> None:
    """Test the √2-conjugation of a ring element."""
    ring_element = Zsqrt2(a, b)
    assert ring_element.conjugate() == Zsqrt2(a, -b)


@pytest.mark.parametrize(
    "n1", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
@pytest.mark.parametrize(
    "n2", [-10, 5, 0, Zsqrt2(-10, 5), Zsqrt2(5, 5), Zsqrt2(5, -10), Zsqrt2(-10, -10), Zsqrt2(0, 0)]
)
def test__eq__(n1: Zsqrt2 | int, n2: Zsqrt2 | int):
    """Test the equality of Zsqrt2 class."""
    if float(n1) == float(n2):
        assert n1 == n2
    else:
        assert n1 != n2


if __name__ == "__main__":
    print(type({1: 1}))
    print([randint(-100, 100, size=2) for i in range(15)])
    print([1, 2] + [3, 4])
    print(float(0))
    n = Zsqrt2(-10, -10)
    n2 = n**2
    print(n2, float(n2))
    print(float(n) ** 2, float(n2))
    print(math.isclose(float(n) ** 2, float(n2)))
    print(1.2132034355964265 ** np.int32(5))
