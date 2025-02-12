import math

import numpy as np
import pytest

import cliffordplust.rings as r
from cliffordplust.rings import Domega

OMEGA = (1 + 1.0j) / math.sqrt(2)

SQRT2_DOMEGA = Domega.from_ring(r.Zsqrt2(0, 1))
INV_DOMEGA = Domega((-1, 1), (0, 0), (1, 1), (0, 0))


@pytest.mark.parametrize(
    "n",
    [
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
    ],
)
def test_real(n):
    """Test the real method of Domega"""
    real = (
        float(n.a) * OMEGA**3 + float(n.b) * OMEGA**2 + float(n.c) * OMEGA + float(n.d)
    ).real
    assert np.isclose(n.real(), real)


@pytest.mark.parametrize(
    "n",
    [
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
    ],
)
def test_imag(n):
    """Test the imag method of Domega"""
    imag = (
        float(n.a) * OMEGA**3 + float(n.b) * OMEGA**2 + float(n.c) * OMEGA + float(n.d)
    ).imag
    assert np.isclose(n.imag(), imag)


@pytest.mark.parametrize(
    "n",
    [
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
    ],
)
def test_complex(n):
    """Test the complex method of Domega"""
    complex_value = (
        float(n.a) * OMEGA**3 + float(n.b) * OMEGA**2 + float(n.c) * OMEGA + float(n.d)
    )
    assert np.isclose(complex(n), complex_value)


@pytest.mark.parametrize(
    "n1",
    [
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
        0,
        5,
        -10,
        100,
    ],
)
@pytest.mark.parametrize(
    "n2",
    [
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
        r.D(1, 1),
        r.D(-67, 23),
    ],
)
def test_addition(n1, n2):
    """Test the addition of two Domega numbers."""
    n = n1
    n += n2
    assert np.isclose(complex(n1 + n2), complex(n1) + complex(n2)) and np.isclose(
        complex(n), complex(n1) + complex(n2)
    )


@pytest.mark.parametrize(
    "n1",
    [
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
        0,
        5,
        -10,
        100,
    ],
)
@pytest.mark.parametrize(
    "n2",
    [
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
        r.D(1, 1),
        r.D(-67, 23),
    ],
)
def test_subtraction(n1, n2):
    """Test the subtraction of two Domega numbers."""
    n = n1
    n -= n2
    assert np.isclose(complex(n1 - n2), complex(n1) - complex(n2)) and np.isclose(
        complex(n), complex(n1) - complex(n2)
    )


@pytest.mark.parametrize(
    "n1",
    [
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
        0,
        5,
        -10,
        100,
    ],
)
@pytest.mark.parametrize(
    "n2",
    [
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
        r.D(1, 1),
        r.D(-67, 23),
    ],
)
def test_multiplication(n1, n2):
    """Test the multiplication of two Domega numbers."""
    n = n1
    n *= n2
    assert np.isclose(complex(n1 * n2), complex(n1) * complex(n2)) and np.isclose(
        complex(n), complex(n1) * complex(n2)
    )


@pytest.mark.parametrize(
    "base",
    [
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((-55, 14), (77, 23), (36, 71), (-41, 94)),
        Domega((131, 211), (23, 43), (-73, 43), (41, 94)),
    ],
)
@pytest.mark.parametrize("exp", [0, 1, 3, 6, 10])
def test_power(base, exp):
    """Test the power of a Domega number."""
    n = base
    n **= exp
    assert np.isclose(complex(base**exp), complex(base) ** exp) and np.isclose(
        complex(n), complex(base) ** exp
    )


def test_sqrt2_conjugate():
    """Test the sqrt2-conjugate method of the class Domega."""
    n = Domega((1, 2), (3, 4), (-5, 6), (-7, 8))
    assert n.sqrt2_conjugate() == Domega((-1, 2), (3, 4), (5, 6), (-7, 8))


def test_complex_conjugate():
    """Test the complex conjugate method of the class Domega."""
    n = Domega((1, 2), (3, 4), (-5, 6), (-7, 8))
    assert n.complex_conjugate() == Domega((5, 6), (-3, 4), (-1, 2), (-7, 8))
    assert np.isclose((n * n.complex_conjugate()).imag(), 0)


def test_init_type_error_tuple():
    """Test the type errors of the Domega class."""
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Domega((1, 1), (1, 1), (1, 1), (1, 1.0))
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Domega((1, 1), ("1", 1), (1, 1), (1, 1))
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Domega((1, 1), (1, 1), (1, 1, 1), (1, 1))


def test_init_value_error():
    """Test the value errors of the Domega class."""
    with pytest.raises(ValueError, match="Denominator exponent must be positive"):
        Domega((1, 1), (1, -1), (1, 1), (1, 1))
    with pytest.raises(ValueError, match="Denominator exponent must be positive"):
        Domega((1, 1), (1, 1), (1, 1), (1, -1))


def test_init_type_error_wrong_type():
    """Test the type errors of the Domega class."""
    with pytest.raises(TypeError, match="lass arguments must be of type tuple"):
        Domega(1, 1, 1, 1)
    with pytest.raises(TypeError, match="lass arguments must be of type tuple"):
        Domega([1, 1], (1, 1), (1, 1), (1, 1))


@pytest.mark.parametrize(
    "n",
    [
        Domega((1, 1), (1, 1), (1, 1), (1, 1)),
        Domega((0, 0), (0, 0), (0, 0), (0, 0)),
        Domega((-1, 1), (-1, 1), (1, 1), (1, 1)),
        Domega((1, 2), (-5, 4), (1, 2), (-27, 4)),
        Domega((10, 11), (12, 13), (14, 15), (16, 17)),
        Domega.from_ring(r.Zsqrt2(0, 2)),
        Domega.from_ring(r.Zsqrt2(2, 8)),
        Domega.from_ring(r.Zomega(4, 8, 6, 12)),
    ],
)
def test_sde(n):
    """Test the sde method of Domega."""
    factor = lambda sde: SQRT2_DOMEGA**sde if sde >= 0 else INV_DOMEGA ** (-sde)
    sde = n.sde()
    if n == 0:
        assert sde == -math.inf
    else:
        assert (n * factor(sde)).is_zomega
        assert not (n * factor(sde - 1)).is_zomega
        assert not (n * factor(sde - 2)).is_zomega


@pytest.mark.parametrize(
    "n",
    [
        (Domega((1, 1), (1, 1), (1, 1), (1, 1)), "1/2^1ω3 + 1/2^1ω2 + 1/2^1ω + 1/2^1"),
        (Domega((0, 0), (0, 0), (0, 0), (0, 0)), "0/2^0ω3 + 0/2^0ω2 + 0/2^0ω + 0/2^0"),
        (Domega((-1, 1), (-1, 2), (1, 3), (1, 4)), "-1/2^1ω3 - 1/2^2ω2 + 1/2^3ω + 1/2^4"),
        (Domega((1, 2), (-5, 4), (1, 2), (-27, 5)), "1/2^2ω3 - 5/2^4ω2 + 1/2^2ω - 27/2^5"),
    ],
)
def test_repr(n):
    """Test the repr method of Domega."""
    assert repr(n[0]) == n[1]


def test_addition_type_errors():
    """Test the type errors of the addition method of the class Domega."""
    with pytest.raises(TypeError, match="Summation operation is not defined between Domega and"):
        Domega((1, 1), (1, 1), (1, 1), (1, 1)) + 1.0j
    with pytest.raises(TypeError, match="Summation operation is not defined between Domega and"):
        Domega((1, 1), (1, 1), (1, 1), (1, 1)) + 1.0


def test_subtraction_type_errors():
    """Test the type errors of the subtraction method of the class Domega."""
    with pytest.raises(TypeError, match="Subtraction operation is not defined between Domega and"):
        Domega((1, 1), (1, 1), (1, 1), (1, 1)) - 1.0j
    with pytest.raises(TypeError, match="Subtraction operation is not defined between Domega and"):
        Domega((1, 1), (1, 1), (1, 1), (1, 1)) - 1.0


def test_multiplication_type_errors():
    """Test the type errors of the multiplication method of the class Domega."""
    with pytest.raises(TypeError, match="Product operation is not defined between Domega and"):
        Domega((1, 1), (1, 1), (1, 1), (1, 1)) * 1.0j
    with pytest.raises(TypeError, match="Product operation is not defined between Domega and"):
        Domega((1, 1), (1, 1), (1, 1), (1, 1)) * 1.0


def test_power_type_errors():
    """Test the type errors of the power method of the class Domega."""
    with pytest.raises(TypeError, match="Exponent must be an integer, but received"):
        Domega((1, 1), (1, 1), (1, 1), (1, 1)) ** 1.0


def test_power_value_errors():
    """Test the value errors of the power method of the class Domega."""
    with pytest.raises(ValueError, match="Expected exponent to be a positive integer"):
        Domega((1, 1), (1, 1), (1, 1), (1, 1)) ** -1


@pytest.mark.parametrize(
    "n",
    [
        (-25, Domega((0, 0), (0, 0), (0, 0), (-25, 0))),
        (12 - 5.0j, Domega((0, 0), (-5, 0), (0, 0), (12, 0))),
        (r.D(5, 13), Domega((0, 0), (0, 0), (0, 0), (5, 13))),
        (r.Zsqrt2(-33, 17), Domega((-17, 0), (0, 0), (17, 0), (-33, 0))),
        (Domega((1, 2), (3, 4), (5, 6), (7, 8)), Domega((1, 2), (3, 4), (5, 6), (7, 8))),
        (r.Dsqrt2((-65, 25), (21, 13)), Domega((-21, 13), (0, 0), (21, 13), (-65, 25))),
        (r.Zomega(33, 12, -55, 23), Domega((33, 0), (12, 0), (-55, 0), (23, 0))),
    ],
)
def test_from_ring(n):
    """Test the from_ring method of the class Domega."""
    assert Domega.from_ring(n[0]) == n[1]


def test_get_item():
    """Test the get item method of a Domega instance."""
    n = Domega((1, 2), (3, 4), (-5, 6), (7, 8))
    assert n[0] == r.D(1, 2) and n[1] == r.D(3, 4) and n[2] == r.D(-5, 6) and n[3] == r.D(7, 8)


def test_from_ring_value_errors():
    """Test the value errors of the from_ring method of the class Domega."""
    with pytest.raises(ValueError, match="Cannot convert"):
        Domega.from_ring(1.0)
    with pytest.raises(ValueError, match="Cannot convert"):
        Domega.from_ring(1 + 1.5 * 1.0j)
    with pytest.raises(ValueError, match="Cannot convert"):
        Domega.from_ring("1")


def test_equality():
    """Test the equality of two Domega instances."""
    n = Domega((1, 2), (3, 4), (-5, 6), (7, 8))
    m = Domega((0, 0), (0, 0), (0, 0), (13, 4))
    p = Domega((0, 0), (0, 0), (0, 0), (13, 0))
    assert n == Domega((1, 2), (3, 4), (-5, 6), (7, 8))
    assert n != Domega((1, 2), (3, 4), (-5, 6), (7, 9))
    assert n == complex(n)
    assert m == r.D(13, 4)
    assert p == 13
    assert p == 13.0
    assert n != [1]


def test_is_dsqrt2():
    """Test the is_dsqrt2 method of Domega."""
    assert not Domega((1, 2), (3, 4), (-5, 6), (7, 8)).is_dsqrt2
    assert Domega.from_ring(r.Dsqrt2((1, 2), (3, 4))).is_dsqrt2


def test_is_zsqrt2():
    """Test the is_zsqrt2 method of Domega."""
    assert not Domega((1, 2), (3, 4), (-5, 6), (7, 8)).is_zsqrt2
    assert Domega.from_ring(r.Zsqrt2(1, 2)).is_zsqrt2


def test_is_d():
    """Test the is_d method of Domega."""
    assert not Domega((1, 2), (3, 4), (-5, 6), (7, 8)).is_d
    assert Domega.from_ring(r.D(1, 2)).is_d


def test_is_integer():
    """Test the is_integer method of Domega."""
    assert not Domega((1, 2), (3, 4), (-5, 6), (7, 8)).is_integer
    assert Domega.from_ring(12).is_integer


def test_is_zomega():
    """Test the is_zomega method of Domega."""
    assert not Domega((1, 2), (3, 4), (-5, 6), (7, 8)).is_zomega
    assert Domega.from_ring(r.Zomega(1, 2, 3, 4)).is_zomega
