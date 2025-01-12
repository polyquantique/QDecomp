from __future__ import annotations

import math
from random import randint, uniform

import pytest

from Rings import *
from Rings import _output_type


@pytest.mark.parametrize("not_int", [1.0, [1], 1 + 1.0j, D(1, 0), (1,), "1"])
def test_D_init_errors(not_int) -> None:
    """Test the raise of an error if the class arguments are not integers and if the denominator is not positive."""
    with pytest.raises(TypeError, match="Class arguments must be of type int but received"):
        D(not_int, 0)
    with pytest.raises(TypeError, match="Class arguments must be of type int"):
        D(0, not_int)
    with pytest.raises(ValueError, match="Denominator exponent must be positive"):
        D(0, -1)


@pytest.mark.parametrize("num", [randint(-500, 500) for _ in range(15)] + [0])
@pytest.mark.parametrize("denom", [randint(0, 50) for _ in range(15)] + [0])
def test_float_D(num: int, denom: int) -> None:
    """Test the float value of the D class."""
    assert math.isclose(num / 2**denom, float(D(num, denom)))


@pytest.mark.parametrize(
    "n1",
    [D(randint(-500, 500), randint(0, 50)) for _ in range(10)]
    + [randint(-50, 500) for _ in range(5)]
    + [uniform(-500, 500) for _ in range(5)],
)
@pytest.mark.parametrize("n2", [D(randint(-500, 500), randint(0, 50)) for _ in range(10)])
def test_comparaisons_D(n1, n2) -> None:
    """Test the different comparaisons of D objects."""
    assert (
        n2 == n2
        and float(n2) == n2
        and (n1 == n2) == (float(n1) == float(n2))
        and (n1 != n2) == (float(n1) != float(n2))
    )
    assert n2 >= n2 and n2 <= n2 and float(n2) <= n2 and float(n2) >= n2
    assert (n1 >= n2) == (float(n1) >= float(n2)) and (n1 <= n2) == (float(n1) <= float(n2))
    assert (
        (n1 > n2) == (float(n1) > float(n2))
        and (n1 < n2) == (float(n1) < float(n2))
        and (n1 > n2) != (n1 < n2)
    )


@pytest.mark.parametrize(
    "n1",
    [D(randint(-500, 500), randint(0, 50)) for _ in range(15)]
    + [D(0, 0)]
    + [randint(-500, 500) for _ in range(5)],
)
@pytest.mark.parametrize(
    "n2",
    [D(randint(-500, 500), randint(0, 50)) for _ in range(15)]
    + [D(0, 0)]
    + [randint(-500, 500) for _ in range(5)],
)
def test_arithmetic_D(n1: D | int, n2: D | int) -> None:
    """Test the +, - and * arithmetic operations for the D class."""
    sum = n1 + n2
    prod = n1 * n2
    sub = n1 - n2
    i_sum = n1
    i_sum += n2
    i_prod = n1
    i_prod *= n2
    i_sub = n1
    i_sub -= n2
    assert (
        math.isclose(float(sum), float(n1) + float(n2))
        and math.isclose(float(sum), float(i_sum))
        and isinstance(sum, (D, int))
    )
    assert (
        math.isclose(float(prod), float(n1) * float(n2))
        and math.isclose(float(prod), float(i_prod))
        and isinstance(prod, (D, int))
    )
    assert (
        math.isclose(float(sub), float(n1) - float(n2))
        and math.isclose(float(sub), float(i_sub))
        and isinstance(sub, (D, int))
    )


@pytest.mark.parametrize(
    "wrong_type", [Domega((1, 0), (1, 0), (1, 0), (1, 0)), 1.0, 1.0j, (1,), [1], "1"]
)
def test_arithmetic_operations_type_errors(wrong_type):
    """Test the raise of a type_error when doing an arithmetic operation with objects that are not integers or D objects."""
    ring_element = D(1, 0)
    with pytest.raises(TypeError, match="Summation operation is not defined between D"):
        ring_element + wrong_type
    with pytest.raises(TypeError, match="Subtraction operation is not defined between D"):
        ring_element - wrong_type
    with pytest.raises(TypeError, match="Product operation is not defined between D"):
        ring_element * wrong_type


@pytest.mark.parametrize(
    "base", [D(randint(-500, 500), randint(0, 20)) for _ in range(15)] + [D(0, 0)]
)
@pytest.mark.parametrize("exponent", [randint(0, 30) for _ in range(10)])
def test_power_D(base: D, exponent: int) -> None:
    """Test the power operation for the D class."""
    result = base**exponent
    assert math.isclose(float(result), float(base) ** exponent) and isinstance(result, D)


def test_power_errors() -> None:
    """Test the raise of a TypeError when the power is not an integer and a ValueError if the power is negative."""
    ring_element = D(1, 0)
    with pytest.raises(TypeError, match="Expected power to be of type int"):
        ring_element**1.0
    with pytest.raises(ValueError, match="Expected power to be a positive integer"):
        ring_element ** (-1)


@pytest.mark.parametrize("n", [D(randint(-500, 500), randint(0, 50)) for _ in range(10)])
def test_rounding_D(n: D) -> None:
    """Test the rounding of the D class."""
    assert (
        round(n) == round(float(n))
        and math.ceil(n) == math.ceil(float(n))
        and math.floor(n) == math.floor(float(n))
    )


@pytest.mark.parametrize(
    "coeff", [[D(randint(-500, 500), randint(0, 50)) for _ in range(4)] for _ in range(15)]
)
def test_real_imag_domega(coeff) -> None:
    """Test the real and complex value of the Domega class."""
    ring_element: Domega = Domega(coeff[0], coeff[1], coeff[2], coeff[3])
    omega: complex = (1 + 1.0j) / math.sqrt(2)
    complex_value: complex = (
        float(ring_element.a) * omega**3
        + float(ring_element.b) * omega**2
        + float(ring_element.c) * omega
        + float(ring_element.d)
    )
    assert math.isclose(ring_element.real(), complex_value.real) and math.isclose(
        ring_element.imag(), complex_value.imag
    )
    assert math.isclose(complex(ring_element).real, complex_value.real) and math.isclose(
        complex(ring_element).imag, complex_value.imag
    )


@pytest.mark.parametrize(
    "n1",
    [
        Domega(
            (randint(-100, 100), randint(0, 40)),
            (randint(-100, 100), randint(0, 40)),
            (randint(-100, 100), randint(0, 40)),
            (randint(-100, 100), randint(0, 40)),
        )
        for _ in range(10)
    ]
    + [Domega((0, 0), (0, 0), (0, 0), (0, 0))]
    + [randint(-500, 500) for _ in range(5)],
)
@pytest.mark.parametrize(
    "n2",
    [
        Domega(
            (randint(-100, 100), randint(0, 40)),
            (randint(-100, 100), randint(0, 40)),
            (randint(-100, 100), randint(0, 40)),
            (randint(-100, 100), randint(0, 40)),
        )
        for _ in range(10)
    ]
    + [Domega((0, 0), (0, 0), (0, 0), (0, 0))]
    + [randint(-500, 500) for _ in range(5)],
)
def test_domega_arithmetic(n1: Domega | int, n2: Domega | int) -> None:
    """Test the +, - and * operation of the Domega class."""
    sum = n1 + n2
    prod = n1 * n2
    sub = n1 - n2
    i_sum = i_prod = i_sub = n1
    i_sum += n2
    i_prod *= n2
    i_sub -= n2
    assert (
        math.isclose(complex(sum).real, complex(n1).real + complex(n2).real)
        and math.isclose(complex(sum).imag, complex(n1).imag + complex(n2).imag)
        and isinstance(sum, (Domega, int))
        and complex(sum) == complex(i_sum)
    )
    assert (
        math.isclose(complex(sub).real, complex(n1).real - complex(n2).real)
        and math.isclose(complex(sub).imag, complex(n1).imag - complex(n2).imag)
        and isinstance(sub, (Domega, int))
        and complex(sub) == complex(i_sub)
    )
    assert (
        math.isclose(complex(prod).real, (complex(n1) * complex(n2)).real)
        and math.isclose(complex(prod).imag, (complex(n1) * complex(n2)).imag)
        and isinstance(prod, (Domega, int))
        and complex(prod) == complex(i_prod)
    )


@pytest.mark.parametrize(
    "base",
    [
        Domega(
            (randint(-50, 50), randint(0, 15)),
            (randint(-50, 50), randint(0, 15)),
            (randint(-50, 50), randint(0, 15)),
            (randint(-50, 50), randint(0, 15)),
        )
        for _ in range(10)
    ]
    + [Domega((0, 0), (0, 0), (0, 0), (0, 0))],
)
@pytest.mark.parametrize("exponent", [randint(0, 15) for _ in range(5)] + [0])
def test_domega_power(base, exponent) -> None:
    """Test the power operation for the Domeg clase."""
    result = base**exponent
    i_result = base
    i_result **= exponent
    assert (
        isinstance(result, Domega)
        and math.isclose(complex(result).real, (complex(base) ** exponent).real)
        and math.isclose(complex(result).imag, (complex(base) ** exponent).imag)
        and result == i_result
    )


@pytest.mark.parametrize(
    "n",
    [
        Domega(
            (randint(-50, 50), randint(0, 25)),
            (randint(-50, 50), randint(0, 25)),
            (randint(-50, 50), randint(0, 25)),
            (randint(-50, 50), randint(0, 25)),
        )
        for _ in range(5)
    ]
    + [
        Dsqrt2((randint(-50, 50), randint(0, 25)), (randint(-50, 50), randint(0, 25)))
        for _ in range(5)
    ]
    + [Zsqrt2(randint(-100, 100), randint(0, 25)) for _ in range(5)]
    + [Zsqrt2(0, 0)]
    + [Zomega(4, 6, 8, 12)],
)
def test_sde_domega(n: Domega) -> None:
    # Test the calculation of smallest denominator exponent.
    sqrt2 = Zsqrt2(0, 1)
    inv_sqrt2 = Dsqrt2((0, 0), (1, 1))
    sde = n.sde()
    if n == 0:
        assert sde == -math.inf
    elif sde < 0:
        assert (n * inv_sqrt2 ** abs(sde))._is_Zomega and not (
            n * inv_sqrt2 ** (abs(sde) + 1)
        )._is_Zomega
    elif sde > 0:
        assert (n * sqrt2**sde)._is_Zomega and not (n * sqrt2 ** (sde - 1))._is_Zomega
    else:
        assert not (n * inv_sqrt2)._is_Zomega


@pytest.mark.parametrize(
    "n",
    [
        Domega(
            (randint(-100, 100), randint(0, 40)),
            (randint(-100, 100), randint(0, 40)),
            (randint(-100, 100), randint(0, 40)),
            (randint(-100, 100), randint(0, 40)),
        )
        for _ in range(10)
    ]
    + [
        Dsqrt2((randint(-100, 100), randint(0, 40)), (randint(-100, 100), randint(0, 40)))
        for _ in range(5)
    ]
    + [Zsqrt2(randint(-100, 100), randint(-100, 100)) for _ in range(5)],
)
def test_conjugation_domega(n: Domega) -> None:
    """Test the sqrt(2) conjugation and the complex conjugation of the Domega class."""
    c_conjugate = n.complex_conjugate()
    sqrt_conjugate = n.sqrt2_conjugate()
    assert complex(c_conjugate).conjugate() == complex(n) and (n * c_conjugate).imag() == 0
    if isinstance(n, Domega):
        assert (n * c_conjugate)._is_Dsqrt2
    if isinstance(n, Dsqrt2):
        assert (sqrt_conjugate * n)._is_D
    if isinstance(n, Zsqrt2):
        assert (sqrt_conjugate * n)._is_integer


def test_domega_init_error() -> None:
    "Test the raise of errors when initializing Domega."
    with pytest.raises(TypeError, match="Class arguments must be of type tuple"):
        Domega((1, 1), (1, 1), (1, 1), [1, 1])
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Domega((1, 1), (1, 1), (1.0, 1), (1, 1))
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Domega((1, 1), (1, 1), (1, 1, 3), (1, 1))
    with pytest.raises(ValueError, match="Denominator exponent must be positive"):
        Domega((1, 1), (1, 1), (1, 1), (1, -1))


def test_get_item_domega() -> None:
    """Test of the getitem method fot Domega."""
    a = (randint(-100, 100), randint(0, 100))
    b = (randint(-100, 100), randint(0, 100))
    c = (randint(-100, 100), randint(0, 100))
    d = (randint(-100, 100), randint(0, 100))
    ring_element = Domega(a, b, c, d)
    assert (
        ring_element[0] == D(*a)
        and ring_element[1] == D(*b)
        and ring_element[2] == D(*c)
        and ring_element[3] == D(*d)
    )
    for index, coeff in enumerate(ring_element):
        assert coeff == (D(*a), D(*b), D(*c), D(*d))[index]


def test_equality_domega() -> None:
    """Test the comparaison of Domega objects."""
    domega_element = Domega(
        (randint(-100, 100), randint(0, 100)),
        (randint(-100, 100), randint(0, 100)),
        (randint(-100, 100), randint(0, 100)),
        (randint(-100, 100), randint(0, 100)),
    )
    zomega_element = Zomega(
        randint(-100, 100), randint(-100, 100), randint(-100, 100), randint(-100, 100)
    )
    assert domega_element == domega_element and zomega_element == zomega_element
    assert not (domega_element != domega_element) and not (zomega_element != zomega_element)
    assert zomega_element == zomega_element.convert(Domega)
    assert Domega((0, 0), (0, 0), (0, 0), (1, 1)) == D(1, 1) and D(1, 1) == Domega(
        (0, 0), (0, 0), (0, 0), (1, 1)
    )
    assert Domega((0, 0), (1, 0), (0, 0), (1, 0)) == complex(
        Domega((0, 0), (1, 0), (0, 0), (1, 0))
    ) and not Domega((0, 0), (1, 0), (0, 0), (1, 0)) == complex(
        Domega((0, 0), (1, 0), (0, 0), (1, 1))
    )
    assert Zsqrt2(0, 0) == 0 and Domega((0, 0), (0, 0), (0, 0), (0, 0)) == 0
    with pytest.raises(TypeError, match="Comparaison between"):
        domega_element == "1"


@pytest.mark.parametrize("wrong_type", [(1,), 1.0, 1.0j, "1", [1]])
def test_arithmetic_errors_domega(wrong_type) -> None:
    """Test the raise of TypeErrors when doing arithmetic operations with wring types."""
    ring_element = Domega((0, 0), (0, 0), (0, 0), (0, 0))
    with pytest.raises(TypeError, match="Summation operation is not defined between"):
        ring_element + wrong_type
    with pytest.raises(TypeError, match="Subtraction operation is not defined between"):
        ring_element - wrong_type
    with pytest.raises(TypeError, match="Product operation is not defined between"):
        ring_element * wrong_type


def test_power_error_domega() -> None:
    """Test the raise of TypeError if the exponent is not integer and the raise of ValueError if the exponent is negative."""
    ring_element = Domega((1, 1), (1, 1), (1, 1), (1, 1))
    with pytest.raises(TypeError, match="Exponent must be an integer"):
        ring_element**1.0
    with pytest.raises(ValueError, match="Expected exponent to be a positive integer"):
        ring_element ** (-1)


def test_operation_output_type():
    """Test the output_type when doing arithmetic operations with ring element."""
    domega = Domega((1, 1), (1, 1), (1, 1), (1, 1))
    zomega = Zomega(1, 1, 1, 1)
    dsqrt2 = Dsqrt2((1, 1), (1, 1))
    zsqrt2 = Zsqrt2(1, 1)
    d = D(1, 1)
    integer = 1
    assert all(
        [
            type(result) is Domega
            for result in map(domega.__add__, (domega, zomega, dsqrt2, zsqrt2, d, integer))
        ]
    )
    assert all(
        [
            type(result) is Domega
            for result in map(domega.__mul__, (domega, zomega, dsqrt2, zsqrt2, d, integer))
        ]
    )
    assert all(
        [
            type(result) is Domega
            for result in map(domega.__sub__, (domega, zomega, dsqrt2, zsqrt2, d, integer))
        ]
    )
    assert all(
        [type(result) is Zomega for result in map(zomega.__add__, (zomega, zsqrt2, integer))]
    )
    assert all(
        [type(result) is Zomega for result in map(zomega.__mul__, (zomega, zsqrt2, integer))]
    )
    assert all(
        [type(result) is Zomega for result in map(zomega.__sub__, (zomega, zsqrt2, integer))]
    )
    assert all(
        [type(result) is Dsqrt2 for result in map(dsqrt2.__add__, (dsqrt2, zsqrt2, d, integer))]
    )
    assert all(
        [type(result) is Dsqrt2 for result in map(dsqrt2.__mul__, (dsqrt2, zsqrt2, d, integer))]
    )
    assert all(
        [type(result) is Dsqrt2 for result in map(dsqrt2.__sub__, (dsqrt2, zsqrt2, d, integer))]
    )
    assert all([type(result) is Zsqrt2 for result in map(zsqrt2.__add__, (zsqrt2, integer))])
    assert all([type(result) is Zsqrt2 for result in map(zsqrt2.__mul__, (zsqrt2, integer))])
    assert all([type(result) is Zsqrt2 for result in map(zsqrt2.__sub__, (zsqrt2, integer))])
    assert all([type(result) is Dsqrt2 for result in [zsqrt2 + d, zsqrt2 * d, zsqrt2 - d]])
    assert all(
        [
            type(result) is Domega
            for result in [
                zomega + d,
                zomega * d,
                zomega - d,
                zomega + dsqrt2,
                zomega * dsqrt2,
                zomega - dsqrt2,
            ]
        ]
    )


@pytest.mark.parametrize("type", (str, int, float, complex, list))
def test_output_type_error(type) -> None:
    "Test the raise of ValueError if the argument of the output_type function not valid."
    with pytest.raises(ValueError, match="Conversion between"):
        _output_type(type)


@pytest.mark.parametrize(
    "nb",
    [
        (D(0, 10), str(0)),
        (D(1, 1), "1/2"),
        (D(1, 2), "1/2^2"),
        (D(-1, 1), "-1/2"),
        (D(-1, 2), "-1/2^2"),
    ],
)
def test_d_str_representation(nb) -> None:
    """Test the sring representation of D objects."""
    assert str(nb[0]) == nb[1]


def test_d_absolute_value():
    """Test the abs method for the D class."""
    d_neg = D(-5, 1)
    d_pos = D(5, 1)
    assert float(abs(d_neg)) == abs(float(d_neg)) and abs(d_neg) == d_pos and abs(d_pos) == d_pos


def test_ring_conversion_errors() -> None:
    """Test the raise of TypeErrors when the conversion target is wrong or when the conversion is not possible."""
    ring_element = Domega((1, 1), (1, 1), (1, 1), (1, 1))
    assert type((Domega((0, 0), (0, 0), (0, 0), (1, 1))).convert(D)) is D
    with pytest.raises(TypeError, match="Target must be a type or a string, but received"):
        ring_element.convert(1)
    with pytest.raises(TypeError, match="Target must be a type or a string, but received"):
        ring_element.convert(["Domega"])
    with pytest.raises(TypeError, match="Could not convert the ring element from"):
        assert not ring_element._is_Zomega
        ring_element.convert(Zomega)
    with pytest.raises(TypeError, match="Could not convert the ring element from"):
        assert not ring_element._is_Dsqrt2
        ring_element.convert(Dsqrt2)
    with pytest.raises(TypeError, match="Could not convert the ring element from"):
        assert not ring_element._is_Zsqrt2
        ring_element.convert(Zsqrt2)
    with pytest.raises(TypeError, match="Could not convert the ring element from"):
        assert not ring_element._is_D
        ring_element.convert(D)
    with pytest.raises(TypeError, match="is not a valid target"):
        ring_element.convert(int)


@pytest.mark.parametrize(
    "nb",
    [
        (Domega((0, 0), (0, 0), (0, 0), (1, 1)), "1/2"),
        (Domega((0, 0), (0, 0), (0, 0), (0, 0)), "0"),
        (Domega((1, 1), (1, 1), (1, 1), (1, 1)), "1/2\u03C93 + 1/2\u03C92 + 1/2\u03C91 + 1/2"),
        (Domega((-1, 1), (1, 1), (-1, 1), (1, 1)), "- 1/2\u03C93 + 1/2\u03C92 - 1/2\u03C91 + 1/2"),
        (Domega((1, 0), (1, 0), (1, 0), (1, 0)), "\u03C93 + \u03C92 + \u03C91 + 1"),
        (Domega((0, 0), (-1, 0), (0, 0), (1, 0)), "- \u03C92 + 1"),
        (Zomega(1, 1, 1, 1), "\u03C93 + \u03C92 + \u03C91 + 1"),
        (Zomega(-1, -1, -1, -1), "- \u03C93 - \u03C92 - \u03C91 - 1"),
        (Dsqrt2((1, 1), (1, 1)), "1/2 + 1/2\u221a2"),
        (Dsqrt2((-1, 1), (-1, 1)), "-1/2 - 1/2\u221a2"),
        (Dsqrt2((1, 0), (1, 0)), "1 + \u221a2"),
        (Dsqrt2((1, 0), (-1, 0)), "1 - \u221a2"),
        (Dsqrt2((0, 0), (0, 0)), "0"),
        (Zsqrt2(1, 1), "1 + \u221a2"),
    ],
)
def test_domega_str_representation(nb) -> None:
    """Test the string representation of the Domega, Zomega, Dsqrt2 and Zsqrt2 class."""
    assert str(nb[0]) == nb[1]


def test_zomega_init_error() -> None:
    """Test the raise of a TypeError when initializing Zomega with non integer values"""
    with pytest.raises(TypeError, match="Class arguments must be integers"):
        Zomega(1.0, 1, 1, 1)
    with pytest.raises(TypeError, match="Class arguments must be integers"):
        Zomega(1, 1, 1, "1")


def test_dsqrt2_init_errors() -> None:
    """Test the raise of TypeErrors and ValueErrors when initializing Dsqrt2 with wrong arguments."""
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Dsqrt2((1, 1, 1), (1, 1))
    with pytest.raises(TypeError, match="Tuples must take two integer values"):
        Dsqrt2((1, 1), (1, "1"))
    with pytest.raises(ValueError, match="Denominator value must be positive but got"):
        Dsqrt2((1, 1), (1, -1))
    with pytest.raises(TypeError, match="Class arguments must be of type tuple"):
        Dsqrt2([1, 1], (1, 1))
    with pytest.raises(TypeError, match="Class arguments must be of type tuple"):
        Dsqrt2((1, 1), "12")


def test_zsqrt2_init_errors() -> None:
    """Test the raise of a TypeError when initializing Zsqr2 with non-integers"""
    with pytest.raises(TypeError, match="Expected class inputs to be of type int, but got"):
        Zsqrt2(1, 1.0)
    with pytest.raises(TypeError, match="Expected class inputs to be of type int, but got"):
        Zsqrt2("1", 1)


@pytest.mark.parametrize(
    "nb",
    [
        Dsqrt2((randint(-100, 100), randint(0, 30)), (randint(-100, 100), randint(0, 30)))
        for _ in range(10)
    ]
    + [Dsqrt2((0, 0), (0, 0))]
    + [Zsqrt2(randint(-100, 100), randint(-100, 100)) for _ in range(10)],
)
def test_float_dsqrt2_zsqrt2(nb: Dsqrt2 | Zsqrt2) -> None:
    """Test the float representation of the Dsqrt2 and Zsqrt2 class."""
    assert math.isclose(float(nb), float(nb.p) + float(nb.q) * math.sqrt(2))
