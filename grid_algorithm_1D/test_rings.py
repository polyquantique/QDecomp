import pytest

from Rings import D, Domega
from random import randint
import math

@pytest.mark.parametrize("num", [randint(-500, 500) for _ in range(15)] + [0])
@pytest.mark.parametrize("denom", [randint(0, 50) for _ in range(15)] + [0])
def test_float_D(num: int, denom: int) -> None:
    """Test the float value of the D class."""
    assert math.isclose(num / 2**denom, float(D(num, denom)))

@pytest.mark.parametrize("n1", [D(randint(-500, 500), randint(0, 50)) for _ in range(15)] + [D(0, 0)] + [randint(-500, 500) for _ in range(5)])
@pytest.mark.parametrize("n2", [D(randint(-500, 500), randint(0, 50)) for _ in range(15)] + [D(0, 0)] + [randint(-500, 500) for _ in range(5)])
def test_arithmetic_D(n1: D | int, n2: D | int) -> None:
    """Test the +, - and * arithmetic operations for the D class."""
    sum = n1 + n2
    prod = n1 * n2
    sub = n1 - n2
    i_sum = n1
    i_sum+=n2
    i_prod = n1
    i_prod *= n2
    i_sub = n1
    i_sub -= n2
    assert math.isclose(float(sum), float(n1) + float(n2)) and math.isclose(float(sum), float(i_sum)) and isinstance(sum, (D, int))
    assert math.isclose(float(prod), float(n1) * float(n2)) and math.isclose(float(prod), float(i_prod)) and isinstance(prod, (D, int))
    assert math.isclose(float(sub), float(n1) - float(n2)) and math.isclose(float(sub), float(i_sub)) and isinstance(sub, (D, int))


@pytest.mark.parametrize("base", [D(randint(-500, 500), randint(0, 20)) for _ in range(15)] + [D(0, 0)])
@pytest.mark.parametrize("exponent", [randint(0, 30) for _ in range(10)])
def test_power_D(base: D, exponent: int) -> None:
    """Test the power operation for the D class."""
    result = base**exponent
    assert math.isclose(float(result), float(base) ** exponent) and isinstance(result, D)


@pytest.mark.parametrize("coeff", [[D(randint(-500, 500), randint(0, 50)) for _ in range(4)] for _ in range(15)])
def test_real_imag_domega(coeff) -> None:
    """ Test the real and complex value of the class."""
    ring_element: Domega = Domega(coeff[0], coeff[1], coeff[2], coeff[3])
    omega: complex = (1 + 1.j) / math.sqrt(2)
    complex_value: complex = float(ring_element.a) * omega**3 + float(ring_element.b) * omega**2 + float(ring_element.c) * omega + float(ring_element.d)
    assert math.isclose(ring_element.real(), complex_value.real) and math.isclose(ring_element.imag(), complex_value.imag)
    assert math.isclose(complex(ring_element).real, complex_value.real) and math.isclose(complex(ring_element).imag, complex_value.imag)

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
    ] + [Domega((0, 0), (0, 0), (0, 0), (0, 0))] + [randint(-500, 500) for _ in range(5)]
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
    ] + [Domega((0, 0), (0, 0), (0, 0), (0, 0))] + [randint(-500, 500) for _ in range(5)]
)
def test_domega_arithmetic(n1: Domega | int, n2: Domega | int):
    """Test the +, - and * operation of the Domega class."""
    sum = n1 + n2
    prod = n1 * n2
    sub = n1 - n2
    i_sum = i_prod = i_sub = n1
    i_sum += n2
    i_prod *= n2
    i_sub -= n2
    assert math.isclose(
        complex(sum).real, complex(n1).real + complex(n2).real
    ) and math.isclose(complex(sum).imag, complex(n1).imag + complex(n2).imag) and isinstance(sum, (Domega, int)) and complex(sum) == complex(i_sum) 
    assert math.isclose(
        complex(sub).real, complex(n1).real - complex(n2).real
    ) and math.isclose(complex(sub).imag, complex(n1).imag - complex(n2).imag) and isinstance(sub, (Domega, int)) and complex(sub) == complex(i_sub)
    assert math.isclose(
        complex(prod).real, (complex(n1) * complex(n2)).real
    ) and math.isclose(complex(prod).imag, (complex(n1) * complex(n2)).imag) and isinstance(prod, (Domega, int)) and complex(prod) == complex(i_prod)
