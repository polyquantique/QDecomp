from cliffordplust.rings import Zsqrt2
from cliffordplust import rings as r
import pytest
import math

ZSQRT2 = math.sqrt(2)

"""Test the Zsqrt2 class."""

@pytest.mark.parametrize("n", [Zsqrt2(1, 1), Zsqrt2(0, 0), Zsqrt2(-1, 1), Zsqrt2(1, -1), Zsqrt2(-55, 14), Zsqrt2(36, 71), Zsqrt2(302, -675)])
def test_float(n):
    """Test the float value of the Zsqrt2 class."""
    assert math.isclose(n.a + n.b * ZSQRT2, float(n))

@pytest.mark.parametrize("n1", [Zsqrt2(1, 1), Zsqrt2(0, 0), Zsqrt2(-1, 1), Zsqrt2(1, -1), Zsqrt2(-55, 14), Zsqrt2(57, 71), Zsqrt2(302, -675), 0, 5, -20])
@pytest.mark.parametrize("n2", [Zsqrt2(1, 1), Zsqrt2(0, 0), Zsqrt2(-1, 1), Zsqrt2(1, -1), Zsqrt2(-55, 14), Zsqrt2(57, 71), Zsqrt2(302, -675)])
def test_addition(n1, n2):
    """Test the addition of two Zsqrt2 numbers."""
    n = n1
    n += n2
    assert math.isclose(float(n1 + n2), float(n1) + float(n2)) and math.isclose(float(n), float(n1) + float(n2))

@pytest.mark.parametrize("n1", [Zsqrt2(1, 1), Zsqrt2(0, 0), Zsqrt2(-1, 1), Zsqrt2(1, -1), Zsqrt2(-55, 14), Zsqrt2(57, 71), Zsqrt2(302, -675), 1])
@pytest.mark.parametrize("n2", [Zsqrt2(1, 1), Zsqrt2(0, 0), Zsqrt2(-1, 1), Zsqrt2(1, -1), Zsqrt2(-55, 14), Zsqrt2(57, 71), Zsqrt2(302, -675), 0, 5, -20])
def test_subtraction(n1, n2):
    """Test the subtraction of two Zsqrt2 numbers."""
    n = n1
    n -= n2
    assert math.isclose(float(n1 - n2), float(n1) - float(n2)) and math.isclose(float(n), float(n1) - float(n2))

@pytest.mark.parametrize("n1", [Zsqrt2(1, 1), Zsqrt2(0, 0), Zsqrt2(-1, 1), Zsqrt2(1, -1), Zsqrt2(-55, 14), Zsqrt2(57, 71), Zsqrt2(302, -675), 0, 5, -20])
@pytest.mark.parametrize("n2", [Zsqrt2(1, 1), Zsqrt2(0, 0), Zsqrt2(-1, 1), Zsqrt2(1, -1), Zsqrt2(-55, 14), Zsqrt2(57, 71), Zsqrt2(302, -675)])
def test_multiplication(n1, n2):
    """Test the multiplication of two Zsqrt2 numbers."""
    n = n1
    n *= n2
    assert math.isclose(float(n1 * n2), float(n1) * float(n2)) and math.isclose(float(n), float(n1) * float(n2))

@pytest.mark.parametrize("base", [Zsqrt2(1, 1), Zsqrt2(0, 0), Zsqrt2(-1, 1), Zsqrt2(1, -1), Zsqrt2(-55, 14), Zsqrt2(57, 71)])
@pytest.mark.parametrize("exponent", [0, 3, 5, 7, 10, 20])
def test_power(base, exponent):
    """Test the power of a Zsqrt2 number."""
    n = base
    n **= exponent
    assert math.isclose(float(base ** exponent), float(base) ** exponent) and math.isclose(float(n), float(base) ** exponent)

def test_equal():
    """Test the equality of two Zsqrt2 numbers."""
    n1 = Zsqrt2(1, 1)
    n2 = Zsqrt2(-1, 2)
    n3 = Zsqrt2(-2, 0)
    assert n1 == Zsqrt2(1, 1) and n2 == Zsqrt2(-1, 2) and n1 != n2 and n3 == -2 and n1 == 1 + ZSQRT2 and n1 != [1]

def test_init_type_error():
    """Test the TypeError when the Zsqrt2 class is initialized with a non-integer."""
    with pytest.raises(TypeError, match="Expected inputs to be of type int"):
        Zsqrt2("1", 1)
    with pytest.raises(TypeError, match="Expected inputs to be of type int"):
        Zsqrt2(1, 1.0)

def test_sqrt2_conjugate():
    """Test the conjugate method of the Zsqrt2 class."""
    n1 = Zsqrt2(1, -7)
    n2 = Zsqrt2(5, 3)
    assert n1.sqrt2_conjugate() == Zsqrt2(1, 7) and n2.sqrt2_conjugate() == Zsqrt2(5, -3)

def test_get_item():
    """Test the get item method of the Zsqrt2 class."""
    n = Zsqrt2(1, 2)
    assert n[0] == 1 and n[1] == 2

@pytest.mark.parametrize("nb", [
        (Zsqrt2(1, 1), "1+1√2"),
        (Zsqrt2(1, -1), "1-1√2"),
        (Zsqrt2(1, -2), "1-2√2"),
        (Zsqrt2(0, -1), "0-1√2"),
        (Zsqrt2(1, 0), "1+0√2"),
        (Zsqrt2(-1, 0), "-1+0√2"),
        (Zsqrt2(0, 0), "0+0√2"),
    ])
def test_repr(nb):
    """Test the string representation of the Zsqrt2 class""" 
    assert str(nb[0]) == nb[1]

def test_summation_type_error():
    """Test the TypeError when the Zsqrt2 class is summed with the wrong type."""
    n = Zsqrt2(1, 1)
    with pytest.raises(TypeError, match="Summation operation is not defined with"):
        n + "1"
    with pytest.raises(TypeError, match="Summation operation is not defined with"):
        n + 1.0

def test_subtraction_type_error():
    """Test the TypeError when the Zsqrt2 class is subtracted with the wrong type."""
    n = Zsqrt2(1, 1)
    with pytest.raises(TypeError, match="Subtraction operation is not defined with"):
        n - "1"
    with pytest.raises(TypeError, match="Subtraction operation is not defined with"):
        n - 1.0

def test_multiplication_type_error():
    """Test the TypeError when the Zsqrt2 class is multiplied with the wrong type."""
    n = Zsqrt2(1, 1)
    with pytest.raises(TypeError, match="Multiplication operation is not defined with"):
        n * "1"
    with pytest.raises(TypeError, match="Multiplication operation is not defined with"):
        n * 1.0

def test_power_type_error():
    """Test the TypeError when the Zsqrt2 class is powered with a non-integer."""
    n = Zsqrt2(1, 1)
    with pytest.raises(TypeError, match="Expected power to be of type int"):
        n ** "1"
    with pytest.raises(TypeError, match="Expected power to be of type int"):
        n ** 1.0

def test_power_value_error():
    """Test the ValueError when the Zsqrt2 class is powered with a negative integer."""
    n = Zsqrt2(1, 1)
    with pytest.raises(ValueError, match="Expected power to be a positive integer"):
        n ** -1

def test_from_ring():
    """Test the from_ring method of the Zsqrt2 class."""
    assert Zsqrt2.from_ring(-15) == Zsqrt2(-15, 0)
    assert Zsqrt2.from_ring(r.D(6, 1)) == Zsqrt2(3, 0)
    assert Zsqrt2.from_ring(Zsqrt2(1, 1)) == Zsqrt2(1, 1)
    n = Zsqrt2.from_ring(r.Dsqrt2((5, 0), (-41, 0)))
    assert n == Zsqrt2(5, -41)
    n = Zsqrt2.from_ring(r.Zomega(21, 0, -21, -56))
    assert n == Zsqrt2(-56, -21)
    n = Zsqrt2.from_ring(r.Domega((21, 0), (0, 0), (-21, 0), (56, 0)))
    assert n == Zsqrt2(56, -21)

def test_from_ring_value_error():
    """Test the ValueError when the from_ring method cannot perform the conversion."""
    from cliffordplust import rings as r
    with pytest.raises(ValueError, match="Cannot convert"):
        Zsqrt2.from_ring(1.0)
    with pytest.raises(ValueError, match="Cannot convert"):
        Zsqrt2.from_ring(r.D(1, 1))
    with pytest.raises(ValueError, match="Cannot convert"):
        Zsqrt2.from_ring(r.Dsqrt2((1, 0), (1, 1)))
    with pytest.raises(ValueError, match="Cannot convert"):
        Zsqrt2.from_ring(r.Zomega(1, 1, 1, 1))
    with pytest.raises(ValueError, match="Cannot convert"):
        Zsqrt2.from_ring(r.Domega((1, 0), (1, 0), (1, 0), (1, 0)))

def test_is_integer():
    """Test the is_integer method of the Zsqrt2 class."""
    assert Zsqrt2(1, 0).is_integer == True
    assert Zsqrt2(1, 1).is_integer == False
