"""Pytests for Dfloat and other tangent-linear methods."""
from forward_propagation import (Dfloat, sin, cos, tan, exp, log, sinh, cosh,
                                 tanh, asin, acos, atan, asinh, acosh, atanh)
import pytest
from numpy import allclose

x1 = Dfloat(2, 1)
x2 = Dfloat(3, 1)
x3 = Dfloat(3.5, 1)
x4 = Dfloat(2, 0.5)


def test_type_error():
    """Test for type error."""
    with pytest.raises(TypeError):
        x1 + "frog"


@pytest.mark.parametrize(
    "f1, y1", (
        (x1, Dfloat(2, 1)),
        (x1 + x1, Dfloat(4, 2)),
        (x1 + 2, Dfloat(4, 1)),
        (x1 + 1 + x1, Dfloat(5, 2)),
        (x2 + 3 + x2, Dfloat(9, 2)),
        (x3 + 4.25 + x3, Dfloat(11.25, 2)),
        (x4 + 1 + x4, Dfloat(5, 1)),
        (x1 + x2 + x3 + x4, Dfloat(10.5, 3.5))
    )
)
def test_add(f1, y1):
    """Test addition on Dfloat."""
    assert allclose((f1.x, f1.dx), (y1.x, y1.dx))


@pytest.mark.parametrize(
    "f2, y2", (
        (x1 - 1, Dfloat(1, 1)),
        (1 - x2, Dfloat(-2, -1)),
        (x1 - x2, Dfloat(-1, 0)),
        (x3 - 1 - x4, Dfloat(0.5, 0.5)),
        (x1 - 1 + x2 - x4 + 2, Dfloat(4, 1.5))
    )
)
def test_sub(f2, y2):
    """Test subtraction on Dfloat."""
    assert allclose((f2.x, f2.dx), (y2.x, y2.dx))


@pytest.mark.parametrize(
    "f3, y3", (
        (x1 * 2, Dfloat(4, 2)),
        (3 * x2, Dfloat(9, 3)),
        (x1 * x4, Dfloat(4, 3)),
        (x3 * 2.5 * x4, Dfloat(17.5, 9.375)),
        ((x1 + 1) * x4 * (2 - x2), Dfloat(-6, -9.5))
    )
)
def test_mul(f3, y3):
    """Test multiplication on Dfloat."""
    assert allclose((f3.x, f3.dx), (y3.x, y3.dx))


@pytest.mark.parametrize(
    "f4, y4", (
        (x1 / 4, Dfloat(0.5, 0.25)),
        (3 / x2, Dfloat(1, -1/3)),
        (x3 / x4, Dfloat(1.75, 1/16)),
        ((x1 / 2) / x2, Dfloat(1/3, 1/18)),
        ((x1 + x2) / (x3 - x4), Dfloat(10/3, 2/9))
    )
)
def test_div(f4, y4):
    """Test division on Dfloat."""
    assert allclose((f4.x, f4.dx), (y4.x, y4.dx))


@pytest.mark.parametrize(
    "f5, y5", (
        (x1 ** 2, Dfloat(4, 4)),
        (3 ** x2, Dfloat(27, 27 * log(3))),
        (x3 ** x4, Dfloat(49/4, (49/4) * (0.5*log(3.5) + 4/7))),
        ((3 + x1) ** (4 - x2), Dfloat(5, -5 * log(5) + 1))
    )
)
def test_pow(f5, y5):
    """Test exponentiation on Dfloat."""
    assert allclose((f5.x, f5.dx), (y5.x, y5.dx))


@pytest.mark.parametrize(
    "f6, y6", (
        (sin(x2), Dfloat(sin(3), cos(3))),
        (sin(x1 * x3), Dfloat(sin(7), 5.5 * cos(7))),
        (sin(x4 + 1) * sin(2 + x1), Dfloat(sin(3) * sin(4), cos(4) * sin(3)
                                           + 0.5 * cos(3) * sin(4))),
        (cos(x3), Dfloat(cos(3.5), -1 * sin(3.5))),
        (cos(x1 / x2), Dfloat(cos(2/3), -1/9 * sin(2/3))),
        (cos(x4 ** 3) - cos(2 * x1), Dfloat(cos(8) - cos(4), -6 * sin(8)
                                            + 2 * sin(4))),
        (tan(x4), Dfloat(tan(2), 0.5 / cos(2)**2)),
        (tan(x2 ** x4), Dfloat(tan(9), (0.5 * log(3) + 2/3) * 9 / cos(9)**2)),
        (tan(x1 / 2) + tan(3 - x3), Dfloat(tan(1) + tan(-0.5), 0.5 / cos(1)**2
                                           - 1 / cos(-0.5)**2)),
        ((sin(x1 * x2 + 1) ** cos(2 / x4)) * tan(x3 ** 2),
         Dfloat((sin(7) ** cos(1)) * tan(49/4),
         ((sin(7) ** cos(1)) * 7 / cos(49/4)**2) +
         (1/4 * sin(1) * log(sin(7)) + 5 * cos(1) / tan(7)) *
         (sin(7) ** cos(1)) * tan(49/4)))
    )
)
def test_trig(f6, y6):
    """Test sin, cos and tan methods."""
    assert allclose((f6.x, f6.dx), (y6.x, y6.dx))
