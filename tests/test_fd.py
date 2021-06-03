"""Pytests for Dfloat and other tangent-linear methods."""
from forward_propagation import (Dfloat, sin, cos, tan, exp, log, sinh, cosh,
                                 tanh, asin, acos, atan, asinh, acosh, atanh)
import math
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


@pytest.mark.parametrize(
    "f7, y7", (
        (exp(x3), Dfloat(exp(3.5), exp(3.5))),
        (exp(x1 * x4), Dfloat(exp(4), 3 * exp(4))),
        (exp(x2 ** x4) * exp(sin(x1 + 1)), Dfloat(exp(9) * exp(sin(3)),
         cos(3) * exp(sin(3)) * exp(9) + (0.5 * log(3) + 2/3) * 9 *
         exp(9) * exp(sin(3)))),
        (log(x3), Dfloat(log(3.5), 1/3.5)),
        (log(x2 / (x1 - 1)), Dfloat(log(3), -2/3)),
        (log(x4 * x1) / log(exp(x2)), Dfloat(log(4) / 3,
         (9 - 4 * log(4)) / 36))
    )
)
def test_exp(f7, y7):
    """Test exp and ln methods."""
    assert allclose((f7.x, f7.dx), (y7.x, y7.dx))


@pytest.mark.parametrize(
    "f8, y8", (
        (sinh(x3), Dfloat(sinh(3.5), cosh(3.5))),
        (sinh(2 / (x2 + x3)), Dfloat(sinh(4/13), (-16/169) * cosh(4/13))),
        (sinh(x1 / x2) * sinh(sin(x4)), Dfloat(sinh(2/3) * sinh(sin(2)),
         (1/9) * cosh(2/3) * sinh(sin(2)) + 0.5 * cos(2) * cosh(sin(2)) *
          sinh(2/3))),
        (cosh(x3), Dfloat(cosh(3.5), sinh(3.5))),
        (cosh(x1 ** x2), Dfloat(cosh(8), (8 * log(2) + 12) * sinh(8))),
        (cosh(x4 / x1) + cosh(exp(x2)), Dfloat(cosh(1) + cosh(exp(3)),
         exp(3) * sinh(exp(3)) - sinh(1))),
        (tanh(x3), Dfloat(tanh(3.5), 1 / cosh(3.5) ** 2)),
        (tanh(x2 + 1 + x3), Dfloat(tanh(7.5), 2 / cosh(7.5) ** 2)),
        (tanh(x1 * x2) * tanh(3 - x4), Dfloat(tanh(6) * tanh(1),
         (5 * tanh(1) / cosh(6) ** 2) - (0.5 * tanh(6) / cosh(1) ** 2))),
        (sinh(sin(x1)) * cosh(cos(x2)) * tanh(tan(x4)),
         Dfloat(sinh(sin(2)) * cosh(cos(3)) * tanh(tan(2)),
                (sinh(sin(2)) * cosh(cos(3))) /
                (2 * cosh(tan(2)) ** 2 * cos(2) ** 2) +
                (cos(2) * cosh(sin(2)) * cosh(cos(3)) -
                 sin(3) * sinh(cos(3)) * sinh(sin(2))) * tanh(tan(2))))
    )
)
def test_hyp(f8, y8):
    """Test sinh, cosh and tanh methods."""
    assert allclose((f8.x, f8.dx), (y8.x, y8.dx))


@pytest.mark.parametrize(
    "f9, y9", (
        (asin(exp(-1 / (x1 + x2))), Dfloat(asin(exp(-1/5)), 2/25 * exp(-1/5) /
         math.sqrt(1 - exp(-2/5)))),
        (acos(log(x4)), Dfloat(acos(log(2)),
         -1 / (4 * math.sqrt(1 - log(2) ** 2)))),
        (atan(sin(x1) * cos(x2)), Dfloat(atan(sin(2) * cos(3)),
         (cos(2) * cos(3) - sin(3) * sin(2)) / (1 + (sin(2) * cos(3)) ** 2))),
        (asinh(cos(x3)), Dfloat(asinh(cos(3.5)), - sin(3.5) /
         math.sqrt(1 + cos(3.5) ** 2))),
        (acosh(3 * tanh(x4)), Dfloat(acosh(3 * tanh(2)), 3 / (2 * cosh(2) ** 2
         * (math.sqrt(3 * tanh(2) - 1) * math.sqrt(3 * tanh(2) + 1))))),
        (atanh(x2 / x3), Dfloat(atanh(6/7), 2 / (49 * (1 - 36/49))))
    )
)
def test_inv(f9, y9):
    """Test arcsin, arccos, arctan, arsinh, arcosh and artanh methods."""
    assert allclose((f9.x, f9.dx), (y9.x, y9.dx))
