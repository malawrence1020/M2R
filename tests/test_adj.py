"""Pytests for AdjFloat and other adjoint methods."""
from back_propagation import (AdjFloat, sin, cos, tan, exp, log, # noqa F401
                              sinh, cosh, tanh, asin, acos, atan, # noqa F401
                              asinh, acosh, atanh, clear_tape) # noqa F401
import pytest
import math
from numpy import allclose

x1 = AdjFloat(2, 1)
x2 = AdjFloat(3, 1)
x3 = AdjFloat(3.5, 1)
x4 = AdjFloat(2, 0.5)
x5 = AdjFloat(3, 2)


def test_type_error():
    """Test for type error."""
    with pytest.raises(TypeError):
        x1 + "frog"


@pytest.mark.parametrize(
    "f1, y1", (
        (x1.derivative(x1), 1),
        ((x1 + x1).derivative(x1), 2),
        ((x2 + 1).derivative(x2), 1),
        ((x2 + 2 + x3).derivative(x1, x2, x3), (0, 1, 1)),
        ((4 + x4).derivative(x4, x5), (1, 0)),
        ((x1 + x2 + x3 + x4).derivative(x1, x2, x3, x4, x5), (1, 1, 1, 1, 0))
    )
)
def test_add(f1, y1):
    """Test addition on AdjFloat."""
    clear_tape()
    assert allclose(f1, y1)


@pytest.mark.parametrize(
    "f2, y2", (
        ((x1 - x2).derivative(x1, x2), (1, -1)),
        ((x3 - 1).derivative(x3, x4), (1, 0)),
        ((2 - x5).derivative(x5), -1),
        ((x4 - 3 - x5).derivative(x4, x5), (1, -1)),
        ((x1 - 2 + x2 - x3 - x3).derivative(x1, x2, x3), (1, 1, -2))
    )
)
def test_sub(f2, y2):
    """Test subtraction on AdjFloat."""
    clear_tape()
    assert allclose(f2, y2)


@pytest.mark.parametrize(
    "f3, y3", (
        ((x1 * 2).derivative(x1, x3), (2, 0)),
        ((3 * x3).derivative(x1, x3), (0, 3)),
        ((x2 * x5).derivative(x2, x3, x5), (3, 0, 3)),
        ((x1 * 2 * x4).derivative(x1, x2, x4), (4, 0, 4)),
        (((x1 + x5) * (x4 - x1)).derivative(x1, x2, x3, x4, x5),
         (-5, 0, 0, 5, 0)),
    )
)
def test_mul(f3, y3):
    """Test multiplication on AdjFloat."""
    clear_tape()
    assert allclose(f3, y3)


@pytest.mark.parametrize(
    "f4, y4", (
        ((x1 / 2).derivative(x1, x3), (0.5, 0)),
        ((3 / x2).derivative(x1, x2), (0, -1/3)),
        ((x3 / x4).derivative(x3, x4, x5), (1/2, -0.875, 0)),
        (((x4 / 2) / x5).derivative(x3, x4, x5), (0, 1/6, -1/9)),
        (((x1 + x2) / (x4 * 2)).derivative(x1, x3, x4), (1/4, 0, -5/8))
    )
)
def test_div(f4, y4):
    """Test division on AdjFloat."""
    clear_tape()
    assert allclose(f4, y4)


@pytest.mark.parametrize(
    "f5, y5", (
        ((x3 ** 2).derivative(x3, x4), (7, 0)),
        ((2 ** x2).derivative(x2, x3), (8 * math.log(2), 0)),
        ((x1 ** x4).derivative(x1, x2, x4), (4, 0, 4 * math.log(2))),
        (((x1 + x2) ** (x5 - x4)).derivative(x1, x2, x3, x4, x5),
         (1, 1, 0, -5 * math.log(5), 5 * math.log(5)))
    )
)
def test_pow(f5, y5):
    """Test exponentiation on AdjFloat."""
    clear_tape()
    assert allclose(f5, y5)


@pytest.mark.parametrize(
    "f6, y6", (
        ((sin(x1 * x2)).derivative(x1, x2, x4), (3 * math.cos(6),
                                                 2 * math.cos(6), 0)),
        ((cos(x4 / x5)).derivative(x1, x4, x5), (0, -1/3 * math.sin(2/3),
                                                 2/9 * math.sin(2/3))),
        ((tan(x1 + x2 - x3)).derivative(x1, x2, x3), (1 / math.cos(1.5) ** 2,
         1 / math.cos(1.5) ** 2, -1 / math.cos(1.5) ** 2)),
        ((sin(x1) + cos(x2) + tan(x4)).derivative(x1, x2, x4),
         (math.cos(2), - math.sin(3), 1 / math.cos(2) ** 2)),
    )
)
def test_trig(f6, y6):
    """Test sin, cos and tan on AdjFloat."""
    clear_tape()
    assert allclose(f6, y6)


@pytest.mark.parametrize(
    "f7, y7", (
        ((exp(x1 * x2 + x3)).derivative(x1, x2, x3), (3 * math.exp(9.5),
         2 * math.exp(9.5), math.exp(9.5))),
        ((log(sin(x4))).derivative(x4, x4, x5), (1 / math.tan(2),
         1 / math.tan(2), 0)),
        ((exp(cos(x5)) + log(cos(2 * x5))).derivative(x1, x1, x5), (0, 0,
         - math.sin(3) * math.exp(math.cos(3)) - 2 * math.tan(6)))
    )
)
def test_exp(f7, y7):
    """Test exp and log on AdjFloat."""
    clear_tape()
    assert allclose(f7, y7)


@pytest.mark.parametrize(
    "f8, y8", (
        ((sinh(cos(-1 * x1))).derivative(x1, x2), (math.sin(-2) *
         math.cosh(math.cos(-2)), 0)),
        ((cosh(sin(1 + x2))).derivative(x1, x2), (0, math.cos(4) *
         math.sinh(math.sin(4)))),
        ((tanh((x3 + 0.5) / 2)).derivative(x3, x4, x5),
         (1 / (2 * math.cosh(2) ** 2), 0, 0)),
        ((sinh(3 - x4) * cosh(x5 / 3)).derivative(x4, x5),
         (-1 * math.cosh(1) ** 2, 1/3 * math.sinh(1) ** 2))
    )
)
def test_hyp(f8, y8):
    """Test sinh, cosh and tanh on AdjFloat."""
    clear_tape()
    assert allclose(f8, y8)


@pytest.mark.parametrize(
    "f9, y9", (
        ((asin(x1 - 1.5)).derivative(x1, x2), (1 / math.sqrt(0.75), 0)),
        ((acos(x2 - 3.5)).derivative(x1, x2), (0, - 1 / math.sqrt(0.75))),
        ((atan(x4 * x5)).derivative(x4, x5), (3/37, 2/37)),
        ((asin(x3 - 3) + acos(x3 - 4) - atan(x3)).derivative(x3, x3),
         (-4/53, -4/53)),
        ((asinh(2 * x4)).derivative(x4, x5), (2 / math.sqrt(17), 0)),
        ((acosh(x5 ** 2)).derivative(x4, x5), (0, 3 / (2 * math.sqrt(5)))),
        ((atanh(x1 / x2)).derivative(x1, x2, x3), (3/5, -2/5, 0)),
        ((asinh(x3) + acosh(x3) + atanh(x3 - 3)).derivative(x3),
         2 / math.sqrt(53) + 2 / math.sqrt(45) + 4/3)
    )
)
def test_inv(f9, y9):
    """Test arcsin, arccos, arctan, arsinh, arcosh and artanh on AdjFloat."""
    clear_tape()
    assert allclose(f9, y9)
