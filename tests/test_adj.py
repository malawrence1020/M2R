"""Pytests for AdjFloat and other adjoint methods."""
from back_propagation import (AdjFloat, sin, cos, tan, exp, log, # noqa F401
                              sinh, cosh, tanh, asin, acos, atan, # noqa F401
                              asinh, acosh, atanh, AddBlock, # noqa F401
                              SubBlock, MulBlock, DivBlock, # noqa F401
                              PowBlock, SinBlock, CosBlock, # noqa F401
                              TanBlock, ExpBlock, LogBlock, # noqa F401
                              SinhBlock, CoshBlock, TanhBlock, # noqa F401
                              AsinBlock, AcosBlock, AtanBlock, # noqa F401
                              AsinhBlock, AcoshBlock, AtanhBlock, # noqa F401
                              clear_tape) # noqa F401
import pytest
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
