from forward_propagation import (Dfloat, sin, cos, tan, exp, log, sinh, cosh,
                                 tanh, asin, acos, atan, asinh, acosh, atanh)
import pytest
from numpy import allclose

x1 = Dfloat(2, 1)
x2 = Dfloat(3, 1)
x3 = Dfloat(3.5, 1)
x4 = Dfloat(2, 0.5)


def test_type_error():
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
    assert allclose((f1.x, f1.dx), (y1.x, y1.dx))


@pytest.mark.parametrize(
    "f2, y2", (
        (x1 - 1, Dfloat(1, 1)),
        (1 - x2, Dfloat(-2, -1)),
        (x3 - 1 - x4, Dfloat(0.5, 0.5)),
        (x1 - 1 + x2 - x4 + 2, Dfloat(4, 1.5))
    )
)
def test_sub(f2, y2):
    assert allclose((f2.x, f2.dx), (y2.x, y2.dx))


@pytest.mark.parametrize(
    "f3, y3", (
        (x1 * 2, Dfloat(4, 2)),
        (3 * x2, Dfloat(9, 3)),
        (x3 * 2.5 * x4, Dfloat(17.5, 9.375)),
        ((x1 + 1) * x4 * (2 - x2), Dfloat(-6, -9.5))
    )
)
def test_mul(f3, y3):
    assert allclose((f3.x, f3.dx), (y3.x, y3.dx))
