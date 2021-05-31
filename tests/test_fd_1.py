from forward_propagation import (Dfloat, sin, cos, tan, exp, log, sinh, cosh,
                              tanh, asin, acos, atan, asinh, acosh, atanh)
import pytest
from numpy import allclose

x1 = Dfloat(2,1)
x2 = Dfloat(3,1)
x3 = Dfloat(3.5,1)
x4 = Dfloat(2,0.5)

@pytest.mark.parametrize(
    "f, y", (
        (x1, Dfloat(2, 1)),
        (x1 + x1, Dfloat(4, 2)),
        (x1 + 2, Dfloat(4, 1)),
        (x1 + 1 + x1, Dfloat(5, 2)),
        (x2 + 3 + x2, Dfloat(9, 2)),
        (x3 + 4.25 + x3, Dfloat(11.25, 2)),
        (x4 + 1 + x4, Dfloat(5, 1)),
        (x1 + x2 + x3 + x4, Dfloat(10.5, 3.5)),
        (x1 - 1, Dfloat(1, 1)),
        (1 - x2, Dfloat(-2, -1)),
        (x3 - 1 - x4, Dfloat(0.5, 0.5)),
        (x1 - 1 + x2 - x4 + 2 , Dfloat(4, 1.5)),
        (x1 + "frog" , Dfloat(2, 1))
    )
)
def test_dx(f, y):
    assert allclose((f.x, f.dx), (y.x, y.dx))
