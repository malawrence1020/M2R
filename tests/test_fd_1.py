from forward_propagation import (Dfloat, sin, cos, tan, exp, log, sinh, cosh,
                              tanh, asin, acos, atan, asinh, acosh, atanh)
import pytest
from numpy import allclose

x1 = Dfloat(2,1)

@pytest.mark.parametrize(
    "f, y", (
        (x1, Dfloat(2, 1)),
        (x1 + x1, Dfloat(4, 2)),
        (x1 + 2, Dfloat(4, 1)),
        (x1 + 1 + x1, Dfloat(5, 2))
    )
)
def test_dx(f, y):
    assert allclose((f.x, f.dx), (y.x, y.dx))
