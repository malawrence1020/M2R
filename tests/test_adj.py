"""Pytests for AdjFloat and other adjoint methods."""
from back_propagation import (AdjFloat, AddBlock, clear_tape)
import pytest
from numpy import allclose

x1 = AdjFloat(2, 1)
x2 = AdjFloat(3, 1)
x3 = AdjFloat(3.5, 1)
x4 = AdjFloat(2, 0.5)
x5 = AdjFloat(3, 2)


@pytest.mark.parametrize(
    "f1, y1", (
        (x1.derivative(x1), 1),
        ((x1 + x1).derivative(x1), 2),
        ((x1 + x2 + x3 + x4 + x5).derivative(x1, x2, x3, x4, x5), (1, 1, 1, 1, 1))
    )
)
def test_add(f1, y1):
    """Test addition on AdjFloat."""
    clear_tape()
    assert allclose(f1, y1)
