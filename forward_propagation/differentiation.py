"""A first attempt at implementing forward-propagation differentiation."""
import math


class Dfloat:
    """Implement forward-propagation differentiation."""

    def __init__(self, x, dx):
        """Initialise Dfloat."""
        self.x = x
        self.dx = dx

    def __add__(self, other):
        """Implement addition"""
        return type(self)(self.x + other.x, self.dx + other.dx)
