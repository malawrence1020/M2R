"""A first attempt at implementing forward-propagation differentiation."""
import math


class Dfloat:
    """Implement forward-propagation differentiation."""

    def __init__(self, x, dx):
        """Initialise Dfloat."""
        self.x = x
        self.dx = dx

    def __add__(self, other):
        """Implement addition."""
        return type(self)(self.x + other.x, self.dx + other.dx)

    def __radd__(self, other):
        """Reverse addition."""
        return self + other

    def __sub__(self, other):
        """Implement subtraction."""
        return type(self)(self.x - other.x, self.dx - other.dx)

    def __rsub__(self, other):
        """Reverse subtraction."""
        return self - other

    def __mul__(self, other):
        """Implement multiplication."""
        return type(self)(self.x * other.x,
                          other.x * self.dx + self.x * other.dx)

    def __rmul__(self, other):
        """Reverse multiplication."""
        return self * other

    def __div__(self, other):
        """Implement division."""
        return type(self)(self.x / other.x, (other.x * self.dx +
                          self.x * other.dx)/(other.x ** 2))

    def __rdiv__(self, other):
        """Reverse division."""
        return self / other

    def sin(x):
        if isinstance(x, Dfloat):
            return Dfloat(math.sin(x.x), x.dx * math.cos(x.x))
        else:
            return math.sin(x)

    def cos(x):
        if isinstance(x, Dfloat):
            return Dfloat(math.cos(x.x), -x.dx * math.sin(x.x))
        else:
            return math.cos(x)

    def tan(x):
        if isinstance(x, Dfloat):
            return Dfloat(math.tan(x.x), x.dx * (1 + (math.tan(x.x))**2))
        else:
            return math.tan(x)
