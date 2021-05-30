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

    def __pow__(self, other):
        """Implement exponentiation."""
        return type(self)(self.x ** other.x, (self.x ** other.x) * (other.dx
                          * log(self.x) + (other.x * self.dx) / self.x))

    def __rpow__(self, other):
        """Reverse exponentiation."""
        return self ** other


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


def exp(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.exp(x.x), x.dx * math.exp(x.x))
    else:
        return math.exp(x)


def log(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.log(x.x), x.dx / x.x)
    else:
        return math.log(x)


def sinh(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.sinh(x.x), x.dx * math.cosh(x.x))
    else:
        return math.sinh(x)


def cosh(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.cosh(x.x), x.dx * math.sinh(x.x))
    else:
        return math.cosh(x)


def tanh(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.tanh(x.x), x.dx / (math.cosh(x.x))**2)
    else:
        return math.tanh(x)


def asin(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.asin(x.x), x.dx / math.sqrt(1 - (x.x**2)))
    else:
        return math.asin(x)


def acos(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.acos(x.x), - x.dx / math.sqrt(1 - (x.x**2)))
    else:
        return math.acos(x)


def atan(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.atan(x.x), x.dx / (1 + (x.x**2)))
    else:
        return math.atan(x)


def asinh(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.asinh(x.x), x.dx / math.sqrt(1 + (x.x**2)))
    else:
        return math.asinh(x)


def acosh(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.acosh(x.x),
                      x.dx / (math.sqrt(x.x - 1)*math.sqrt(x.x + 1)))
    else:
        return math.acosh(x)


def atanh(x):
    if isinstance(x, Dfloat):
        return Dfloat(math.atanh(x.x), x.dx / (1 - (x.x**2)))
    else:
        return math.atanh(x)
