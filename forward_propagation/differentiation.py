"""A first attempt at implementing forward-propagation differentiation."""
import math
from functools import wraps
from numbers import Number


def dealing_with_other_types(meth):
    """Cast the second argument of a method to Dfloat when needed."""
    @wraps(meth)
    def fn(self, other):
        if not isinstance(other, Dfloat):
            if isinstance(other, Number):
                other = Dfloat(other, 0)
            else:
                raise TypeError(
                    (f"Can only operate on a Dfloat or a Number, "
                     f"not a {type(other).__name__}"))
        return meth(self, other)
    return fn


class Dfloat:
    """Implement forward-propagation differentiation."""

    def __init__(self, x, dx):
        """Initialise Dfloat."""
        self.x = x
        self.dx = dx

    def __repr__(self):
        """Representation of Dfloat."""
        return (self.__class__.__name__ + "(" + str(self.x) +
                "," + str(self.dx) + ")")

    @dealing_with_other_types
    def __add__(self, other):
        """Implement addition."""
        return type(self)(self.x + other.x, self.dx + other.dx)

    @dealing_with_other_types
    def __radd__(self, other):
        """Reverse addition."""
        return self + other

    @dealing_with_other_types
    def __sub__(self, other):
        """Implement subtraction."""
        return type(self)(self.x - other.x, self.dx - other.dx)

    @dealing_with_other_types
    def __rsub__(self, other):
        """Reverse subtraction."""
        return other - self

    @dealing_with_other_types
    def __mul__(self, other):
        """Implement multiplication."""
        return type(self)(self.x * other.x,
                          other.x * self.dx + self.x * other.dx)

    @dealing_with_other_types
    def __rmul__(self, other):
        """Reverse multiplication."""
        return self * other

    @dealing_with_other_types
    def __truediv__(self, other):
        """Implement division."""
        return type(self)(self.x / other.x, (other.x * self.dx -
                          self.x * other.dx) / (other.x ** 2))

    @dealing_with_other_types
    def __rtruediv__(self, other):
        """Reverse division."""
        return other / self

    @dealing_with_other_types
    def __pow__(self, other):
        """Implement exponentiation."""
        return type(self)(self.x ** other.x, (self.x ** other.x) * (other.dx
                          * log(self.x) + (other.x * self.dx) / self.x))

    @dealing_with_other_types
    def __rpow__(self, other):
        """Reverse exponentiation."""
        return other ** self


def sin(x):
    """Define sin for Dfloat, else use math.sin."""
    if isinstance(x, Dfloat):
        return Dfloat(math.sin(x.x), x.dx * math.cos(x.x))
    else:
        return math.sin(x)


def cos(x):
    """Define cos for Dfloat, else use math.cos."""
    if isinstance(x, Dfloat):
        return Dfloat(math.cos(x.x), -x.dx * math.sin(x.x))
    else:
        return math.cos(x)


def tan(x):
    """Define tan for Dfloat, else use math.tan."""
    if isinstance(x, Dfloat):
        return Dfloat(math.tan(x.x), x.dx * (1 + (math.tan(x.x))**2))
    else:
        return math.tan(x)


def exp(x):
    """Define exp for Dfloat, else use math.exp."""
    if isinstance(x, Dfloat):
        return Dfloat(math.exp(x.x), x.dx * math.exp(x.x))
    else:
        return math.exp(x)


def log(x):
    """Define log for Dfloat, else use math.log."""
    if isinstance(x, Dfloat):
        return Dfloat(math.log(x.x), x.dx / x.x)
    else:
        return math.log(x)


def sinh(x):
    """Define sinh for Dfloat, else use math.sinh."""
    if isinstance(x, Dfloat):
        return Dfloat(math.sinh(x.x), x.dx * math.cosh(x.x))
    else:
        return math.sinh(x)


def cosh(x):
    """Define cosh for Dfloat, else use math.cosh."""
    if isinstance(x, Dfloat):
        return Dfloat(math.cosh(x.x), x.dx * math.sinh(x.x))
    else:
        return math.cosh(x)


def tanh(x):
    """Define tanh for Dfloat, else use math.tanh."""
    if isinstance(x, Dfloat):
        return Dfloat(math.tanh(x.x), x.dx / (math.cosh(x.x))**2)
    else:
        return math.tanh(x)


def asin(x):
    """Define asin for Dfloat, else use math.asin."""
    if isinstance(x, Dfloat):
        return Dfloat(math.asin(x.x), x.dx / math.sqrt(1 - (x.x**2)))
    else:
        return math.asin(x)


def acos(x):
    """Define acos for Dfloat, else use math.acos."""
    if isinstance(x, Dfloat):
        return Dfloat(math.acos(x.x), - x.dx / math.sqrt(1 - (x.x**2)))
    else:
        return math.acos(x)


def atan(x):
    """Define atan for Dfloat, else use math.atan."""
    if isinstance(x, Dfloat):
        return Dfloat(math.atan(x.x), x.dx / (1 + (x.x**2)))
    else:
        return math.atan(x)


def asinh(x):
    """Define asinh for Dfloat, else use math.asinh."""
    if isinstance(x, Dfloat):
        return Dfloat(math.asinh(x.x), x.dx / math.sqrt(1 + (x.x**2)))
    else:
        return math.asinh(x)


def acosh(x):
    """Define acosh for Dfloat, else use math.acosh."""
    if isinstance(x, Dfloat):
        return Dfloat(math.acosh(x.x),
                      x.dx / (math.sqrt(x.x - 1) * math.sqrt(x.x + 1)))
    else:
        return math.acosh(x)


def atanh(x):
    """Define atanh for Dfloat, else use math.atanh."""
    if isinstance(x, Dfloat):
        return Dfloat(math.atanh(x.x), x.dx / (1 - (x.x**2)))
    else:
        return math.atanh(x)
