"""A first attempt at implementing adjoint algorithmic differentiation."""
import math
from functools import wraps
from numbers import Number


def deal_with_other_types(meth):
    """Cast the second argument of a method to AdjFloat when needed."""
    @wraps(meth)
    def fn(self, other):
        if not isinstance(other, AdjFloat):
            if isinstance(other, Number):
                other = AdjFloat(other, 0)
            else:
                raise TypeError(
                    (f"Can only operate on a AdjFloat or a Number, "
                     f"not a {type(other).__name__}"))
        return meth(self, other)
    return fn


def deal_with_other_types2(meth):
    """Cast the argument of a method to AdjFloat when needed."""
    @wraps(meth)
    def fn(self):
        if not isinstance(self, AdjFloat):
            if isinstance(self, Number):
                self = AdjFloat(self, 0)
            else:
                raise TypeError(
                    (f"Can only operate on a AdjFloat or a Number, "
                     f"not a {type(self).__name__}"))
        return meth(self)
    return fn


class AdjFloat:
    """Implement backward-propagation differentiation."""

    global tape
    tape = []

    def __init__(self, val, adj):
        """Initialise AdjFloat."""
        self.val = val
        self.adj = adj

    def __repr__(self):
        """Representation of AdjFloat."""
        return (self.__class__.__name__ + "(" + str(self.val) +
                "," + str(self.adj) + ")")

    def derivative(self, *vars):
        """Return the derivative of AdjFloat by running the tape backwards."""
        for v in vars:
            v.adj = 0
        for block in tape:
            block.result.adj = 0
        self.adj = 1
        for block in reversed(tape):
            block.compute_adjoint()
        return tuple(v.adj for v in vars)

    @deal_with_other_types
    def __add__(self, other):
        """Implement addition."""
        result = type(self)(self.val + other.val, 0)
        tape.append(AddBlock(result, self, other))
        return result

    @deal_with_other_types
    def __radd__(self, other):
        """Reverse addition."""
        return self + other

    @deal_with_other_types
    def __sub__(self, other):
        """Implement subtraction."""
        result = type(self)(self.val - other.val, 0)
        tape.append(SubBlock(result, self, other))
        return result

    @deal_with_other_types
    def __rsub__(self, other):
        """Reverse subtraction."""
        return other - self

    @deal_with_other_types
    def __mul__(self, other):
        """Implement multiplication."""
        result = type(self)(self.val * other.val, 0)
        tape.append(MulBlock(result, self, other))
        return result

    @deal_with_other_types
    def __rmul__(self, other):
        """Reverse multiplication."""
        return self * other

    @deal_with_other_types
    def __truediv__(self, other):
        """Implement division."""
        result = type(self)(self.val / other.val, 0)
        tape.append(DivBlock(result, self, other))
        return result

    @deal_with_other_types
    def __rtruediv__(self, other):
        """Reverse division."""
        return other / self

    @deal_with_other_types
    def __pow__(self, other):
        """Implement subtraction."""
        result = type(self)(self.val ** other.val, 0)
        tape.append(PowBlock(result, self, other))
        return result

    @deal_with_other_types
    def __rpow__(self, other):
        """Reverse exponentiation."""
        return other ** self


@deal_with_other_types2
def sin(x):
    """Implement sin for AdjFloat."""
    result = type(x)(math.sin(x.val), 0)
    tape.append(SinBlock(result, x))
    return result


@deal_with_other_types2
def cos(x):
    """Implement cos for AdjFloat."""
    result = type(x)(math.cos(x.val), 0)
    tape.append(CosBlock(result, x))
    return result


@deal_with_other_types2
def tan(x):
    """Implement tan for AdjFloat."""
    result = type(x)(math.tan(x.val), 0)
    tape.append(TanBlock(result, x))
    return result


@deal_with_other_types2
def exp(x):
    """Implement exp for AdjFloat."""
    result = type(x)(math.exp(x.val), 0)
    tape.append(ExpBlock(result, x))
    return result


@deal_with_other_types2
def log(x):
    """Implement log for AdjFloat."""
    result = type(x)(math.log(x.val), 0)
    tape.append(LogBlock(result, x))
    return result


@deal_with_other_types2
def sinh(x):
    """Implement sinh for AdjFloat."""
    result = type(x)(math.sinh(x.val), 0)
    tape.append(SinhBlock(result, x))
    return result


@deal_with_other_types2
def cosh(x):
    """Implement cosh for AdjFloat."""
    result = type(x)(math.cosh(x.val), 0)
    tape.append(CoshBlock(result, x))
    return result


@deal_with_other_types2
def tanh(x):
    """Implement tanh for AdjFloat."""
    result = type(x)(math.tanh(x.val), 0)
    tape.append(TanhBlock(result, x))
    return result


@deal_with_other_types2
def asin(x):
    """Implement asin for AdjFloat."""
    result = type(x)(math.asin(x.val), 0)
    tape.append(AsinBlock(result, x))
    return result


@deal_with_other_types2
def acos(x):
    """Implement acos for AdjFloat."""
    result = type(x)(math.acos(x.val), 0)
    tape.append(AcosBlock(result, x))
    return result


@deal_with_other_types2
def atan(x):
    """Implement atan for AdjFloat."""
    result = type(x)(math.atan(x.val), 0)
    tape.append(AtanBlock(result, x))
    return result


@deal_with_other_types2
def asinh(x):
    """Implement asinh for AdjFloat."""
    result = type(x)(math.asinh(x.val), 0)
    tape.append(AsinhBlock(result, x))
    return result


@deal_with_other_types2
def acosh(x):
    """Implement acosh for AdjFloat."""
    result = type(x)(math.acosh(x.val), 0)
    tape.append(AcoshBlock(result, x))
    return result


@deal_with_other_types2
def atanh(x):
    """Implement atanh for AdjFloat."""
    result = type(x)(math.atanh(x.val), 0)
    tape.append(AtanhBlock(result, x))
    return result


class Block:
    """Log an operation onto the tape to implement chain rule in reverse."""

    def __init__(self, result, *ops):
        """Initialise Block."""
        self.result = result
        self.ops = ops

    def __repr__(self):
        """Representation of Block."""
        return (self.__class__.__name__ + "(" + str(self.result) +
                "," + str(self.ops) + ")")


class AddBlock(Block):
    """Log an addition operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        for o in self.ops:
            o.adj += 1*self.result.adj


class SubBlock(Block):
    """Log a subtraction operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += 1*self.result.adj
        self.ops[1].adj -= 1*self.result.adj


class MulBlock(Block):
    """Log a multiplication operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        for i, o in enumerate(self.ops):
            o.adj += self.ops[(i+1) % 2].val * self.result.adj


class DivBlock(Block):
    """Log a division operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += self.result.adj / self.ops[1].val
        self.ops[1].adj -= (self.ops[0].val * self.result.adj /
                            (self.ops[1].val ** 2))


class PowBlock(Block):
    """Log an exponentiation operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += ((self.ops[1].val / self.ops[0].val) *
                            self.result.adj *
                            self.ops[0].val ** self.ops[1].val)
        self.ops[1].adj += (math.log(self.ops[0].val) * self.result.adj *
                            self.ops[0].val ** self.ops[1].val)


class SinBlock(Block):
    """Log a sin operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += math.cos(self.ops[0].val) * self.result.adj


class CosBlock(Block):
    """Log a cos operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj -= math.sin(self.ops[0].val) * self.result.adj


class TanBlock(Block):
    """Log a tan operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += ((1 + (math.tan(self.ops[0].val)) ** 2) *
                            self.result.adj)


class ExpBlock(Block):
    """Log an exp operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += math.exp(self.ops[0].val) * self.result.adj


class LogBlock(Block):
    """Log a log operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += self.result.adj / self.ops[0].val


class SinhBlock(Block):
    """Log a sinh operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += math.cosh(self.ops[0].val) * self.result.adj


class CoshBlock(Block):
    """Log a cosh operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += math.sinh(self.ops[0].val) * self.result.adj


class TanhBlock(Block):
    """Log a tanh operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += self.result.adj / (math.cosh(self.ops[0].val) ** 2)


class AsinBlock(Block):
    """Log an arcsin operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += (self.result.adj /
                            math.sqrt(1 - (self.ops[0].val ** 2)))


class AcosBlock(Block):
    """Log an arccos operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj -= (self.result.adj /
                            math.sqrt(1 - (self.ops[0].val ** 2)))


class AtanBlock(Block):
    """Log an arctan operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += self.result.adj / (1 + (self.ops[0].val ** 2))


class AsinhBlock(Block):
    """Log an arsinh operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += (self.result.adj /
                            math.sqrt(1 + (self.ops[0].val ** 2)))


class AcoshBlock(Block):
    """Log an arcosh operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += (self.result.adj / (math.sqrt(self.ops[0].val - 1) *
                            math.sqrt(self.ops[0].val + 1)))


class AtanhBlock(Block):
    """Log an artanh operation onto the tape."""

    def compute_adjoint(self):
        """Pass the result of the chain rule back to AdjFloat."""
        self.ops[0].adj += self.result.adj / (1 - (self.ops[0].val ** 2))


def clear_tape():
    """Clear the tape to allow for a new AdjFloat calculation to be run."""
    tape = [] # noqa F841
