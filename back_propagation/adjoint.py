"""A first attempt at implementing adjoint algorithmic differentiation."""


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

    @property
    def derivative(self, *vars):
        for v in vars:
            v.adj = 0
        for block in tape:
            block.ops.adj = 0
        self.adj = 1
        for block in reversed(tape):
            block.compute_adjoint()
        return tuple(v.adj for v in vars)

    def __add__(self, other):
        """Implement addition."""
        result = type(self)(self.val + other.val, 0)
        tape.append(AddBlock(result, self, other))
        return result

    def __sub__(self, other):
        """Implement subtraction."""
        result = type(self)(self.val - other.val, 0)
        tape.append(SubBlock(result, self, other))
        return result

    def __mul__(self, other):
        """Implement multiplication."""
        result = type(self)(self.val * other.val, 0)
        tape.append(MulBlock(result, self, other))
        return result

    def __truediv__(self, other):
        """Implement division."""
        result = type(self)(self.val / other.val, 0)
        tape.append(DivBlock(result, self, other))
        return result

    def __pow__(self, other):
        """Implement subtraction."""
        result = type(self)(self.val ** other.val, 0)
        tape.append(PowBlock(result, self, other))
        return result


class Block:

    def __init__(self, result, *ops):
        """Initialise AdjFloat."""
        self.result = result
        self.ops = tuple(ops)


class AddBlock(Block):

    def compute_adjoint(self):
        for o in self.ops:
            o.adj += 1*self.result.adj
