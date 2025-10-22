from .add.add import add
from .mul.mul import mul
from .sub.sub import sub
from .div.div import div


class Cal:
    """Calculator class that uses all operation modules"""

    def add(self, a, b):
        return add(a, b)

    def mul(self, a, b):
        return mul(a, b)

    def sub(self, a, b):
        return sub(a, b)

    def div(self, a, b):
        return div(a, b)
