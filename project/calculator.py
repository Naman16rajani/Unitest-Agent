from project.utils.helper import validate_numbers


class Calculator:
    def add(self, a, b):
        if validate_numbers(a, b):
            return a + b
        return None

    def subtract(self, a, b):
        if validate_numbers(a, b):
            return a - b
        return None
