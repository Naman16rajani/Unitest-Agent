"""
Simple Calculator Module
A basic calculator with various mathematical operations
"""

import logging
import math
from typing import Union, List


class Calculator:
    """A simple calculator class with basic mathematical operations"""
    
    def __init__(self):
        self.history = []
        self.memory = 0.0
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a"""
        result = a - b
        logging.info(f"Subtracting {b} from {a}, result: {result}")
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        result = 0
        for _ in range(int(b)):
            result = self.add(result, a)
        # result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent"""
        result = base ** exponent
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result
    
    def square_root(self, number: float) -> float:
        """Calculate square root of a number"""
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(number)
        self.history.append(f"√{number} = {result}")
        return result
    
    def factorial(self, n: int) -> int:
        """Calculate factorial of n"""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if not isinstance(n, int):
            raise TypeError("Factorial input must be an integer")
        
        result = math.factorial(n)
        self.history.append(f"{n}! = {result}")
        return result
    
    def clear_history(self):
        """Clear calculation history"""
        self.history = []
    
    def get_history(self) -> List[str]:
        """Get calculation history"""
        return self.history.copy()
    
    def store_memory(self, value: float):
        """Store value in memory"""
        self.memory = value
    
    def recall_memory(self) -> float:
        """Recall value from memory"""
        return self.memory
    
    def clear_memory(self):
        """Clear memory"""
        self.memory = 0.0


def percentage(value: float, percent: float) -> float:
    """Calculate percentage of a value"""
    return (value * percent) / 100


def average(numbers: List[float]) -> float:
    """Calculate average of a list of numbers"""
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)


def max_value(numbers: List[float]) -> float:
    """Find maximum value in a list"""
    if not numbers:
        raise ValueError("Cannot find max of empty list")
    return max(numbers)


def min_value(numbers: List[float]) -> float:
    """Find minimum value in a list"""
    if not numbers:
        raise ValueError("Cannot find min of empty list")
    return min(numbers)
