# Code added at 20251022-153510
import pytest
from sample_code.simple_cal.mul.mul import mul

@pytest.mark.parametrize("a, b, expected", [
    # 1. Positive integers
    (5, 10, 50),
    # 2. Mixed signs
    (-5, 10, -50),
    # 3. Negative integers
    (-5, -10, 50),
    # 4. Zero inputs
    (0, 100, 0),
    (100, 0, 0),
    # 5. Floating point numbers
    (1.5, 2.0, 3.0),
    (0.1, 0.2, 0.02),
    (10, 0.5, 5.0),
    # 6. Large numbers
    (1000000, 1000000, 1000000000000),
])
def test_mul_standard_cases(a: float, b: float, expected: float):
    """
    Tests the 'mul' function for standard multiplication scenarios, 
    covering positive, negative, zero, and floating-point inputs to ensure 
    correct arithmetic operation.
    """
    result = mul(a, b)
    
    # Use pytest.approx for floating point comparisons to handle potential precision issues
    if isinstance(expected, float) or isinstance(a, float) or isinstance(b, float):
        assert result == pytest.approx(expected)
    else:
        assert result == expected
