# Code added at 20251022-153521
import pytest
from sample_code.simple_cal.sub.sub import sub

@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Case 1: Positive integers, positive result
        (10, 4, 6),
        # Case 2: Positive integers, zero result
        (7, 7, 0),
        # Case 3: Positive integers, negative result (a < b)
        (3, 8, -5),
        # Case 4: Subtraction involving zero (b=0)
        (15, 0, 15),
        # Case 5: Subtraction involving zero (a=0)
        (0, 15, -15),
        # Case 6: Negative inputs
        (-10, -3, -7),
        # Case 7: Mixed signs (a positive, b negative)
        (5, -2, 7),
        # Case 8: Mixed signs (a negative, b positive)
        (-5, 2, -7),
        # Case 9: Floating point numbers
        (10.5, 3.2, 7.3),
        # Case 10: Floating point resulting in zero
        (1.1, 1.1, 0.0),
    ],
    ids=[
        "pos_int_pos_res",
        "pos_int_zero_res",
        "pos_int_neg_res",
        "sub_zero_b",
        "sub_zero_a",
        "negative_inputs",
        "mixed_signs_a_pos",
        "mixed_signs_a_neg",
        "float_subtraction",
        "float_zero_result",
    ]
)
def test_sub_various_inputs(a, b, expected):
    """
    Tests the 'sub' function across various input types (integers, floats) and sign combinations
    to ensure correct arithmetic subtraction, covering all basic operational scenarios.
    """
    result = sub(a, b)
    
    # Use pytest.approx for float comparisons to handle potential precision issues
    if isinstance(expected, float) or isinstance(result, float):
        assert result == pytest.approx(expected)
    else:
        assert result == expected
