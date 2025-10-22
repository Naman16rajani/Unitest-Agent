# Code added at 20251022-153529
import pytest
from sample_code.simple_cal.add.add import add

@pytest.mark.parametrize(
    "a, b, expected",
    [
        # Normal numerical addition (integers)
        (1, 2, 3),
        (-5, 3, -2),
        (100, 0, 100),
        # Floating point addition
        (1.5, 2.5, 4.0),
        (-10.5, 5.5, -5.0),
        # Sequence concatenation (testing generic '+' operator behavior for Any types)
        ("hello", "world", "helloworld"),
        ([1, 2], [3, 4], [1, 2, 3, 4]),
    ]
)
def test_add_normal_cases(a, b, expected):
    """
    Tests the 'add' function with various compatible types (numbers, strings, lists)
    to ensure correct addition or concatenation.
    """
    # Invoke the function
    result = add(a, b)
    # Assert the result matches the expected output
    assert result == expected

def test_add_type_error():
    """
    Tests that adding incompatible types (e.g., integer and string) raises a TypeError,
    validating the behavior of the underlying '+' operator.
    """
    # Expect a TypeError when attempting to add incompatible types
    with pytest.raises(TypeError):
        add(1, "a")
