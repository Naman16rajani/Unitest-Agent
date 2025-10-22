# Code added at 20251022-153459
import pytest
# Assuming standard module structure based on file_path: sample_code/simple_cal/div/div.py
# Note: If this were a real project structure, the import path would need adjustment based on the test file location.
# We assume the ability to import the function directly based on the provided source path structure.
try:
    from sample_code.simple_cal.div.div import div
except ImportError:
    # Fallback for environments where the module structure is flat or relative imports are needed
    # Since the function is simple, we assume it's accessible if the test runner is configured correctly.
    # If the function definition was provided in a class context, we would import the class.
    # Since it's a standalone function, we rely on the module path derived from the source location.
    pass


@pytest.mark.parametrize("a, b, expected", [
    (10, 2, 5.0),
    (5, 2, 2.5),
    (-10, 5, -2.0),
    (0, 5, 0.0),
    (10.5, 3, 3.5),
])
def test_div_success(a, b, expected):
    """
    Tests successful division for various numeric inputs (positive, negative, zero, floats).
    Covers the 'return a / b' branch.
    """
    # Act
    result = div(a, b)

    # Assert
    assert result == expected


def test_div_by_zero_error():
    """
    Tests the case where the divisor (b) is zero, ensuring a ValueError is raised.
    Covers the 'if b == 0: raise ValueError' branch.
    """
    # Arrange
    a = 10
    b = 0

    # Act & Assert
    with pytest.raises(ValueError) as excinfo:
        div(a, b)

    # Validate the exception message
    assert "Cannot divide by zero" in str(excinfo.value)
