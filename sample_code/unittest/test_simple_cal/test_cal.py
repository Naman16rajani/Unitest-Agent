# Code added at 20251022-153404
import pytest
from unittest.mock import MagicMock

# Import the target class Cal from sample_code/simple_cal/cal.py
try:
    from sample_code.simple_cal.cal import Cal
except ImportError:
    # Assuming the project root is configured correctly for imports
    # If not, adjust import path based on execution environment
    pass

"""
- Import necessary standard library and target modules.
- Define a Pytest fixture for the Cal class.
- Initialize the fixture as a MagicMock with the Cal class specification.
- Ensure the fixture adheres to the required naming convention (mock_Cal_instance).
- Provide a clear docstring explaining the fixture's purpose.
"""

@pytest.fixture
def mock_Cal_instance() -> MagicMock:
    """
    Provides a MagicMock instance of the Cal class (sample_code.simple_cal.cal.Cal).

    This fixture is used to isolate tests that depend on the Cal object,
    preventing the execution of its actual constructor logic or methods
    unless explicitly configured. Since no constructor dependencies were
    provided in the schema, the mock is initialized solely with the spec.
    """
    # Initialize the mock instance with the spec of the Cal class
    mock_Cal = MagicMock(spec=Cal)

    # If the Cal constructor required dependencies, they would be mocked here.
    # Example: mock_Cal.some_dependency = MagicMock()

    return mock_Cal
# Code added at 20251022-153425
import pytest
from unittest.mock import MagicMock, patch
from sample_code.simple_cal.cal import Cal

# Fixture mock_Cal_instance is provided

@patch('sample_code.simple_cal.cal.add')
@pytest.mark.parametrize("a, b, expected_result", [
    (10, 5, 15),
    (-10, 5, -5),
    (0, 5, 5),
    (1.5, 2.5, 4.0),
])
def test_add_normal_cases(mock_add, mock_Cal_instance, a, b, expected_result):
    """
    Tests the Cal.add method for various numerical inputs, ensuring it correctly 
    delegates the call to the external 'add' function and returns the result.
    
    Uses the mock_Cal_instance fixture configured to execute the real method 
    implementation while patching the external dependency 'add'.
    """
    # 1. Configure the mock instance to execute the real Cal.add method implementation.
    # This is crucial for achieving coverage of the wrapper logic 'return add(a, b)'.
    mock_Cal_instance.add.side_effect = Cal.add.__get__(mock_Cal_instance, Cal)
    
    # 2. Configure the mocked external dependency 'add' to return the expected result.
    mock
# Code added at 20251022-153433
import pytest
from unittest.mock import MagicMock, patch
# Assuming Cal is imported correctly in the existing context
from sample_code.simple_cal.cal import Cal 

# Fixture mock_Cal_instance is provided

@patch('sample_code.simple_cal.cal.mul')
@pytest.mark.parametrize("a, b, expected_result", [
    (5, 3, 15),
    (-5, 3, -15),
    (0, 100, 0),
    (2.5, 2, 5.0),
    (-1.5, -2, 3.0),
])
def test_mul_normal_cases(mock_mul, mock_Cal_instance, a, b, expected_result):
    """
    Tests the Cal.mul method for various numerical inputs, ensuring it correctly 
    delegates the call to the external 'mul' function and returns the result.
    """
    # 1. Configure the mock instance to execute the real Cal.mul method implementation
    # to cover the wrapper logic 'return mul(a, b)'.
    mock_Cal_instance.mul.side_effect = Cal.mul.__get__(mock_Cal_instance, Cal)
    
    # 2. Configure the mocked external dependency 'mul' to return the expected result.
    mock_mul.return_value = expected_result

    # 3. Invoke the method
    result = mock_Cal_instance.mul(a, b)
    
    # 4. Assertions
    assert result == expected_result
    # Verify that the external dependency was called exactly once with the correct arguments
    mock_mul.assert_called_once_with(a, b)

@patch('sample_code.simple_cal.cal.mul')
def test_mul_dependency_error(mock_mul, mock_Cal_instance):
    """
    Tests that the Cal.mul method correctly propagates exceptions raised by the 
    underlying external 'mul' function (e.g., due to invalid input types).
    """
    # 1. Configure the mock instance to execute the real Cal.mul method implementation.
    mock_Cal_instance.mul.side_effect = Cal.mul.__get__(mock_Cal_instance, Cal)
    
    # 2. Configure the mocked external dependency 'mul' to raise a specific exception.
    test_exception = TypeError("Cannot multiply non-numeric types")
    mock_mul.side_effect = test_exception

    a, b = "a", 5
    
    # 3. Invoke the method and assert that the exception is raised
    with pytest.raises(TypeError) as excinfo:
        mock_Cal_instance.mul(a, b)
        
    # 4. Assertions
    assert str(excinfo.value) == str(test_exception)
    mock_mul.assert_called_once_with(a, b)
# Code added at 20251022-153442
import pytest
from unittest.mock import MagicMock, patch
# Assuming Cal is imported correctly in the existing context
from sample_code.simple_cal.cal import Cal 

# Fixture mock_Cal_instance is provided

@patch('sample_code.simple_cal.cal.sub')
@pytest.mark.parametrize("a, b, expected_result", [
    (10, 5, 5),      # Positive result
    (5, 10, -5),     # Negative result
    (10, -5, 15),    # Subtracting a negative
    (0, 5, -5),      # Zero start
    (5.5, 2.5, 3.0), # Float subtraction
    (100, 100, 0),   # Zero result
])
def test_sub_normal_cases(mock_sub, mock_Cal_instance, a, b, expected_result):
    """
    Tests the Cal.sub method for various numerical inputs, ensuring it correctly 
    delegates the call to the external 'sub' function and returns the result.
    
    The test ensures coverage of the wrapper logic by executing the real method 
    implementation while patching the external dependency 'sub'.
    """
    # 1. Configure the mock instance to execute the real Cal.sub method implementation
    # to cover the wrapper logic 'return sub(a, b)'.
    mock_Cal_instance.sub.side_effect = Cal.sub.__get__(mock_Cal_instance, Cal)
    
    # 2. Configure the mocked external dependency 'sub' to return the expected result.
    mock_sub.return_value = expected_result

    # 3. Invoke the method
    result = mock_Cal_instance.sub(a, b)
    
    # 4. Assertions
    assert result == expected_result
    # Verify that the external dependency was called exactly once with the correct arguments
    mock_sub.assert_called_once_with(a, b)

@patch('sample_code.simple_cal.cal.sub')
def test_sub_dependency_error(mock_sub, mock_Cal_instance):
    """
    Tests that the Cal.sub method correctly propagates exceptions raised by the 
    underlying external 'sub' function (e.g., due to invalid input types or internal errors).
    """
    # 1. Configure the mock instance to execute the real Cal.sub method implementation.
    mock_Cal_instance.sub.side_effect = Cal.sub.__get__(mock_Cal_instance, Cal)
    
    # 2. Configure the mocked external dependency 'sub' to raise a specific exception.
    test_exception = TypeError("Unsupported operand types for subtraction")
    mock_sub.side_effect = test_exception

    a, b = 10, "five"
    
    # 3. Invoke the method and assert that the exception is raised
    with pytest.raises(TypeError) as excinfo:
        mock_Cal_instance.sub(a, b)
        
    # 4. Assertions
    assert str(excinfo.value) == str(test_exception)
    mock_sub.assert_called_once_with(a, b)
# Code added at 20251022-153451
import pytest
from unittest.mock import MagicMock, patch
# Assuming Cal is imported correctly in the existing context
from sample_code.simple_cal.cal import Cal 

# Fixture mock_Cal_instance is provided

@patch('sample_code.simple_cal.cal.div')
@pytest.mark.parametrize("a, b, expected_result", [
    (10, 2, 5.0),      # Standard positive division
    (-10, 2, -5.0),    # Negative result
    (10, -5, -2.0),    # Negative divisor
    (5, 2, 2.5),       # Float result
    (0, 5, 0.0),       # Zero numerator
])
def test_div_normal_cases(mock_div, mock_Cal_instance, a, b, expected_result):
    """
    Tests the Cal.div method for various numerical inputs, ensuring it correctly 
    delegates the call to the external 'div' function and returns the result.
    
    The test ensures coverage of the wrapper logic by executing the real method 
    implementation while patching the external dependency 'div'.
    """
    # 1. Configure the mock instance to execute the real Cal.div method implementation
    # to cover the wrapper logic 'return div(a, b)'.
    mock_Cal_instance.div.side_effect = Cal.div.__get__(mock_Cal_instance, Cal)
    
    # 2. Configure the mocked external dependency 'div' to return the expected result.
    mock_div.return_value = expected_result

    # 3. Invoke the method
    result = mock_Cal_instance.div(a, b)
    
    # 4. Assertions
    assert result == expected_result
    # Verify that the external dependency was called exactly once with the correct arguments
    mock_div.assert_called_once_with(a, b)

@patch('sample_code.simple_cal.cal.div')
def test_div_zero_division_error(mock_div, mock_Cal_instance):
    """
    Tests that the Cal.div method correctly propagates a ZeroDivisionError 
    when the divisor (b) is zero, ensuring the underlying dependency handles this case.
    """
    # 1. Configure the mock instance to execute the real Cal.div method implementation.
    mock_Cal_instance.div.side_effect = Cal.div.__get__(mock_Cal_instance, Cal)
    
    # 2. Configure the mocked external dependency 'div' to raise ZeroDivisionError.
    a, b = 10, 0
    test_exception = ZeroDivisionError("division by zero")
    mock_div.side_effect = test_exception

    # 3. Invoke the method and assert that the exception is raised
    with pytest.raises(ZeroDivisionError) as excinfo:
        mock_Cal_instance.div(a, b)
        
    # 4. Assertions
    assert str(excinfo.value) == str(test_exception)
    mock_div.assert_called_once_with(a, b)

@patch('sample_code.simple_cal.cal.div')
def test_div_dependency_type_error(mock_div, mock_Cal_instance):
    """
    Tests that the Cal.div method correctly propagates exceptions (e.g., TypeError) 
    raised by the underlying external 'div' function due to invalid input types.
    """
    # 1. Configure the mock instance to execute the real Cal.div method implementation.
    mock_Cal_instance.div.side_effect = Cal.div.__get__(mock_Cal_instance, Cal)
    
    # 2. Configure the mocked external dependency 'div' to raise a specific exception.
    a, b = 10, "two"
    test_exception = TypeError("Unsupported operand types for division")
    mock_div.side_effect = test_exception

    # 3. Invoke the method and assert that the exception is raised
    with pytest.raises(TypeError) as excinfo:
        mock_Cal_instance.div(a, b)
        
    # 4. Assertions
    assert str(excinfo.value) == str(test_exception)
    mock_div.assert_called_once_with(a, b)
