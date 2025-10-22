# Code added at 20251022-152826
import pytest
from unittest.mock import MagicMock

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # If running tests from a different context, adjust import path if necessary
    # Assuming standard package structure where 'sample_code' is accessible
    from sample_code.calculator import Calculator


"""
- Define the necessary imports (pytest, MagicMock, Calculator).
- Create a Pytest fixture named `mock_calculator_instance`.
- Initialize the mock using `MagicMock(spec=Calculator)`.
- Set initial state variables (`history`, `memory`) based on the constructor logic.
- Provide a descriptive docstring for the fixture.
"""


@pytest.fixture
def mock_calculator_instance() -> MagicMock:
    """
    A MagicMock instance for the Calculator class (from sample_code/calculator.py).

    This mock is initialized with the internal state variables defined in the
    constructor: `history` (list) and `memory` (float).
    No external dependencies are mocked as the constructor is simple.
    """
    # Initialize the mock instance, ensuring it adheres to the Calculator interface
    mock_calculator = MagicMock(spec=Calculator)

    # --- Initialize simple variables based on constructor logic ---
    # Constructor: self.history = []
    mock_calculator.history = []

    # Constructor: self.memory = 0.0
    mock_calculator.memory = 0.0

    # No methods are called in the constructor, so no function mocks are needed.

    return mock_calculator
# Code added at 20251022-152842
import pytest
from unittest.mock import MagicMock
# Assuming Calculator is imported from the existing setup context
try:
    from ..calculator import Calculator
except ImportError:
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "a, b, expected_result, expected_history_entry",
    [
        # Normal positive addition
        (5.0, 3.0, 8.0, "5.0 + 3.0 = 8.0"),
        # Addition involving negative numbers
        (-10.0, 4.0, -6.0, "-10.0 + 4.0 = -6.0"),
        # Addition involving zero
        (0.0, 7.5, 7.5, "0.0 + 7.5 = 7.5"),
        # Addition of two negative numbers
        (-2.5, -1.5, -4.0, "-2.5 + -1.5 = -4.0"),
        # Floating point addition
        (1.1, 2.2, 3.3, "1.1 + 2.2 = 3.3"),
    ]
)
def test_add_success(mock_calculator_instance: MagicMock, a: float, b: float, expected_result: float, expected_history_entry: str):
    """
    Tests the Calculator.add method for various numerical inputs, verifying the
    correct calculation and the update of the history log.

    We patch the real 'add' method onto the mock instance to execute the actual logic
    while utilizing the provided fixture structure.
    """
    # 1. Patch the mock instance's 'add' method to use the real implementation
    # This ensures the actual calculation and history appending logic is executed.
    mock_calculator_instance.add.side_effect = Calculator.add.__get__(
        mock_calculator_instance, Calculator
    )

    # Ensure history is empty before the test run (guaranteed by function scope fixture)
    assert mock_calculator_instance.history == []

    # 2. Call the method
    result = mock_calculator_instance.add(a, b)

    # 3. Assert the return value
    # Use approximate comparison for floats if necessary, but standard equality is fine for these simple cases
    assert result == pytest.approx(expected_result)

    # 4. Assert history update (side effect)
    # Check that the history list was updated exactly once with the correct formatted string
    assert len(mock_calculator_instance.history) == 1
    assert mock_calculator_instance.history[0] == expected_history_entry
# Code added at 20251022-152900
import pytest
from unittest.mock import MagicMock, patch

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "a, b, expected_result, expected_history_entry",
    [
        # 1. Positive - Positive
        (10.0, 5.0, 5.0, "10.0 - 5.0 = 5.0"),
        # 2. Negative - Positive
        (-10.0, 4.0, -14.0, "-10.0 - 4.0 = -14.0"),
        # 3. Positive - Negative (10 - (-5) = 15)
        (10.0, -5.0, 15.0, "10.0 - -5.0 = 15.0"),
        # 4. Negative - Negative (-2.5 - (-1.5) = -1.0)
        (-2.5, -1.5, -1.0, "-2.5 - -1.5 = -1.0"),
        # 5. Zero involvement (0 - 7.5 = -7.5)
        (0.0, 7.5, -7.5, "0.0 - 7.5 = -7.5"),
        # 6. Floating point subtraction
        (3.3, 1.1, 2.2, "3.3 - 1.1 = 2.2"),
    ]
)
# Patch logging module used inside calculator.py
@patch('sample_code.calculator.logging')
def test_subtract_success(mock_logging, mock_calculator_instance: MagicMock, a: float, b: float, expected_result: float, expected_history_entry: str):
    """
    Tests the Calculator.subtract method for various numerical inputs, verifying the
    correct calculation, history log update, and logging call.
    Achieves 100% coverage for the method logic.
    """
    # 1. Patch the mock instance's 'subtract' method to use the real implementation
    # This ensures the actual calculation and history appending logic is executed
    # while using the mock's state (self.history).
    mock_calculator_instance.subtract.side_effect = Calculator.subtract.__get__(
        mock_calculator_instance, Calculator
    )

    # Ensure initial state is correct (history is empty due to fixture scope)
    initial_history_length = len(mock_calculator_instance.history)

    # 2. Call the method
    result = mock_calculator_instance.subtract(a, b)

    # 3. Assert the return value, using approx for float comparison
    assert result == pytest.approx(expected_result)

    # 4. Assert history update (side effect)
    # History should have exactly one new entry
    assert len(mock_calculator_instance.history) == initial_history_length + 1
    assert mock_calculator_instance.history[-1] == expected_history_entry

    # 5. Assert logging call
    expected_log_message = f"Subtracting {b} from {a}, result: {expected_result}"
    mock_logging.info.assert_called_once_with(expected_log_message)
# Code added at 20251022-152937
import pytest
from unittest.mock import MagicMock

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "a, b, expected_result, expected_history_entry, add_return_sequence, expected_add_calls",
    [
        # Case 1: Positive integer multiplication (Loop runs 3 times)
        (5.0, 3.0, 15.0, "5.0 * 3.0 = 15.0", [5.0, 10.0, 15.0], 3),
        # Case 2: Zero multiplier (Loop runs 0 times)
        (10.0, 0.0, 0.0, "10.0 * 0.0 = 0.0", [], 0),
        # Case 3: Float multiplier (Truncation: int(3.9) = 3. Loop runs 3 times)
        (2.0, 3.9, 6.0, "2.0 * 3.9 = 6.0", [2.0, 4.0, 6.0], 3),
        # Case 4: Negative multiplier (Edge case: int(-2.0) = -2. range(-2) is empty. Loop runs 0 times)
        (5.0, -2.0, 0.0, "5.0 * -2.0 = 0.0", [], 0),
        # Case 5: Negative multiplicand (Loop runs 2 times)
        (-4.0, 2.0, -8.0, "-4.0 * 2.0 = -8.0", [-4.0, -8.0], 2),
    ]
)
def test_multiply_various_cases(
    mock_calculator_instance: MagicMock,
    a: float,
    b: float,
    expected_result: float,
    expected_history_entry: str,
    add_return_sequence: list[float],
    expected_add_calls: int
):
    """
    Tests the Calculator.multiply method, covering positive, zero, float truncation,
    and negative multiplier edge cases. Verifies the use of the internal self.add
    dependency and history logging, achieving 100% coverage for the method logic.
    """
    # 1. Configure the dependency mock (self.add)
    # Set the side_effect to return the sequence of intermediate results expected
    # from the repeated addition loop.
    mock_calculator_instance.add.side_effect = add_return_sequence

    # 2. Patch the mock instance's 'multiply' method to use the real implementation
    # This executes the actual loop logic using the mock's state (self.history)
    # and the configured mock dependency (self.add).
    mock_calculator_instance.multiply.side_effect = Calculator.multiply.__get__(
        mock_calculator_instance, Calculator
    )

    # Store initial history length for verification
    initial_history_length = len(mock_calculator_instance.history)

    # 3. Call the method
    result = mock_calculator_instance.multiply(a, b)

    # 4. Assert the return value
    assert result == pytest.approx(expected_result)

    # 5. Assert history update (side effect)
    assert len(mock_calculator_instance.history) == initial_history_length + 1
    assert mock_calculator_instance.history[-1] == expected_history_entry

    # 6. Assert dependency calls (self.add)
    # Verify that self.add was called the correct number of times based on int(b)
    assert mock_calculator_instance.add.call_count == expected_add_calls

    # 7. Verify call arguments for self.add (if calls were made)
    if expected_add_calls > 0:
        # Check that the second argument passed to add was always 'a'
        for call in mock_calculator_instance.add.call_args_list:
            # call[0] is the positional arguments tuple (result, a)
            assert call[0][1] == a
# Code added at 20251022-152949
import pytest
from unittest.mock import MagicMock

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "a, b, expected_result",
    [
        # 1. Positive / Positive
        (10.0, 2.0, 5.0),
        # 2. Negative / Positive
        (-10.0, 5.0, -2.0),
        # 3. Positive / Negative
        (10.0, -4.0, -2.5),
        # 4. Float result (1/3)
        (1.0, 3.0, 1/3),
        # 5. Zero numerator
        (0.0, 5.0, 0.0),
    ]
)
def test_divide_success(mock_calculator_instance: MagicMock, a: float, b: float, expected_result: float):
    """
    Tests the Calculator.divide method for successful division (b != 0), verifying the
    correct calculation and history update.
    Achieves coverage for the successful execution path.
    """
    # 1. Patch the mock instance's 'divide' method to use the real implementation
    # This ensures the actual calculation and history appending logic is executed.
    mock_calculator_instance.divide.side_effect = Calculator.divide.__get__(
        mock_calculator_instance, Calculator
    )

    # Ensure history is empty initially
    initial_history_length = len(mock_calculator_instance.history)

    # 2. Call the method
    result = mock_calculator_instance.divide(a, b)

    # 3. Assert the return value (using approx for float precision)
    assert result == pytest.approx(expected_result)

    # 4. Assert history update (side effect)
    # The history entry uses the string representation of the calculated result
    expected_history_entry = f"{a} / {b} = {result}"
    assert len(mock_calculator_instance.history) == initial_history_length + 1
    assert mock_calculator_instance.history[-1] == expected_history_entry


@pytest.mark.parametrize("a", [10.0, -5.0, 0.0])
def test_divide_by_zero_raises_value_error(mock_calculator_instance: MagicMock, a: float):
    """
    Tests the Calculator.divide method when the divisor (b) is zero, ensuring
    a ValueError is raised and history is not updated.
    Achieves coverage for the division by zero error branch.
    """
    b = 0.0

    # 1. Patch the mock instance's 'divide' method to use the real implementation
    mock_calculator_instance.divide.side_effect = Calculator.divide.__get__(
        mock_calculator_instance, Calculator
    )

    # Ensure history is empty initially
    initial_history_length = len(mock_calculator_instance.history)

    # 2. Assert that ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        mock_calculator_instance.divide(a, b)

    # 3. Assert the error message
    assert "Cannot divide by zero" in str(excinfo.value)

    # 4. Assert history was NOT updated
    assert len(mock_calculator_instance.history) == initial_history_length
# Code added at 20251022-153005
import pytest
from unittest.mock import MagicMock

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "base, exponent, expected_result, expected_history_suffix",
    [
        # 1. Standard positive integer power (2^3 = 8)
        (2.0, 3.0, 8.0, " = 8.0"),
        # 2. Zero exponent (5^0 = 1)
        (5.0, 0.0, 1.0, " = 1.0"),
        # 3. Negative exponent (4^-0.5 = 0.5)
        (4.0, -0.5, 0.5, " = 0.5"),
        # 4. Negative base, odd exponent ((-2)^3 = -8)
        (-2.0, 3.0, -8.0, " = -8.0"),
        # 5. Negative base, even exponent ((-3)^2 = 9)
        (-3.0, 2.0, 9.0, " = 9.0"),
        # 6. Fractional exponent (9^0.5 = 3)
        (9.0, 0.5, 3.0, " = 3.0"),
        # 7. Zero base, positive exponent (0^5 = 0)
        (0.0, 5.0, 0.0, " = 0.0"),
    ]
)
def test_power_success_finite_results(
    mock_calculator_instance: MagicMock,
    base: float,
    exponent: float,
    expected_result: float,
    expected_history_suffix: str
):
    """
    Tests the Calculator.power method for various inputs resulting in finite numbers,
    verifying the calculation and history update.
    """
    # 1. Patch the mock instance's 'power' method to use the real implementation
    # This ensures the actual calculation and history appending logic is executed.
    mock_calculator_instance.power.side_effect = Calculator.power.__get__(
        mock_calculator_instance, Calculator
    )

    initial_history_length = len(mock_calculator_instance.history)

    # 2. Call the method
    result = mock_calculator_instance.power(base, exponent)

    # 3. Assert the return value, using approx for float precision
    assert result == pytest.approx(expected_result)

    # 4. Assert history update (side effect)
    expected_history_entry = f"{base} ^ {exponent}{expected_history_suffix}"
    assert len(mock_calculator_instance.history) == initial_history_length + 1
    assert mock_calculator_instance.history[-1] == expected_history_entry


def test_power_zero_base_negative_exponent_inf(mock_calculator_instance: MagicMock):
    """
    Tests the Calculator.power method for the edge case 0.0 ** -n, which results
    in positive infinity (inf), verifying the result and history logging of 'inf'.
    This ensures 100% coverage for potential floating point edge cases.
    """
    base = 0.0
    exponent = -1.0

    # 1. Patch the mock instance's 'power' method to use the real implementation
    mock_calculator_instance.power.side_effect = Calculator.power.__get__(
        mock_calculator_instance, Calculator
    )

    initial_history_length = len(mock_calculator_instance.history)

    # 2. Call the method
    result = mock_calculator_instance.power(base, exponent)

    # 3. Assert the return value is positive infinity
    assert result == float('inf')

    # 4. Assert history update
    # f-string formatting of float('inf') results in 'inf'
    expected_history_entry = f"{base} ^ {exponent} = inf"
    assert len(mock_calculator_instance.history) == initial_history_length + 1
    assert mock_calculator_instance.history[-1] == expected_history_entry
# Code added at 20251022-153015
import pytest
from unittest.mock import MagicMock, patch
import math # Required for expected results calculation if not mocking return values directly

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "number, expected_result",
    [
        # 1. Perfect square positive number
        (9.0, 3.0),
        # 2. Zero (Edge case)
        (0.0, 0.0),
        # 3. Positive float
        (2.25, 1.5),
        # 4. Large number
        (10000.0, 100.0),
    ]
)
@patch('sample_code.calculator.math')
def test_square_root_success(mock_math, mock_calculator_instance: MagicMock, number: float, expected_result: float):
    """
    Tests the Calculator.square_root method for successful calculation (number >= 0),
    verifying the correct result, dependency call (math.sqrt), and history update.
    Achieves coverage for the successful execution path.
    """
    # 1. Configure the dependency mock (math.sqrt)
    mock_math.sqrt.return_value = expected_result

    # 2. Patch the mock instance's 'square_root' method to use the real implementation
    mock_calculator_instance.square_root.side_effect = Calculator.square_root.__get__(
        mock_calculator_instance, Calculator
    )

    initial_history_length = len(mock_calculator_instance.history)

    # 3. Call the method
    result = mock_calculator_instance.square_root(number)

    # 4. Assert the return value
    assert result == pytest.approx(expected_result)

    # 5. Assert dependency call
    mock_math.sqrt.assert_called_once_with(number)

    # 6. Assert history update (side effect)
    expected_history_entry = f"√{number} = {expected_result}"
    assert len(mock_calculator_instance.history) == initial_history_length + 1
    assert mock_calculator_instance.history[-1] == expected_history_entry


@pytest.mark.parametrize("number", [-1.0, -0.001, -100.0])
@patch('sample_code.calculator.math')
def test_square_root_negative_raises_value_error(mock_math, mock_calculator_instance: MagicMock, number: float):
    """
    Tests the Calculator.square_root method when the input is negative, ensuring
    a ValueError is raised and history is not updated.
    Achieves coverage for the negative number error branch.
    """
    # 1. Patch the mock instance's 'square_root' method to use the real implementation
    mock_calculator_instance.square_root.side_effect = Calculator.square_root.__get__(
        mock_calculator_instance, Calculator
    )

    initial_history_length = len(mock_calculator_instance.history)

    # 2. Assert that ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        mock_calculator_instance.square_root(number)

    # 3. Assert the error message
    assert "Cannot calculate square root of negative number" in str(excinfo.value)

    # 4. Assert dependency was NOT called
    mock_math.sqrt.assert_not_called()

    # 5. Assert history was NOT updated
    assert len(mock_calculator_instance.history) == initial_history_length
# Code added at 20251022-153028
import pytest
from unittest.mock import MagicMock, patch

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


# We patch the math module used inside calculator.py
@patch('sample_code.calculator.math')
@pytest.mark.parametrize(
    "n, expected_result",
    [
        # 1. Standard positive integer
        (5, 120),
        # 2. Edge case: Zero (0! = 1)
        (0, 1),
        # 3. Small positive integer
        (1, 1),
    ]
)
def test_factorial_success(mock_math, mock_calculator_instance: MagicMock, n: int, expected_result: int):
    """
    Tests the Calculator.factorial method for successful calculation (n >= 0 and integer),
    verifying the result, the call to math.factorial, and history update.
    Covers the main execution path.
    """
    # 1. Configure the dependency mock (math.factorial)
    # We mock the return value since the real math.factorial is an external dependency
    mock_math.factorial.return_value = expected_result

    # 2. Patch the mock instance's 'factorial' method to use the real implementation
    # This ensures the internal logic (error checks, history update) runs.
    mock_calculator_instance.factorial.side_effect = Calculator.factorial.__get__(
        mock_calculator_instance, Calculator
    )

    initial_history_length = len(mock_calculator_instance.history)

    # 3. Call the method
    result = mock_calculator_instance.factorial(n)

    # 4. Assert the return value
    assert result == expected_result

    # 5. Assert dependency call
    mock_math.factorial.assert_called_once_with(n)

    # 6. Assert history update (side effect)
    expected_history_entry = f"{n}! = {expected_result}"
    assert len(mock_calculator_instance.history) == initial_history_length + 1
    assert mock_calculator_instance.history[-1] == expected_history_entry


@patch('sample_code.calculator.math')
@pytest.mark.parametrize("n", [-1, -5])
def test_factorial_error_negative_raises_value_error(mock_math, mock_calculator_instance: MagicMock, n: int):
    """
    Tests the Calculator.factorial method when the input n is negative, ensuring
    a ValueError is raised and history is not updated.
    Covers the 'if n < 0' branch.
    """
    # 1. Patch the mock instance's 'factorial' method to use the real implementation
    mock_calculator_instance.factorial.side_effect = Calculator.factorial.__get__(
        mock_calculator_instance, Calculator
    )

    initial_history_length = len(mock_calculator_instance.history)

    # 2. Assert that ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        mock_calculator_instance.factorial(n)

    # 3. Assert the error message
    assert "Factorial is not defined for negative numbers" in str(excinfo.value)

    # 4. Assert dependency was NOT called
    mock_math.factorial.assert_not_called()

    # 5. Assert history was NOT updated
    assert len(mock_calculator_instance.history) == initial_history_length


@patch('sample_code.calculator.math')
@pytest.mark.parametrize("n", [5.0, "a string", None, 1.5])
def test_factorial_error_non_integer_raises_type_error(mock_math, mock_calculator_instance: MagicMock, n):
    """
    Tests the Calculator.factorial method when the input n is not an integer, ensuring
    a TypeError is raised and history is not updated.
    Covers the 'if not isinstance(n, int)' branch.
    """
    # 1. Patch the mock instance's 'factorial' method to use the real implementation
    mock_calculator_instance.factorial.side_effect = Calculator.factorial.__get__(
        mock_calculator_instance, Calculator
    )

    initial_history_length = len(mock_calculator_instance.history)

    # 2. Assert that TypeError is raised
    with pytest.raises(TypeError) as excinfo:
        mock_calculator_instance.factorial(n)

    # 3. Assert the error message
    assert "Factorial input must be an integer" in str(excinfo.value)

    # 4. Assert dependency was NOT called
    mock_math.factorial.assert_not_called()

    # 5. Assert history was NOT updated
    assert len(mock_calculator_instance.history) == initial_history_length
# Code added at 20251022-153033
import pytest
from unittest.mock import MagicMock

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


def test_clear_history_resets_list(mock_calculator_instance: MagicMock):
    """
    Tests the Calculator.clear_history method, ensuring that the internal
    history list is successfully reset to an empty list.
    Achieves 100% coverage for this method.
    """
    # 1. Patch the mock instance's 'clear_history' method to use the real implementation
    mock_calculator_instance.clear_history.side_effect = Calculator.clear_history.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Setup: Populate the history list to ensure it's not empty initially
    mock_calculator_instance.history = ["1 + 1 = 2", "5 * 5 = 25"]
    assert len(mock_calculator_instance.history) > 0

    # 3. Call the method
    mock_calculator_instance.clear_history()

    # 4. Assert the side effect: history must be an empty list
    assert mock_calculator_instance.history == []
    assert len(mock_calculator_instance.history) == 0
# Code added at 20251022-153041
import pytest
from unittest.mock import MagicMock
from typing import List

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


def test_get_history_empty(mock_calculator_instance: MagicMock):
    """
    Tests Calculator.get_history when the history is empty, ensuring an empty list is returned.
    Covers the case where self.history is initialized but empty.
    """
    # 1. Patch the mock instance's 'get_history' method to use the real implementation
    mock_calculator_instance.get_history.side_effect = Calculator.get_history.__get__(
        mock_calculator_instance, Calculator
    )

    # Ensure history is empty (default fixture state)
    assert mock_calculator_instance.history == []

    # 2. Call the method
    history = mock_calculator_instance.get_history()

    # 3. Assert the result
    assert history == []
    assert isinstance(history, list)


def test_get_history_populated_and_is_copy(mock_calculator_instance: MagicMock):
    """
    Tests Calculator.get_history when history is populated, verifying the content
    and ensuring the returned list is a defensive copy (using .copy()) to prevent
    external modification of the internal state.
    Achieves 100% coverage for the method logic.
    """
    # 1. Patch the mock instance's 'get_history' method to use the real implementation
    mock_calculator_instance.get_history.side_effect = Calculator.get_history.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Setup: Populate the history list
    initial_history = ["1 + 1 = 2", "10 / 2 = 5.0", "5! = 120"]
    mock_calculator_instance.history = initial_history

    # 3. Call the method
    returned_history = mock_calculator_instance.get_history()

    # 4. Assert content matches
    assert returned_history == initial_history

    # 5. Assert it is a copy (identity check: returned object is not the internal object)
    assert returned_history is not mock_calculator_instance.history

    # 6. Verify defensive copy: Modify the returned list
    returned_history.append("External modification")

    # Internal history should remain unchanged
    assert mock_calculator_instance.history == initial_history
    assert len(mock_calculator_instance.history) == 3
# Code added at 20251022-153046
import pytest
from unittest.mock import MagicMock

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "value_to_store",
    [
        # 1. Positive float
        123.45,
        # 2. Negative float
        -50.0,
        # 3. Zero (Edge case)
        0.0,
        # 4. Large number
        1e6,
    ]
)
def test_store_memory_success(mock_calculator_instance: MagicMock, value_to_store: float):
    """
    Tests the Calculator.store_memory method, ensuring that the input value is
    correctly assigned to the internal self.memory attribute.
    Achieves 100% coverage for the method logic.
    """
    # 1. Patch the mock instance's 'store_memory' method to use the real implementation
    # This ensures the actual assignment logic (self.memory = value) is executed.
    mock_calculator_instance.store_memory.side_effect = Calculator.store_memory.__get__(
        mock_calculator_instance, Calculator
    )

    # Ensure initial memory state is 0.0 (from fixture)
    assert mock_calculator_instance.memory == 0.0

    # 2. Call the method
    mock_calculator_instance.store_memory(value_to_store)

    # 3. Assert the side effect: memory must be updated to the stored value
    assert mock_calculator_instance.memory == pytest.approx(value_to_store)
# Code added at 20251022-153053
import pytest
from unittest.mock import MagicMock

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "initial_memory, expected_result",
    [
        # 1. Initial state (0.0)
        (0.0, 0.0),
        # 2. Positive stored value
        (15.7, 15.7),
        # 3. Negative stored value
        (-99.9, -99.9),
    ]
)
def test_recall_memory_returns_current_value(
    mock_calculator_instance: MagicMock,
    initial_memory: float,
    expected_result: float
):
    """
    Tests the Calculator.recall_memory method, ensuring it correctly returns the
    current value stored in the internal self.memory attribute, covering initial
    zero state and stored positive/negative values.
    Achieves 100% coverage for the method logic.
    """
    # 1. Patch the mock instance's 'recall_memory' method to use the real implementation
    # This ensures the actual attribute access logic (return self.memory) is executed.
    mock_calculator_instance.recall_memory.side_effect = Calculator.recall_memory.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Setup: Set the internal memory state of the mock instance
    mock_calculator_instance.memory = initial_memory

    # 3. Call the method
    result = mock_calculator_instance.recall_memory()

    # 4. Assert the return value matches the memory state
    assert result == pytest.approx(expected_result)
    assert isinstance(result, float)
# Code added at 20251022-153058
import pytest
from unittest.mock import MagicMock

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "initial_value",
    [
        100.0,
        -50.5,
        1e6,
    ]
)
def test_clear_memory_resets_to_zero(mock_calculator_instance: MagicMock, initial_value: float):
    """
    Tests the Calculator.clear_memory method, ensuring that the internal
    self.memory attribute is successfully reset to 0.0, regardless of its
    previous value.
    Achieves 100% coverage for the method logic.
    """
    # 1. Patch the mock instance's 'clear_memory' method to use the real implementation
    # This ensures the actual assignment logic (self.memory = 0.0) is executed.
    mock_calculator_instance.clear_memory.side_effect = Calculator.clear_memory.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Setup: Set the internal memory state to a non-zero value
    mock_calculator_instance.memory = initial_value
    assert mock_calculator_instance.memory == pytest.approx(initial_value)

    # 3. Call the method
    mock_calculator_instance.clear_memory()

    # 4. Assert the side effect: memory must be reset to 0.0
    assert mock_calculator_instance.memory == pytest.approx(0.0)
# Code added at 20251022-153105
import pytest
from unittest.mock import MagicMock

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "value, percent, expected_result",
    [
        # 1. Normal positive calculation (10% of 100 is 10)
        (100.0, 10.0, 10.0),
        # 2. 50% calculation
        (200.0, 50.0, 100.0),
        # 3. Percentage is zero
        (50.0, 0.0, 0.0),
        # 4. Value is zero
        (0.0, 25.0, 0.0),
        # 5. Negative percentage (-20% of 100 is -20)
        (100.0, -20.0, -20.0),
        # 6. Negative value (10% of -50 is -5)
        (-50.0, 10.0, -5.0),
        # 7. Floating point result requiring precision check
        (3.0, 33.33, 0.9999),
        # 8. Large numbers
        (10000.0, 1.5, 150.0),
    ]
)
def test_percentage_calculation(mock_calculator_instance: MagicMock, value: float, percent: float, expected_result: float):
    """
    Tests the Calculator.percentage method for various inputs, ensuring the correct
    arithmetic calculation (value * percent / 100) is performed.
    Achieves 100% coverage for this stateless method.
    """
    # 1. Patch the mock instance's 'percentage' method to use the real implementation
    # This ensures the actual calculation logic is executed.
    mock_calculator_instance.percentage.side_effect = Calculator.percentage.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Call the method
    result = mock_calculator_instance.percentage(value, percent)

    # 3. Assert the return value, using approx for float precision
    assert result == pytest.approx(expected_result)

    # 4. Verify no side effects on state (optional, but good practice for stateless methods)
    assert mock_calculator_instance.history == []
    assert mock_calculator_instance.memory == 0.0
# Code added at 20251022-153113
import pytest
from unittest.mock import MagicMock
from typing import List

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "numbers, expected_result",
    [
        # 1. Standard positive numbers
        ([1.0, 2.0, 3.0], 2.0),
        # 2. Mixed positive and negative numbers
        ([10.0, -5.0, 1.0], 2.0),  # Sum=6.0, Count=3, Avg=2.0
        # 3. Single element list
        ([5.5], 5.5),
        # 4. Zeroes
        ([0.0, 0.0, 0.0], 0.0),
        # 5. Floating point numbers requiring precision check
        ([1.0, 2.0, 4.0], 7.0 / 3.0),
    ]
)
def test_average_success(mock_calculator_instance: MagicMock, numbers: List[float], expected_result: float):
    """
    Tests the Calculator.average method for successful calculation with non-empty lists,
    verifying the correct arithmetic result.
    Covers the successful execution path (return sum(numbers) / len(numbers)).
    """
    # 1. Patch the mock instance's 'average' method to use the real implementation
    mock_calculator_instance.average.side_effect = Calculator.average.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Call the method
    result = mock_calculator_instance.average(numbers)

    # 3. Assert the return value, using approx for float precision
    assert result == pytest.approx(expected_result)


def test_average_empty_list_raises_value_error(mock_calculator_instance: MagicMock):
    """
    Tests the Calculator.average method when provided with an empty list, ensuring
    a ValueError is raised as required by the method specification.
    Covers the 'if not numbers:' branch.
    """
    numbers: List[float] = []

    # 1. Patch the mock instance's 'average' method to use the real implementation
    mock_calculator_instance.average.side_effect = Calculator.average.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Assert that ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        mock_calculator_instance.average(numbers)

    # 3. Assert the error message
    assert "Cannot calculate average of empty list" in str(excinfo.value)
# Code added at 20251022-153121
import pytest
from unittest.mock import MagicMock
from typing import List

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "numbers, expected_result",
    [
        # 1. Standard positive numbers
        ([1.0, 5.0, 2.0], 5.0),
        # 2. Mixed positive and negative numbers
        ([-10.0, 0.0, 10.0, -5.0], 10.0),
        # 3. Negative numbers only
        ([-10.0, -5.0, -20.0], -5.0),
        # 4. Single element list (Edge case)
        ([3.14], 3.14),
        # 5. Floating point numbers
        ([1.1, 1.11, 1.09], 1.11),
    ]
)
def test_max_value_success(mock_calculator_instance: MagicMock, numbers: List[float], expected_result: float):
    """
    Tests the Calculator.max_value method for successful execution with various
    non-empty lists, ensuring the correct maximum value is returned.
    Covers the successful execution path (return max(numbers)).
    """
    # 1. Patch the mock instance's 'max_value' method to use the real implementation
    mock_calculator_instance.max_value.side_effect = Calculator.max_value.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Call the method
    result = mock_calculator_instance.max_value(numbers)

    # 3. Assert the return value
    assert result == pytest.approx(expected_result)


def test_max_value_empty_list_raises_value_error(mock_calculator_instance: MagicMock):
    """
    Tests the Calculator.max_value method when provided with an empty list, ensuring
    a ValueError is raised as required by the method specification.
    Covers the 'if not numbers:' branch, achieving 100% coverage.
    """
    numbers: List[float] = []

    # 1. Patch the mock instance's 'max_value' method to use the real implementation
    mock_calculator_instance.max_value.side_effect = Calculator.max_value.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Assert that ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        mock_calculator_instance.max_value(numbers)

    # 3. Assert the error message
    assert "Cannot find max of empty list" in str(excinfo.value)
# Code added at 20251022-153128
import pytest
from unittest.mock import MagicMock
from typing import List

# Import the class Calculator from the correct relative path
# file_path: sample_code/calculator.py
# unittest_path: sample_code/unittest/test_calculator.py
try:
    from ..calculator import Calculator
except ImportError:
    # Fallback for different execution contexts
    from sample_code.calculator import Calculator


@pytest.mark.parametrize(
    "numbers, expected_result",
    [
        # 1. Standard positive numbers
        ([1.0, 5.0, 2.0], 1.0),
        # 2. Mixed positive and negative numbers
        ([-10.0, 0.0, 10.0, -5.0], -10.0),
        # 3. Negative numbers only
        ([-10.0, -5.0, -20.0], -20.0),
        # 4. Single element list (Edge case)
        ([3.14], 3.14),
        # 5. Floating point numbers
        ([1.1, 1.11, 1.09], 1.09),
    ]
)
def test_min_value_success(mock_calculator_instance: MagicMock, numbers: List[float], expected_result: float):
    """
    Tests the Calculator.min_value method for successful execution with various
    non-empty lists, ensuring the correct minimum value is returned.
    Covers the successful execution path (return min(numbers)).
    """
    # 1. Patch the mock instance's 'min_value' method to use the real implementation
    mock_calculator_instance.min_value.side_effect = Calculator.min_value.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Call the method
    result = mock_calculator_instance.min_value(numbers)

    # 3. Assert the return value
    assert result == pytest.approx(expected_result)


def test_min_value_empty_list_raises_value_error(mock_calculator_instance: MagicMock):
    """
    Tests the Calculator.min_value method when provided with an empty list, ensuring
    a ValueError is raised as required by the method specification.
    Covers the 'if not numbers:' branch, achieving 100% coverage.
    """
    numbers: List[float] = []

    # 1. Patch the mock instance's 'min_value' method to use the real implementation
    mock_calculator_instance.min_value.side_effect = Calculator.min_value.__get__(
        mock_calculator_instance, Calculator
    )

    # 2. Assert that ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        mock_calculator_instance.min_value(numbers)

    # 3. Assert the error message
    assert "Cannot find min of empty list" in str(excinfo.value)
