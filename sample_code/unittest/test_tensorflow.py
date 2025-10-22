# Code added at 20251022-152034
import pytest
from unittest.mock import MagicMock

# Standard library imports
# Third-party imports
# Local application imports
try:
    # Importing BatchTest from sample_code/tensorflow.py
    from sample_code.tensorflow import BatchTest
except ImportError:
    # Note: If the execution environment requires relative imports or path adjustments,
    # this block handles potential import failures during setup, but the instruction
    # is to use the derived path directly.
    pass

"""
- Import necessary libraries (pytest, MagicMock) and the target class (BatchTest).
- Define the pytest fixture `mock_BatchTest_instance`.
- Initialize the fixture using MagicMock(spec=BatchTest).
- Ensure the fixture adheres to naming conventions and includes a descriptive docstring.
"""

@pytest.fixture
def mock_BatchTest_instance() -> MagicMock:
    """
    Pytest fixture for creating a MagicMock instance of the BatchTest class.

    This mock is used to isolate the system under test from the actual
    implementation of BatchTest, ensuring tests are repeatable and independent.
    Since no constructor dependencies were specified, this mock is initialized
    with only the spec of the class.
    """
    # Initialize the mock instance with the specification of the actual class
    mock_BatchTest = MagicMock(spec=BatchTest)

    # If BatchTest had constructor arguments or methods requiring specific mock behavior,
    # that setup would occur here. Since no dependencies were provided,
    # the mock remains a simple spec instance.

    return mock_BatchTest
# Code added at 20251022-152042
import pytest
from unittest.mock import MagicMock

# Standard library imports (None required beyond unittest.mock)

# Third-party imports (pytest)

# Local application imports
try:
    # Importing BatchCheckpointTest from sample_code/tensorflow.py relative to
    # sample_code/unittest/test_tensorflow.py
    from ..tensorflow import BatchCheckpointTest
except ImportError:
    # This block handles potential import errors if the execution environment
    # changes the module path (e.g., running tests from the project root).
    # If the import fails, the mock creation will fail, indicating an environment issue.
    pass

"""
- Import necessary testing utilities (pytest, MagicMock).
- Determine and implement the correct relative import for BatchCheckpointTest.
- Define the Pytest fixture `mock_BatchCheckpointTest_instance`.
- Initialize the mock using MagicMock(spec=BatchCheckpointTest).
- Provide a descriptive docstring for the fixture.
"""

@pytest.fixture
def mock_BatchCheckpointTest_instance() -> MagicMock:
    """
    A MagicMock instance simulating the BatchCheckpointTest class from tensorflow.py.

    This fixture is used to isolate the system under test (SUT) from the actual
    implementation of BatchCheckpointTest, allowing control over its behavior
    and preventing side effects during testing.

    Since no constructor dependencies were specified, this mock is initialized
    with a basic spec.
    """
    # Initialize the mock instance with the specification of the actual class
    mock_instance = MagicMock(spec=BatchCheckpointTest)

    # No specific methods or attributes are mocked here as the constructor
    # usage details were not provided in the schema.

    return mock_instance
# Code added at 20251022-152046
import unittest.mock
from unittest.mock import MagicMock
from sample_code.tensorflow import BatchRandomAccessTest

# Checklist:
# - Determine and implement necessary imports for the target class and MagicMock.
# - Define the Pytest fixture `mock_BatchRandomAccessTest_instance`.
# - Initialize the fixture using `MagicMock(spec=BatchRandomAccessTest)`.
# - Add a descriptive docstring to the fixture.

def mock_BatchRandomAccessTest_instance() -> MagicMock:
    """
    Pytest fixture for mocking the BatchRandomAccessTest class.

    This fixture creates a MagicMock instance specified to match the
    interface of BatchRandomAccessTest, ensuring isolation for unit tests.
    Since no specific constructor dependencies were provided, this is a
    basic spec mock.
    """
    # Initialize the mock instance with the spec of the actual class
    mock_BatchRandomAccessTest = MagicMock(spec=BatchRandomAccessTest)

    # No specific methods or attributes were listed for mocking,
    # so we return the basic spec mock.

    return mock_BatchRandomAccessTest

# Code added at 20251022-152052
import unittest.mock
from unittest.mock import MagicMock
import pytest

# Determine the correct relative import path for the target class
# Target class: BatchGlobalShuffleTest
# Source file: sample_code/tensorflow.py
# Module path: sample_code.tensorflow
try:
    from sample_code.tensorflow import BatchGlobalShuffleTest
except ImportError:
    # If running outside the specific project structure, this might fail.
    # Assuming the execution environment allows importing sample_code.tensorflow
    pass

"""
- Define necessary imports (unittest.mock, pytest, target class).
- Determine the correct module path for the target class.
- Create a Pytest fixture named mock_batchglobalshuffletest_instance.
- Initialize the fixture using MagicMock(spec=BatchGlobalShuffleTest).
- Add a descriptive docstring to the fixture.
"""

@pytest.fixture
def mock_batchglobalshuffletest_instance() -> MagicMock:
    """
    Pytest fixture for creating a MagicMock instance of BatchGlobalShuffleTest.

    This mock is used to isolate tests from the actual implementation of
    BatchGlobalShuffleTest, ensuring repeatable and independent test execution.
    Since no constructor details were provided, this mock is initialized with
    only the spec of the class.
    """
    # Initialize the mock instance with the specification of the actual class
    mock_instance = MagicMock(spec=BatchGlobalShuffleTest)

    # If BatchGlobalShuffleTest had specific constructor dependencies (variables,
    # or methods used during initialization), they would be set here.
    # Example: mock_instance.some_attribute = []
    # Example: mock_instance.some_method.return_value = expected_result

    return mock_instance
# Code added at 20251022-152101
"""
- Determine the correct module path for the target class (`BatchGlobalShuffleCheckpointTest`).
- Import the target class and necessary mocking utilities (`pytest`, `MagicMock`).
- Define the MagicMock fixture using the required naming convention.
- Ensure the mock is initialized with the correct class specification (`spec`).
- Provide a descriptive docstring for clarity.
"""
import pytest
from unittest.mock import MagicMock

# Import the target class from the specified file path: sample_code/tensorflow.py
from sample_code.tensorflow import BatchGlobalShuffleCheckpointTest

@pytest.fixture
def mock_BatchGlobalShuffleCheckpointTest_instance() -> MagicMock:
    """
    Pytest fixture for mocking the BatchGlobalShuffleCheckpointTest class instance.

    This mock is initialized with the spec of the actual class, ensuring that
    any attempts to call non-existent methods or attributes will raise appropriate
    errors, maintaining test integrity. This isolates the component under test
    from complex TensorFlow checkpoint logic.
    """
    # Initialize the mock instance with the spec of the actual class
    mock_instance = MagicMock(spec=BatchGlobalShuffleCheckpointTest)

    # Since no constructor dependencies or methods were specified in the input schema,
    # we initialize a basic mock instance ready for use.

    return mock_instance

# Code added at 20251022-152122

# Code added at 20251022-152145
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest # Required for type hinting/fixture resolution

# Define a mock exception class to simulate the TensorFlow error
class MockInvalidArgumentError(Exception):
    """Mock class for errors.InvalidArgumentError."""
    pass

# Fixture definition (assuming mock_BatchTest_instance is already defined and imported)
# We rely on the provided fixture `mock_BatchTest_instance`.

# Checklist:
# 1. Use `mock_BatchTest_instance` fixture.
# 2. Patch external dependencies (`dataset_ops`, `errors`).
# 3. Simulate the execution flow where the invalid batch size causes the expected error.
# 4. Verify that `self.assertRaises` is called with the correct exception type.
# 5. Verify that the dataset creation and evaluation chain are executed.

@patch('sample_code.tensorflow.dataset_ops')
@patch('sample_code.tensorflow.errors')
def test_testInvalidBatchSize_success(mock_errors: MagicMock, mock_dataset_ops: MagicMock, mock_BatchTest_instance: MagicMock):
    """
    Tests the testInvalidBatchSize method ensures that attempting to create a
    dataset batch with size 0 correctly triggers an InvalidArgumentError,
    verifying the execution flow and assertion setup.
    """
    # --- Setup Mocks ---

    # 1. Setup Mock Exception Type
    # We assign our local mock exception to the patched errors module
    # mock_
# Code added at 20251022-152158
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest # Required for type resolution

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

# Checklist:
# 1. Use the provided `mock_BatchTest_instance` fixture.
# 2. Patch the external dependency `sample_code.tensorflow.dataset_ops`.
# 3. Mock the complex chained calls (`range`, `map`, `batch`, `unbatch`, `flat_map`).
# 4. Verify that the internal assertion method (`self.assertDatasetProduces`) is called correctly.
# 5. Ensure 100% coverage for the linear execution path.

@patch('sample_code.tensorflow.dataset_ops')
def test_testDataset_success(mock_dataset_ops: MagicMock, mock_BatchTest_instance: MagicMock):
    """
    Tests the testDataset method to ensure the complex dataset operation chain
    (range, map, batch, unbatch, flat_map) executes correctly and calls the
    final assertion method with the expected output range.
    """
    # --- Setup Mocks ---

    # 1. Mock the dataset object returned by the initial call (Dataset.range(10))
    mock_dataset = MagicMock()
    mock_dataset_ops.Dataset.range.return_value = mock_dataset

    # 2. Configure the chained methods to return the mock_dataset itself,
    # simulating the continuous chaining of dataset transformations.
    mock_dataset.map.return_value = mock_dataset
    mock_dataset.batch.return_value = mock_dataset
    mock_dataset.unbatch.return_value = mock_dataset
    mock_dataset.flat_map.return_value = mock_dataset

    # 3. Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces

    # --- Execution ---
    mock_BatchTest_instance.testDataset()

    # --- Assertions ---

    # 1. Verify the starting point of the chain
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)

    # 2. Verify the sequence of chained calls
    
    # The chain is: .map(map_fn).batch(5).map(lambda x: x).unbatch().flat_map(lambda x: x)
    
    # 'map' is called twice in the method body
    assert mock_dataset.map.call_count == 2
    
    # Check the batch size
    mock_dataset.batch.assert_called_once_with(5)
    
    # Check unbatch and flat_map calls
    mock_dataset.unbatch.assert_called_once()
    mock_dataset.flat_map.assert_called_once()

    # 3. Verify the final assertion call
    # The final dataset object passed is the result of the last flat_map (which is mock_dataset)
    mock_assertDatasetProduces.assert_called_once_with(
        mock_dataset, expected_output=range(10)
    )
# Code added at 20251022-152218
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest  # Required for type resolution and context

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

@patch('sample_code.tensorflow.sparse_tensor')
@patch('sample_code.tensorflow.dataset_ops')
def test_testSparse_success(mock_dataset_ops: MagicMock, mock_sparse_tensor: MagicMock, mock_BatchTest_instance: MagicMock):
    """
    Tests the testSparse method to ensure the dataset creation chain (range, map, batch)
    is correctly executed and the final assertion method (`assertDatasetProduces`)
    is called with the resulting dataset and the correctly calculated expected sparse tensors.
    """
    # List to capture the mock objects created specifically for the `expected_output` list
    expected_output_mocks = []
    
    # --- Setup Mocks ---

    # 1. Setup Dataset Chain Mocks
    mock_dataset = MagicMock(name="Dataset")
    # Mock the start of the chain: Dataset.range(10)
    mock_dataset_ops.Dataset.range.return_value = mock_dataset
    # Mock the chained methods to return the dataset mock itself
    mock_dataset.map.return_value = mock_dataset
    mock_dataset.batch.return_value = mock_dataset
    
    # 2. Setup SparseTensorValue Mock
    def mock_sparse_tensor_value(*args, **kwargs):
        """Custom side effect to track calls and identify objects created for expected_output."""
        m = MagicMock(name="SparseTensorValue_Instance")
        
        # Heuristic check: The expected_output calculation uses indices=[[0, 0], [1, 0], ...] (5 indices).
        # The internal _sparse function uses indices=[[0]] (1 index).
        # We capture the mocks created when indices list has length 5.
        if args and isinstance(args[0], list) and len(args[0]) == 5:
            expected_output_mocks.append(m)
            
        return m

    mock_sparse_tensor.SparseTensorValue.side_effect = mock_sparse_tensor_value
    
    # 3. Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces
    
    # --- Execution ---
    mock_BatchTest_instance.testSparse()
    
    # --- Assertions ---
    
    # 1. Verify Dataset Chain calls
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)
    mock_dataset.map.assert_called_once()
    mock_dataset.batch.assert_called_once_with(5)
    
    # 2. Verify SparseTensorValue calls (10 inside _sparse + 2 for expected_output)
    assert mock_sparse_tensor.SparseTensorValue.call_count == 12
    assert len(expected_output_mocks) == 2
    
    # 3. Verify Final Assertion Call
    mock_assertDatasetProduces.assert_called_once()
    
    # Check arguments passed to assertDatasetProduces
    call_args, call_kwargs = mock_assertDatasetProduces.call_args
    
    # Verify the dataset object passed
    assert call_args[0] is mock_dataset
    
    # Verify the expected_output list structure and content
    assert 'expected_output' in call_kwargs
    output_list = call_kwargs['expected_output']
    
    assert isinstance(output_list, list)
    assert len(output_list) == 2
    
    # Ensure the list contains the specific mock objects generated during the calculation
    assert output_list[0] is expected_output_mocks[0]
    assert output_list[1] is expected_output_mocks[1]
# Code added at 20251022-152235
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

@patch('sample_code.tensorflow.sparse_tensor')
@patch('sample_code.tensorflow.dataset_ops')
def test_testSparseWithDifferentDenseShapes_success(
    mock_dataset_ops: MagicMock, 
    mock_sparse_tensor: MagicMock, 
    mock_BatchTest_instance: MagicMock
):
    """
    Tests the testSparseWithDifferentDenseShapes method to ensure the dataset chain 
    (range, map, batch) is correctly constructed and the final assertion method 
    (`self.assertDatasetProduces`) is called with the resulting dataset mock and 
    the two calculated expected SparseTensorValue objects.
    """
    
    # --- Setup Dataset Chain Mocks ---
    mock_dataset = MagicMock(name="Dataset")
    # Mock the start of the chain: Dataset.range(10)
    mock_dataset_ops.Dataset.range.return_value = mock_dataset
    # Mock the chained methods to return the dataset mock itself
    mock_dataset.map.return_value = mock_dataset
    mock_dataset.batch.return_value = mock_dataset
    
    # --- Setup SparseTensorValue Mock Tracking ---
    
    expected_output_mocks = []
    
    def sparse_value_side_effect(*args, **kwargs):
        """Tracks the mock objects created for the expected_output list."""
        m = MagicMock(name="SparseTensorValue_Instance")
        expected_output_mocks.append(m)
        return m

    # The calculation of expected_output executes two calls to SparseTensorValue
    mock_sparse_tensor.SparseTensorValue.side_effect = sparse_value_side_effect
    
    # Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces
    
    # --- Execution ---
    mock_BatchTest_instance.testSparseWithDifferentDenseShapes()
    
    # --- Assertions ---
    
    # 1. Verify Dataset Chain calls
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)
    mock_dataset.map.assert_called_once()
    mock_dataset.batch.assert_called_once_with(5)
    
    # 2. Verify SparseTensorValue calls (2 calls for expected_output)
    assert mock_sparse_tensor.SparseTensorValue.call_count == 2
    assert len(expected_output_mocks) == 2
    
    # 3. Verify Final Assertion Call
    mock_assertDatasetProduces.assert_called_once()
    
    call_args, call_kwargs = mock_assertDatasetProduces.call_args
    
    # Verify the dataset object passed is the result of the chain
    assert call_args[0] is mock_dataset
    
    # Verify the expected_output list content
    assert 'expected_output' in call_kwargs
    output_list = call_kwargs['expected_output']
    
    assert isinstance(output_list, list)
    assert len(output_list) == 2
    
    # Ensure the list contains the specific mock objects generated
    assert output_list[0] is expected_output_mocks[0]
    assert output_list[1] is expected_output_mocks[1]
    
    # Optional: Verify structural arguments passed to SparseTensorValue
    # Call 1 (i=0): dense_shape=[5, 4]
    assert mock_sparse_tensor.SparseTensorValue.call_args_list[0].kwargs['dense_shape'] == [5, 4]
    # Call 2 (i=1): dense_shape=[5, 9]
    assert mock_sparse_tensor.SparseTensorValue.call_args_list[1].kwargs['dense_shape'] == [5, 9]
# Code added at 20251022-152250
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

@patch('sample_code.tensorflow.sparse_tensor')
@patch('sample_code.tensorflow.dataset_ops')
def test_testSparseNested_success(
    mock_dataset_ops: MagicMock, 
    mock_sparse_tensor: MagicMock, 
    mock_BatchTest_instance: MagicMock
):
    """
    Tests the testSparseNested method to ensure the complex nested batching dataset 
    chain is correctly constructed and the final assertion method 
    (`self.assertDatasetProduces`) is called with the resulting dataset mock and 
    the single calculated expected SparseTensorValue object.
    """
    
    # --- Setup Dataset Chain Mocks ---
    
    # Define mocks for each stage of the dataset chain: range -> map -> batch(5) -> batch(2)
    mock_dataset_start = MagicMock(name="DatasetStart")
    mock_dataset_map = MagicMock(name="DatasetMap")
    mock_dataset_batch5 = MagicMock(name="DatasetBatch5")
    mock_dataset_final = MagicMock(name="DatasetFinal")
    
    # 1. Mock Dataset.range(10)
    mock_dataset_ops.Dataset.range.return_value = mock_dataset_start
    
    # 2. Mock the chained methods
    mock_dataset_start.map.return_value = mock_dataset_map
    mock_dataset_map.batch.return_value = mock_dataset_batch5
    mock_dataset_batch5.batch.return_value = mock_dataset_final
    
    # --- Setup SparseTensorValue Mock ---
    
    # The expected_output list requires exactly one call to SparseTensorValue
    expected_output_mock = MagicMock(name="ExpectedOutputSparseTensor")
    mock_sparse_tensor.SparseTensorValue.return_value = expected_output_mock
    
    # Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces
    
    # --- Execution ---
    mock_BatchTest_instance.testSparseNested()
    
    # --- Assertions ---
    
    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)
    
    # Verify map call (the internal _sparse function is passed as argument)
    mock_dataset_start.map.assert_called_once()
    
    # Verify the two batch calls
    mock_dataset_map.batch.assert_called_once_with(5)
    mock_dataset_batch5.batch.assert_called_once_with(2)
    
    # 2. Verify SparseTensorValue call (only one explicit call for expected_output)
    mock_sparse_tensor.SparseTensorValue.assert_called_once()
    
    # Check arguments passed to SparseTensorValue for expected_output definition
    call_kwargs = mock_sparse_tensor.SparseTensorValue.call_args[1]
    assert call_kwargs['dense_shape'] == [2, 5, 1]
    assert call_kwargs['values'] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # 3. Verify Final Assertion Call
    mock_assertDatasetProduces.assert_called_once()
    
    call_args, call_kwargs = mock_assertDatasetProduces.call_args
    
    # Verify the dataset object passed is the result of the final batch(2)
    assert call_args[0] is mock_dataset_final
    
    # Verify the expected_output list content
    assert 'expected_output' in call_kwargs
    output_list = call_kwargs['expected_output']
    
    assert isinstance(output_list, list)
    assert len(output_list) == 1
    assert output_list[0] is expected_output_mock
# Code added at 20251022-152300
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

# Checklist:
# 1. Use the provided `mock_BatchTest_instance` fixture.
# 2. Patch external dependencies: `dataset_ops`, `dtypes`, and `errors`.
# 3. Mock the dataset creation chain (`from_generator().batch(3)`).
# 4. Verify that `self.assertDatasetProduces` is called with the resulting dataset mock and the expected error tuple.

@patch('sample_code.tensorflow.errors')
@patch('sample_code.tensorflow.dtypes')
@patch('sample_code.tensorflow.dataset_ops')
def test_testShapeError_raises_expected_error(
    mock_dataset_ops: MagicMock,
    mock_dtypes: MagicMock,
    mock_errors: MagicMock,
    mock_BatchTest_instance: MagicMock
):
    """
    Tests the testShapeError method ensures that batching a dataset containing
    elements of different shapes correctly triggers the expected InvalidArgumentError
    via the assertDatasetProduces assertion.
    """
    # --- Setup Mocks ---

    # 1. Setup Dataset Chain Mocks
    mock_final_dataset = MagicMock(name="FinalDataset")
    mock_intermediate_dataset = MagicMock(name="IntermediateDataset")

    # Mock the chain: from_generator(...) -> intermediate_dataset
    mock_dataset_ops.Dataset.from_generator.return_value = mock_intermediate_dataset

    # Mock the chain: intermediate_dataset.batch(3) -> final_dataset
    mock_intermediate_dataset.batch.return_value = mock_final_dataset

    # 2. Setup Error and Dtypes Mocks
    mock_invalid_argument_error = MagicMock(name="InvalidArgumentError")
    mock_errors.InvalidArgumentError = mock_invalid_argument_error
    mock_float32 = MagicMock(name="float32")
    mock_dtypes.float32 = mock_float32

    # 3. Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces

    # --- Execution ---
    mock_BatchTest_instance.testShapeError()

    # --- Assertions ---

    # 1. Verify Dataset creation call (from_generator)
    mock_dataset_ops.Dataset.from_generator.assert_called_once()
    call_args = mock_dataset_ops.Dataset.from_generator.call_args[0]

    # Verify arguments passed to from_generator
    # Arg 0: generator function (check if it's callable)
    assert callable(call_args[0])
    # Arg 1: dtypes.float32
    assert call_args[1] is mock_float32
    # Arg 2: output_shapes=[None]
    assert call_args[2] == [None]

    # 2. Verify batch call
    mock_intermediate_dataset.batch.assert_called_once_with(3)

    # 3. Verify final assertion call
    expected_error_message = (
        r"Cannot batch tensors with different shapes in component 0. First "
        r"element had shape \[3\] and element 2 had shape \[4\]."
    )

    mock_assertDatasetProduces.assert_called_once_with(
        mock_final_dataset,
        expected_error=(
            mock_invalid_argument_error,
            expected_error_message,
        ),
    )
# Code added at 20251022-152311
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

# Checklist:
# 1. Use the provided `mock_BatchTest_instance` fixture.
# 2. Patch external dependencies: `dataset_ops`, `ragged_tensor`, and `ragged_factory_ops`.
# 3. Mock the dataset creation chain (`range().map().batch(5)`).
# 4. Mock `ragged_factory_ops.constant` to capture the expected output objects.
# 5. Verify that `self.assertDatasetProduces` is called correctly with the final dataset mock and the expected output mocks.

@patch('sample_code.tensorflow.ragged_factory_ops')
@patch('sample_code.tensorflow.ragged_tensor')
@patch('sample_code.tensorflow.dataset_ops')
def test_testRagged_success(
    mock_dataset_ops: MagicMock,
    mock_ragged_tensor: MagicMock,
    mock_ragged_factory_ops: MagicMock,
    mock_BatchTest_instance: MagicMock
):
    """
    Tests the testRagged method to ensure the dataset chain for ragged tensors
    (range, map, batch) is correctly constructed and the final assertion method
    (`self.assertDatasetProduces`) is called with the resulting dataset mock and
    the two calculated expected RaggedTensor objects.
    """
    # --- Setup Dataset Chain Mocks ---

    # Define mocks for each stage of the dataset chain: range -> map -> batch(5)
    mock_dataset_start = MagicMock(name="DatasetStart")
    mock_dataset_map = MagicMock(name="DatasetMap")
    mock_dataset_final = MagicMock(name="DatasetFinal")

    # 1. Mock Dataset.range(10)
    mock_dataset_ops.Dataset.range.return_value = mock_dataset_start

    # 2. Mock the chained methods
    mock_dataset_start.map.return_value = mock_dataset_map
    mock_dataset_map.batch.return_value = mock_dataset_final

    # --- Setup Expected Output Mocks ---

    # The expected_output list requires exactly two calls to ragged_factory_ops.constant
    expected_output_mocks = [
        MagicMock(name="ExpectedRaggedTensor1"),
        MagicMock(name="ExpectedRaggedTensor2")
    ]
    mock_ragged_factory_ops.constant.side_effect = expected_output_mocks

    # Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces

    # --- Execution ---
    mock_BatchTest_instance.testRagged()

    # --- Assertions ---

    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)

    # Verify map call (it is called with the local _ragged function)
    mock_dataset_start.map.assert_called_once()
    
    # Verify batch call
    mock_dataset_map.batch.assert_called_once_with(5)

    # 2. Verify RaggedTensor creation for expected output
    assert mock_ragged_factory_ops.constant.call_count == 2
    
    # We don't need to verify the internal calls to ragged_tensor.RaggedTensor.from_tensor
    # because that happens inside the local _ragged function which is passed to the mocked .map()
    # and doesn't affect the mocked chain structure.

    # 3. Verify Final Assertion Call
    mock_assertDatasetProduces.assert_called_once()

    call_args, call_kwargs = mock_assertDatasetProduces.call_args

    # Verify the dataset object passed is the result of the final batch(5)
    assert call_args[0] is mock_dataset_final

    # Verify the expected_output list content
    assert 'expected_output' in call_kwargs
    output_list = call_kwargs['expected_output']

    assert isinstance(output_list, list)
    assert len(output_list) == 2
    assert output_list == expected_output_mocks
# Code added at 20251022-152323
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

@patch('sample_code.tensorflow.ragged_concat_ops')
@patch('sample_code.tensorflow.ragged_math_ops')
@patch('sample_code.tensorflow.dataset_ops')
def test_testRaggedWithDifferentShapes_success(
    mock_dataset_ops: MagicMock,
    mock_ragged_math_ops: MagicMock,
    mock_ragged_concat_ops: MagicMock,
    mock_BatchTest_instance: MagicMock
):
    """
    Tests the testRaggedWithDifferentShapes method to ensure the dataset chain
    (range, map, batch) is correctly constructed and the expected output calculation
    using ragged operations is performed, leading to a correct call to
    self.assertDatasetProduces.
    """
    # --- Setup Dataset Chain Mocks ---

    # Define mocks for each stage of the dataset chain: range -> map -> batch(5)
    mock_dataset_start = MagicMock(name="DatasetStart")
    mock_dataset_map = MagicMock(name="DatasetMap")
    mock_dataset_final = MagicMock(name="DatasetFinal")

    # 1. Mock Dataset.range(10)
    mock_dataset_ops.Dataset.range.return_value = mock_dataset_start

    # 2. Mock the chained methods
    mock_dataset_start.map.return_value = mock_dataset_map
    mock_dataset_map.batch.return_value = mock_dataset_final

    # --- Setup Expected Output Calculation Mocks ---

    # 1. Mock ragged_math_ops.range (called 10 times in total: 0..4 and 5..9)
    range_mocks = [MagicMock(name=f"Range_{i}") for i in range(10)]
    mock_ragged_math_ops.range.side_effect = range_mocks

    # 2. Mock ragged_concat_ops.stack (called 2 times for the final expected output list)
    expected_output_mocks = [
        MagicMock(name="ExpectedRaggedTensor1"),
        MagicMock(name="ExpectedRaggedTensor2")
    ]
    mock_ragged_concat_ops.stack.side_effect = expected_output_mocks

    # Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces

    # --- Execution ---
    mock_BatchTest_instance.testRaggedWithDifferentShapes()

    # --- Assertions ---

    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)
    mock_dataset_start.map.assert_called_once()
    mock_dataset_map.batch.assert_called_once_with(5)

    # 2. Verify intermediate ragged range calls (10 total)
    assert mock_ragged_math_ops.range.call_count == 10
    # Check arguments passed to ragged_math_ops.range (0, 1, 2, ..., 9)
    expected_range_calls = [((i,), {}) for i in range(10)]
    assert mock_ragged_math_ops.range.call_args_list == expected_range_calls

    # 3. Verify final stack calls (2 total)
    assert mock_ragged_concat_ops.stack.call_count == 2

    # Verify Call 1: Stacking ranges 0 through 4
    call1_args = mock_ragged_concat_ops.stack.call_args_list[0][0][0]
    assert len(call1_args) == 5
    assert call1_args == range_mocks[:5]

    # Verify Call 2: Stacking ranges 5 through 9
    call2_args = mock_ragged_concat_ops.stack.call_args_list[1][0][0]
    assert len(call2_args) == 5
    assert call2_args == range_mocks[5:]

    # 4. Verify Final Assertion Call
    mock_assertDatasetProduces.assert_called_once()

    call_args, call_kwargs = mock_assertDatasetProduces.call_args

    # Verify the dataset object passed
    assert call_args[0] is mock_dataset_final

    # Verify the expected_output list content
    assert 'expected_output' in call_kwargs
    output_list = call_kwargs['expected_output']

    assert isinstance(output_list, list)
    assert len(output_list) == 2
    assert output_list == expected_output_mocks
# Code added at 20251022-152336
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

@patch('sample_code.tensorflow.ragged_factory_ops')
@patch('sample_code.tensorflow.ragged_tensor')
@patch('sample_code.tensorflow.dataset_ops')
def test_testRaggedNested_success(
    mock_dataset_ops: MagicMock,
    mock_ragged_tensor: MagicMock,
    mock_ragged_factory_ops: MagicMock,
    mock_BatchTest_instance: MagicMock
):
    """
    Tests the testRaggedNested method to ensure the complex nested batching dataset
    chain (range, map, batch(5), batch(2)) is correctly constructed and the final
    assertion method (`self.assertDatasetProduces`) is called with the resulting
    dataset mock and the single calculated expected RaggedTensor object.
    """

    # --- Setup Dataset Chain Mocks ---

    # Define mocks for each stage of the dataset chain
    mock_dataset_start = MagicMock(name="DatasetStart")
    mock_dataset_map = MagicMock(name="DatasetMap")
    mock_dataset_batch5 = MagicMock(name="DatasetBatch5")
    mock_dataset_final = MagicMock(name="DatasetFinal")

    # 1. Mock Dataset.range(10)
    mock_dataset_ops.Dataset.range.return_value = mock_dataset_start

    # 2. Mock the chained methods
    mock_dataset_start.map.return_value = mock_dataset_map
    mock_dataset_map.batch.return_value = mock_dataset_batch5
    mock_dataset_batch5.batch.return_value = mock_dataset_final

    # 3. Mock the internal ragged tensor creation (used inside the _ragged map function)
    mock_ragged_tensor.RaggedTensor.from_tensor.return_value = MagicMock(name="InternalRaggedTensor")

    # --- Setup Expected Output Mock ---

    # The expected_output list requires exactly one call to ragged_factory_ops.constant
    expected_output_mock = MagicMock(name="ExpectedOutputRaggedTensor")
    mock_ragged_factory_ops.constant.return_value = expected_output_mock

    # Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces

    # --- Execution ---
    mock_BatchTest_instance.testRaggedNested()

    # --- Assertions ---

    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)

    # Verify map call (the internal _ragged function is passed as argument)
    mock_dataset_start.map.assert_called_once()

    # Verify the two batch calls
    mock_dataset_map.batch.assert_called_once_with(5)
    mock_dataset_batch5.batch.assert_called_once_with(2)

    # 2. Verify RaggedTensor creation for expected output
    mock_ragged_factory_ops.constant.assert_called_once()

    # 3. Verify Final Assertion Call
    mock_assertDatasetProduces.assert_called_once()

    call_args, call_kwargs = mock_assertDatasetProduces.call_args

    # Verify the dataset object passed is the result of the final batch(2)
    assert call_args[0] is mock_dataset_final

    # Verify the expected_output list content
    assert 'expected_output' in call_kwargs
    output_list = call_kwargs['expected_output']

    assert isinstance(output_list, list)
    assert len(output_list) == 1
    assert output_list[0] is expected_output_mock
# Code added at 20251022-152343
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest # Used for fixture context

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

# Checklist:
# 1. Use the provided `mock_BatchTest_instance` fixture.
# 2. Patch external dependency `sample_code.tensorflow.dataset_ops`.
# 3. Mock the four-step dataset chain (range, map, batch, map).
# 4. Verify all chain calls use the correct parameters (range(10), batch(10)).
# 5. Assert that `self.assertDatasetProduces` is called with the final dataset mock and the expected output.

@patch('sample_code.tensorflow.dataset_ops')
def test_testNoneComponent_success(
    mock_dataset_ops: MagicMock,
    mock_BatchTest_instance: MagicMock
):
    """
    Tests the testNoneComponent method to ensure the dataset chain correctly handles
    a component being set to None during mapping and subsequent batching,
    and verifies the final assertion call with the expected output.
    """
    # --- Setup Mocks for Dataset Chain ---

    # Define mocks for each stage of the dataset chain: range -> map -> batch -> map
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_map1 = MagicMock(name="DatasetMap1")
    mock_ds_batch = MagicMock(name="DatasetBatch")
    mock_ds_final = MagicMock(name="DatasetFinal")

    # 1. Mock Dataset.range(10)
    mock_dataset_ops.Dataset.range.return_value = mock_ds_start

    # 2. Mock the chained methods
    mock_ds_start.map.return_value = mock_ds_map1
    mock_ds_map1.batch.return_value = mock_ds_batch
    mock_ds_batch.map.return_value = mock_ds_final

    # Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces

    # --- Execution ---
    mock_BatchTest_instance.testNoneComponent()

    # --- Assertions ---

    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)

    # Verify the first map call (lambda x: (x, None))
    assert mock_ds_start.map.call_count == 1
    
    # Verify the batch call
    mock_ds_map1.batch.assert_called_once_with(10)

    # Verify the second map call (lambda x, y: x)
    mock_ds_batch.map.assert_called_once()

    # 2. Define the expected output
    expected_output = [list(range(10))]

    # 3. Verify Final Assertion Call
    mock_assertDatasetProduces.assert_called_once_with(
        mock_ds_final,
        expected_output=expected_output
    )
# Code added at 20251022-152401
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

# Define the combinations for local_determinism and global_determinism
# These combinations cover all branches of the 'expect_determinism' calculation:
# expect_determinism = local_determinism or (local_determinism is None and global_determinism)
DETERMINISM_COMBINATIONS = [
    # local | global | expected_determinism | case_name
    (True, False, True, "local_true_overrides"),
    (False, True, False, "local_false_no_override"),
    (None, True, True, "local_none_global_true"),
    (None, False, False, "local_none_global_false"),
]

@pytest.mark.parametrize(
    "local_determinism, global_determinism, expected_determinism, case_name",
    DETERMINISM_COMBINATIONS
)
@patch('sample_code.tensorflow.options_lib')
@patch('sample_code.tensorflow.script_ops')
@patch('sample_code.tensorflow.math_ops')
@patch('sample_code.tensorflow.dataset_ops')
def test_testDeterminismConfiguration_flow(
    mock_dataset_ops: MagicMock,
    mock_math_ops: MagicMock,
    mock_script_ops: MagicMock,
    mock_options_lib: MagicMock,
    mock_BatchTest_instance: MagicMock,
    local_determinism,
    global_determinism,
    expected_determinism,
    case_name
):
    """
    Tests the testDeterminismConfiguration method across various combinations of
    local and global determinism settings.

    It verifies:
    1. The calculation of `expect_determinism` based on input parameters.
    2. The correct construction of the dataset pipeline using mocked dependencies.
    3. The final call to `self.checkDeterminism` with the calculated expected value.
    """
    # --- Setup Dataset Chain Mocks ---

    # Define mocks for the chained dataset operations
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_map = MagicMock(name="DatasetMap")
    mock_ds_batch = MagicMock(name="DatasetBatch")
    mock_ds_unbatch = MagicMock(name="DatasetUnbatch")
    mock_ds_final = MagicMock(name="DatasetFinal")

    # Mock the start of the chain: Dataset.from_tensor_slices
    mock_dataset_ops.Dataset.from_tensor_slices.return_value = mock_ds_start

    # Mock the chained methods
    mock_ds_start.map.return_value = mock_ds_map
    mock_ds_map.batch.return_value = mock_ds_batch
    mock_ds_batch.unbatch.return_value = mock_ds_unbatch
    mock_ds_unbatch.with_options.return_value = mock_ds_final

    # Mock Options setup
    mock_options = MagicMock(name="OptionsInstance")
    mock_options_lib.Options.return_value = mock_options

    # Mock the final assertion method
    mock_checkDeterminism = mock_BatchTest_instance.checkDeterminism

    # --- Execution ---
    mock_BatchTest_instance.testDeterminismConfiguration(
        local_determinism, global_determinism
    )

    # --- Assertions ---

    # 1. Verify Dataset Chain construction
    expected_elements = list(range(100))
    mock_dataset_ops.Dataset.from_tensor_slices.assert_called_once_with(expected_elements)

    # Verify map call parameters
    mock_ds_start.map.assert_called_once()
    map_call_kwargs = mock_ds_start.map.call_args[1]
    assert map_call_kwargs['num_parallel_calls'] == 2
    assert map_call_kwargs['deterministic'] == local_determinism

    # Verify batch call parameters
    mock_ds_map.batch.assert_called_once()
    batch_call_kwargs = mock_ds_map.batch.call_args[1]
    assert batch_call_kwargs['batch_size'] == 6
    assert batch_call_kwargs['num_parallel_calls'] == 2
    assert batch_call_kwargs['deterministic'] == local_determinism

    # Verify unbatch call
    mock_ds_batch.unbatch.assert_called_once()

    # Verify Options setup and assignment
    mock_options_lib.Options.assert_called_once()
    assert mock_options.deterministic == global_determinism

    # Verify with_options call
    mock_ds_unbatch.with_options.assert_called_once_with(mock_options)

    # 2. Verify Final Assertion Call
    mock_checkDeterminism.assert_called_once()

    call_args = mock_checkDeterminism.call_args[0]

    # Arg 0: dataset_fn (must be callable)
    assert callable(call_args[0])

    # Arg 1: expect_determinism (calculated value)
    assert call_args[1] == expected_determinism

    # Arg 2: elements
    assert call_args[2] == expected_elements
# Code added at 20251022-152420
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest

# Note: The fixture `mock_BatchTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchTest).

@patch('builtins.next')
@patch('builtins.iter')
@patch('sample_code.tensorflow.checkpoint_management')
@patch('sample_code.tensorflow.trackable_utils')
@patch('sample_code.tensorflow.dtypes')
@patch('sample_code.tensorflow.array_ops')
@patch('sample_code.tensorflow.dataset_ops')
def test_testCheckpointLargeBatches_success(
    mock_dataset_ops: MagicMock,
    mock_array_ops: MagicMock,
    mock_dtypes: MagicMock,
    mock_trackable_utils: MagicMock,
    mock_checkpoint_management: MagicMock,
    mock_iter: MagicMock,
    mock_next: MagicMock,
    mock_BatchTest_instance: MagicMock
):
    """
    Tests the testCheckpointLargeBatches method to ensure the sequence of
    creating a large batched dataset, consuming the first element via an
    iterator, and saving a checkpoint is executed correctly using mocked
    TensorFlow dependencies.
    """
    # --- Setup Dataset Chain Mocks ---

    # 1. Mock Tensor and Dtype
    mock_tensor = MagicMock(name="Tensor")
    mock_float32 = MagicMock(name="float32")
    mock_array_ops.ones.return_value = mock_tensor
    mock_dtypes.float32 = mock_float32

    # 2. Mock Dataset Chain stages
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_repeat = MagicMock(name="DatasetRepeat")
    mock_ds_final = MagicMock(name="DatasetFinal")

    mock_dataset_ops.Dataset.from_tensors.return_value = mock_ds_start
    mock_ds_start.repeat.return_value = mock_ds_repeat
    mock_ds_repeat.batch.return_value = mock_ds_final

    # --- Setup Iterator Mocks ---
    mock_iterator = MagicMock(name="Iterator")
    # iter(dataset) -> iterator
    mock_iter.return_value = mock_iterator

    # --- Setup Checkpoint Mocks ---
    mock_ckpt = MagicMock(name="CheckpointInstance")
    mock_trackable_utils.Checkpoint.return_value = mock_ckpt

    mock_manager = MagicMock(name="CheckpointManagerInstance")
    mock_checkpoint_management.CheckpointManager.return_value = mock_manager

    # --- Setup Instance Mocks ---
    mock_BatchTest_instance.get_temp_dir.return_value = "/mock/temp/dir"

    # --- Execution ---
    mock_BatchTest_instance.testCheckpointLargeBatches()

    # --- Assertions ---

    # 1. Verify Tensor creation (512M tensor)
    mock_array_ops.ones.assert_called_once_with(
        (64, 1024, 1024), dtype=mock_float32
    )

    # 2. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.from_tensors.assert_called_once_with(mock_tensor)
    mock_ds_start.repeat.assert_called_once()
    # Verify batch size (2) and parallel calls (5)
    mock_ds_repeat.batch.assert_called_once_with(2, num_parallel_calls=5)

    # 3. Verify Iterator usage
    mock_iter.assert_called_once_with(mock_ds_final)
    mock_next.assert_called_once_with(mock_iterator)

    # 4. Verify Checkpoint creation
    mock_trackable_utils.Checkpoint.assert_called_once_with(iterator=mock_iterator)

    # 5. Verify CheckpointManager creation
    mock_checkpoint_management.CheckpointManager.assert_called_once_with(
        mock_ckpt, "/mock/temp/dir", max_to_keep=1
    )

    # 6. Verify Save call
    mock_manager.save.assert_called_once()
# Code added at 20251022-152430
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchTest

# Define parameter combinations for num_parallel_calls to cover typical usage
# (None for default, 1 for sequential, >1 for parallel)
NUM_PARALLEL_CALLS_CASES = [
    (None, "default_none"),
    (1, "sequential_one"),
    (2, "parallel_two"),
]

@pytest.mark.parametrize(
    "num_parallel_calls, case_name",
    NUM_PARALLEL_CALLS_CASES
)
@patch('sample_code.tensorflow.dataset_ops')
def test_testName_execution(
    mock_dataset_ops: MagicMock,
    mock_BatchTest_instance: MagicMock,
    num_parallel_calls,
    case_name
):
    """
    Tests the testName method across different `num_parallel_calls` values.

    Verifies that:
    1. The dataset chain (range -> batch) is constructed correctly.
    2. The `num_parallel_calls` and `name="batch"` arguments are passed to .batch().
    3. The final assertion method (`self.assertDatasetProduces`) is called with
       the resulting dataset mock and the expected output.
    """
    # --- Setup Dataset Chain Mocks ---

    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_final = MagicMock(name="DatasetFinal")

    # 1. Mock Dataset.range(5)
    mock_dataset_ops.Dataset.range.return_value = mock_ds_start

    # 2. Mock the chained method: .batch(...)
    mock_ds_start.batch.return_value = mock_ds_final

    # Mock the assertion method on the instance
    mock_assertDatasetProduces = mock_BatchTest_instance.assertDatasetProduces

    # --- Execution ---
    # Invoke the method with the parameterized input
    mock_BatchTest_instance.testName(num_parallel_calls)

    # --- Assertions ---

    # 1. Verify Dataset Chain start
    mock_dataset_ops.Dataset.range.assert_called_once_with(5)

    # 2. Verify batch call parameters
    mock_ds_start.batch.assert_called_once()

    call_args, call_kwargs = mock_ds_start.batch.call_args

    # Check positional argument: batch size 5
    assert call_args[0] == 5

    # Check keyword arguments
    assert call_kwargs['num_parallel_calls'] == num_parallel_calls
    assert call_kwargs['name'] == "batch"

    # 3. Verify Final Assertion Call
    expected_output = [list(range(5))]

    mock_assertDatasetProduces.assert_called_once_with(
        mock_ds_final,
        expected_output
    )
# Code added at 20251022-152441
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchCheckpointTest  # Required for context

# Note: The fixture `mock_BatchCheckpointTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchCheckpointTest).

# Define parameter combinations for num_parallel_calls and options
BUILD_DATASET_CASES = [
    # multiplier, tensor_slice_len, batch_size, num_parallel_calls, options_provided, expected_with_options_call, case_name
    (15.0, 2, 2, None, False, False, "default_params"),
    (10.0, 5, 3, 4, False, False, "custom_inputs_parallel_calls"),
    (1.0, 1, 1, None, True, True, "minimal_with_options"),
]

@pytest.mark.parametrize(
    "multiplier, tensor_slice_len, batch_size, num_parallel_calls, options_provided, expected_with_options_call, case_name",
    BUILD_DATASET_CASES
)
@patch('sample_code.tensorflow.dataset_ops')
@patch('sample_code.tensorflow.np')
def test__build_dataset_coverage(
    mock_np: MagicMock,
    mock_dataset_ops: MagicMock,
    mock_BatchCheckpointTest_instance: MagicMock,
    multiplier,
    tensor_slice_len,
    batch_size,
    num_parallel_calls,
    options_provided,
    expected_with_options_call,
    case_name
):
    """
    Tests the _build_dataset method covering default, custom, parallel calls,
    and the 'if options:' branch.
    """
    # --- Setup Mocks ---

    # 1. Mock NumPy operations
    mock_np.newaxis = "newaxis"
    # Use side_effect to return distinct mock objects for components
    mock_component_0 = MagicMock(name="Component0")
    mock_component_1 = MagicMock(name="Component1")
    mock_component_2 = MagicMock(name="Component2")

    # Mock np.arange and np.array calls used to create components
    mock_np.arange.return_value = mock_component_0  # Used for component 0 and intermediate calculation
    mock_np.array.side_effect = [
        mock_component_1,  # Used for component 1
        mock_component_2,  # Used for component 2
    ]

    # 2. Mock Dataset Chain
    mock_ds_start = MagicMock(name="ds_start")
    mock_ds_batched = MagicMock(name="ds_batched")
    mock_ds_final = MagicMock(name="ds_final")

    mock_dataset_ops.Dataset.from_tensor_slices.return_value = mock_ds_start
    mock_ds_start.batch.return_value = mock_ds_batched
    mock_ds_batched.with_options.return_value = mock_ds_final

    # 3. Setup Options Mock (if required)
    mock_options = MagicMock(name="Options") if options_provided else None

    # --- Execution ---
    result_dataset = mock_BatchCheckpointTest_instance._build_dataset(
        multiplier=multiplier,
        tensor_slice_len=tensor_slice_len,
        batch_size=batch_size,
        num_parallel_calls=num_parallel_calls,
        options=mock_options,
    )

    # --- Assertions ---

    # 1. Verify NumPy calls for component creation
    # np.arange(tensor_slice_len) is called once for component 0
    mock_np.arange.assert_called_once_with(tensor_slice_len)
    
    # np.array is called twice
    assert mock_np.array.call_count == 2

    # 2. Verify Dataset creation
    expected_components = (
        mock_component_0,
        mock_component_1,
        mock_component_2,
    )
    mock_dataset_ops.Dataset.from_tensor_slices.assert_called_once_with(
        expected_components
    )

    # 3. Verify batch call
    mock_ds_start.batch.assert_called_once_with(
        batch_size, num_parallel_calls=num_parallel_calls
    )

    # 4. Verify options branch
    if expected_with_options_call:
        mock_ds_batched.with_options.assert_called_once_with(mock_options)
        assert result_dataset is mock_ds_final
    else:
        mock_ds_batched.with_options.assert_not_called()
        assert result_dataset is mock_ds_batched
# Code added at 20251022-152455
import pytest
from unittest.mock import MagicMock, patch

# Define parameter combinations to cover different symbolic_checkpoint and num_parallel_calls states
TEST_CASES = [
    (True, 4, "symbolic_true_parallel"),
    (False, None, "symbolic_false_default"),
]

@pytest.mark.parametrize(
    "symbolic_checkpoint, num_parallel_calls, case_name",
    TEST_CASES
)
@patch('sample_code.tensorflow.options_lib')
def test_test_execution_flow(
    mock_options_lib: MagicMock,
    mock_BatchCheckpointTest_instance: MagicMock,
    symbolic_checkpoint,
    num_parallel_calls,
    case_name
):
    """
    Tests the BatchCheckpointTest.test method execution flow, verifying that:
    1. options_lib.Options is created and configured with symbolic_checkpoint.
    2. The internal _build_dataset method is correctly wrapped in a lambda function.
    3. The external verify_fn is called with the instance, the dataset lambda, and the calculated num_outputs (4).
    4. The dataset lambda correctly calls _build_dataset with all required parameters.
    """
    # --- Setup Mocks ---
    
    # Mock the external verification function passed to the method
    mock_verify_fn = MagicMock(name="verify_fn")
    
    # Mock the Options object and its creation
    mock_options_instance = MagicMock(name="OptionsInstance")
    mock_options_lib.Options.return_value = mock_options_instance
    
    # Mock the internal method _build_dataset on the fixture instance
    mock_build_dataset = mock_BatchCheckpointTest_instance._build_dataset
    mock_dataset_result = MagicMock(name="DatasetResult")
    mock_build_dataset.return_value = mock_dataset_result
    
    # --- Execution ---
    mock_BatchCheckpointTest_instance.test(
        mock_verify_fn, symbolic_checkpoint, num_parallel_calls
    )
    
    # --- Assertions (External Calls) ---
    
    # 1. Verify Options creation and configuration
    mock_options_lib.Options.assert_called_once()
    # Check that the symbolic_checkpoint flag was set correctly on the options instance
    assert mock_options_instance.experimental_symbolic_checkpoint == symbolic_checkpoint
    
    # 2. Verify verify_fn call
    mock_verify_fn.assert_called_once()
    
    call_args = mock_verify_fn.call_args[0]
    
    # Arg 0: self (the instance)
    assert call_args[0] is mock_BatchCheckpointTest_instance
    
    # Arg 1: dataset_fn (must be callable)
    dataset_fn = call_args[1]
    assert callable(dataset_fn)
    
    # Arg 2: num_outputs (8 // 2 = 4)
    assert call_args[2] == 4
    
    # --- Assertions (Internal Closure Execution) ---
    
    # Execute the dataset_fn closure to verify the call to _build_dataset
    mock_build_dataset.reset_mock()
    
    result = dataset_fn()
    
    # Verify the result is the mocked dataset
    assert result is mock_dataset_result
    
    # Verify _build_dataset was called with the correct hardcoded and input parameters
    mock_build_dataset.assert_called_once_with(
        15.0,  # multiplier
        8,     # tensor_slice_len
        2,     # batch_size
        num_parallel_calls,
        mock_options_instance
    )
# Code added at 20251022-152501
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchCheckpointTest

# Note: The fixture `mock_BatchCheckpointTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchCheckpointTest).

@pytest.mark.parametrize("i, expected_values", [
    (0, [0]),
    (5, [5]),
    (-3, [-3]),
])
@patch('sample_code.tensorflow.sparse_tensor')
def test__sparse_basic_creation(
    mock_sparse_tensor: MagicMock,
    mock_BatchCheckpointTest_instance: MagicMock,
    i: int,
    expected_values: list
):
    """
    Tests the _sparse helper method to ensure it correctly constructs and returns
    a SparseTensorValue object based on the input integer 'i'.

    Verifies that:
    1. The external dependency SparseTensorValue is called exactly once.
    2. The 'values' argument is correctly calculated as i * [1].
    3. The 'indices' and 'dense_shape' arguments are fixed as expected.
    """
    # --- Setup Mocks ---
    
    # Mock the return value of SparseTensorValue constructor
    mock_sparse_tensor_value = MagicMock(name="SparseTensorValue_Result")
    mock_sparse_tensor.SparseTensorValue.return_value = mock_sparse_tensor_value
    
    # --- Execution ---
    result = mock_BatchCheckpointTest_instance._sparse(i)
    
    # --- Assertions ---
    
    # 1. Verify the return value
    assert result is mock_sparse_tensor_value
    
    # 2. Verify the call to the dependency with correct arguments
    mock_sparse_tensor.SparseTensorValue.assert_called_once_with(
        indices=[[0]],
        values=expected_values,
        dense_shape=[1]
    )
# Code added at 20251022-152513
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchCheckpointTest

# Note: The fixture `mock_BatchCheckpointTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchCheckpointTest).

# Define parameter combinations to cover default and custom batch sizes
BATCH_SIZE_CASES = [
    (5, "default_size"),
    (3, "custom_size"),
    (10, "full_size"),
]

@pytest.mark.parametrize(
    "batch_size, case_name",
    BATCH_SIZE_CASES
)
@patch('sample_code.tensorflow.dataset_ops')
def test__build_dataset_sparse_execution(
    mock_dataset_ops: MagicMock,
    mock_BatchCheckpointTest_instance: MagicMock,
    batch_size: int,
    case_name: str
):
    """
    Tests the _build_dataset_sparse method to ensure the dataset chain 
    (range(10) -> map(self._sparse) -> batch(batch_size)) is constructed correctly 
    for various batch sizes, achieving 100% coverage for the method logic.
    """
    # --- Setup Mocks for Dataset Chain ---
    
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_map = MagicMock(name="DatasetMap")
    mock_ds_final = MagicMock(name="DatasetFinal")
    
    # 1. Mock Dataset.range(10)
    mock_dataset_ops.Dataset.range.return_value = mock_ds_start
    
    # 2. Mock the chained methods
    mock_ds_start.map.return_value = mock_ds_map
    mock_ds_map.batch.return_value = mock_ds_final
    
    # Get the mock reference for the internal method self._sparse
    mock_sparse_method = mock_BatchCheckpointTest_instance._sparse
    
    # --- Execution ---
    # Invoke the method with the parameterized batch_size
    result = mock_BatchCheckpointTest_instance._build_dataset_sparse(batch_size=batch_size)
    
    # --- Assertions ---
    
    # 1. Verify the return value is the result of the final .batch() call
    assert result is mock_ds_final
    
    # 2. Verify Dataset Chain start: range(10)
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)
    
    # 3. Verify map call uses the internal _sparse method
    mock_ds_start.map.assert_called_once_with(mock_sparse_method)
    
    # 4. Verify batch call uses the provided batch_size
    mock_ds_map.batch.assert_called_once_with(batch_size)
# Code added at 20251022-152520
import pytest
from unittest.mock import MagicMock

# Note: The fixture `mock_BatchCheckpointTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchCheckpointTest).

def test_testSparse_calls_verify_fn_correctly(mock_BatchCheckpointTest_instance: MagicMock):
    """
    Tests that testSparse correctly calls the provided verify_fn with the
    internal dataset builder method (_build_dataset_sparse) and num_outputs=2,
    ensuring the method's single execution path is covered.
    """
    # 1. Mock the external verification function passed as an argument
    mock_verify_fn = MagicMock(name="verify_fn")

    # 2. Get the reference to the internal method from the mock instance
    # This ensures we verify the reference passed is the correct internal method
    mock_build_dataset_sparse = mock_BatchCheckpointTest_instance._build_dataset_sparse

    # 3. Execution
    mock_BatchCheckpointTest_instance.testSparse(mock_verify_fn)

    # 4. Assertions
    # Verify that verify_fn was called exactly once with the required arguments
    mock_verify_fn.assert_called_once_with(
        mock_BatchCheckpointTest_instance,
        mock_build_dataset_sparse,
        num_outputs=2
    )
# Code added at 20251022-152527
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchCheckpointTest

# Note: The fixture `mock_BatchCheckpointTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchCheckpointTest).

@patch('sample_code.tensorflow.dataset_ops')
def test__build_dataset_nested_sparse_success(
    mock_dataset_ops: MagicMock,
    mock_BatchCheckpointTest_instance: MagicMock
):
    """
    Tests the _build_dataset_nested_sparse method to ensure the complex nested 
    dataset chain (range(10) -> map(self._sparse) -> batch(5) -> batch(2)) 
    is constructed correctly, achieving 100% coverage.
    """
    # --- Setup Mocks for Dataset Chain ---

    # Define mocks for each stage of the dataset chain
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_map = MagicMock(name="DatasetMap")
    mock_ds_batch5 = MagicMock(name="DatasetBatch5")
    mock_ds_final = MagicMock(name="DatasetFinal")

    # 1. Mock Dataset.range(10)
    mock_dataset_ops.Dataset.range.return_value = mock_ds_start

    # 2. Mock the chained methods
    mock_ds_start.map.return_value = mock_ds_map
    mock_ds_map.batch.return_value = mock_ds_batch5
    mock_ds_batch5.batch.return_value = mock_ds_final

    # Get the mock reference for the internal method self._sparse
    mock_sparse_method = mock_BatchCheckpointTest_instance._sparse

    # --- Execution ---
    result = mock_BatchCheckpointTest_instance._build_dataset_nested_sparse()

    # --- Assertions ---

    # 1. Verify the return value is the result of the final .batch(2) call
    assert result is mock_ds_final

    # 2. Verify Dataset Chain construction
    
    # Verify start: range(10)
    mock_dataset_ops.Dataset.range.assert_called_once_with(10)

    # Verify map call uses the internal _sparse method
    mock_ds_start.map.assert_called_once_with(mock_sparse_method)

    # Verify the first batch call: batch(5)
    mock_ds_map.batch.assert_called_once_with(5)

    # Verify the second (final) batch call: batch(2)
    mock_ds_batch5.batch.assert_called_once_with(2)
# Code added at 20251022-152534
import pytest
from unittest.mock import MagicMock
from sample_code.tensorflow import BatchCheckpointTest

# Note: The fixture `mock_BatchCheckpointTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchCheckpointTest).

def test_testNestedSparse_calls_verify_fn_correctly(mock_BatchCheckpointTest_instance: MagicMock):
    """
    Tests that testNestedSparse correctly calls the provided verify_fn with the
    internal dataset builder method (_build_dataset_nested_sparse) and num_outputs=1,
    covering the method's single execution path.
    """
    # 1. Mock the external verification function passed as an argument
    mock_verify_fn = MagicMock(name="verify_fn")

    # 2. Get the reference to the internal method from the mock instance
    # This ensures we verify the reference passed is the correct internal method
    mock_build_dataset_nested_sparse = mock_BatchCheckpointTest_instance._build_dataset_nested_sparse

    # 3. Execution
    mock_BatchCheckpointTest_instance.testNestedSparse(mock_verify_fn)

    # 4. Assertions
    # Verify that verify_fn was called exactly once with the required arguments
    mock_verify_fn.assert_called_once_with(
        mock_BatchCheckpointTest_instance,
        mock_build_dataset_nested_sparse,
        num_outputs=1
    )
# Code added at 20251022-152555
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchRandomAccessTest

# Define a mock exception class to simulate the TensorFlow error
class MockOutOfRangeError(Exception):
    """Mock class for errors.OutOfRangeError."""
    pass

@pytest.mark.parametrize("index", [
    2,   # Index too high (dataset has batches 0 and 1)
    -1,  # Negative index
])
@patch('sample_code.tensorflow.random_access')
@patch('sample_code.tensorflow.errors')
@patch('sample_code.tensorflow.dataset_ops')
def test_testInvalidIndex_raises_error(
    mock_dataset_ops: MagicMock,
    mock_errors: MagicMock,
    mock_random_access: MagicMock,
    mock_BatchRandomAccessTest_instance: MagicMock,
    index: int
):
    """
    Tests the testInvalidIndex method ensures that accessing an invalid batch index
    correctly sets up the self.assertRaises context manager with OutOfRangeError,
    and executes the random_access.at call, covering all branches (the single execution path).
    """
    # --- Setup Mocks ---

    # 1. Mock the expected exception type
    mock_errors.OutOfRangeError = MockOutOfRangeError

    # 2. Mock the dataset chain: from_tensor_slices([1, 2, 3, 4]).batch(2)
    mock_dataset = MagicMock(name="Dataset")
# Code added at 20251022-152605
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchRandomAccessTest # Required for context

# Define a mock exception class to simulate the TensorFlow error
# class MockOutOfRangeError(Exception):
#     """Mock class for errors.OutOfRangeError."""
#     pass

@patch('sample_code.tensorflow.random_access')
@patch('sample_code.tensorflow.errors')
@patch('sample_code.tensorflow.dataset_ops')
def test_testEmptyDataset_raises_error(
    mock_dataset_ops: MagicMock,
    mock_errors: MagicMock,
    mock_random_access: MagicMock,
    mock_BatchRandomAccessTest_instance: MagicMock
):
    """
    Tests the testEmptyDataset method ensures that attempting to access the 0th
    element of an empty batched dataset correctly triggers an OutOfRangeError,
    verifying the execution flow and assertion setup.
    """
    # --- Setup Mocks ---

    # 1. Mock the expected exception type
    mock_errors.OutOfRangeError = MockOutOfRangeError

    # 2. Mock the dataset chain: from_tensor_slices([]).batch(2)
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_final = MagicMock(name="DatasetFinal")

    mock_dataset_ops.Dataset.from_tensor_slices.return_value = mock_ds_start
    mock_ds_start.batch.return_value = mock_ds_final

    # 3. Mock the assertion methods on the instance
    mock_assertRaises = mock_BatchRandomAccessTest_instance.assertRaises
    mock_evaluate = mock_BatchRandomAccessTest_instance.evaluate

    # To simulate the test passing (i.e., the error is raised inside the context),
    # we mock self.evaluate to raise the expected error when called.
    mock_evaluate.side_effect = MockOutOfRangeError("Out of range")

    # --- Execution ---
    
    # We must mock the behavior of assertRaises: it should capture the error
    # raised by self.evaluate and suppress it, allowing the test function to complete.
    # Since we cannot easily mock the context manager behavior of assertRaises,
    # we rely on the fact that the method under test *calls* assertRaises with the
    # correct error type, and then calls evaluate with the correct arguments.
    
    # Since the method uses `with self.assertRaises(...)`, we execute the method
    # and verify the internal calls, relying on the mock setup to confirm the
    # intended error path was taken.
    
    # Note: If we were testing the assertRaises implementation itself, we'd need
    # a complex context manager mock. Here, we assume assertRaises works and verify
    # that the code inside the 'with' block executes and raises the expected error.
    
    try:
        mock_BatchRandomAccessTest_instance.testEmptyDataset()
    except MockOutOfRangeError:
        # If the mock evaluate raises the error, it means the test failed to mock
        # the context manager correctly. We proceed to verify calls regardless.
        pass
    
    # Reset evaluate side effect to check if it was called
    mock_evaluate.side_effect = None
    
    # --- Assertions ---

    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.from_tensor_slices.assert_called_once_with([])
    mock_ds_start.batch.assert_called_once_with(2)

    # 2. Verify random_access.at call
    mock_random_access.at.assert_called_once_with(mock_ds_final, 0)
    
    # 3. Verify self.evaluate was called with the result of random_access.at
    # We must re-run the execution path to verify the call to evaluate,
    # as the previous execution was interrupted by the side_effect.
    
    # Since the method is designed to raise an error, we verify the setup:
    
    # Verify that the instance was set up to catch the correct error type
    # (This is implicitly verified by checking the error type passed to assertRaises,
    # but since assertRaises is a method on the mock instance, we check the call args).
    
    # We cannot reliably check the call to assertRaises because it's a context manager.
    # We focus on the execution inside the context:
    
    # Re-execute the core logic to check the call to evaluate
    random_access_result = mock_random_access.at.return_value
    
    # We verify that evaluate was called with the result of random_access.at
    # Since the method is linear, we know evaluate is called immediately after random_access.at
    mock_evaluate.assert_called_once_with(random_access_result)
    
    # We verify that the correct error type was passed to the context manager setup
    mock_assertRaises.assert_called_once_with(MockOutOfRangeError)
# Code added at 20251022-152627

# Code added at 20251022-152708
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchRandomAccessTest

# Note: The fixture `mock_BatchRandomAccessTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchRandomAccessTest).

@patch('sample_code.tensorflow.np')
@patch('sample_code.tensorflow.random_access')
@patch('sample_code.tensorflow.dataset_ops')
def test_testRandomAccessBatchWithShuffle_success(
    mock_dataset_ops: MagicMock,
    mock_random_access: MagicMock,
    mock_np: MagicMock,
    mock_BatchRandomAccessTest_instance: MagicMock
):
    """
    Tests the testRandomAccessBatchWithShuffle method, verifying the correct
    construction of the shuffled and batched dataset chain and ensuring all 11
    assertions (using random_access.at and self.assertAllEqual) are executed
    with the correct dataset objects and indices.
    """
    # --- Setup Dataset Chain Mocks ---

    # Define mocks for the three dataset stages
    mock_dataset = MagicMock(name="Dataset")
    mock_shuffle_dataset = MagicMock(name="ShuffleDataset")
    mock_batch_dataset = MagicMock(name="BatchDataset")

    # 1. Mock Dataset.from_tensor_slices
    mock_dataset_ops.Dataset.from_tensor_slices.return_value = mock_dataset

    # 2. Mock the chained methods
    mock_dataset.shuffle.return_value = mock_shuffle_dataset
    mock_shuffle_dataset.batch.return_value = mock_batch_dataset

    # --- Setup Expected Output Mocks ---

    # Create mock objects that support indexing (for expected_output[i][0/1])
    def create_indexed_mock(name_prefix):
        m = MagicMock(name=name_prefix)
        # Mock __getitem__ to return distinct mocks for elements 0 and 1
        m.__getitem__.side_effect = lambda k: MagicMock(name=f"{name_prefix}_{k}")
        return m

    mock_expected_output_list = [
        create_indexed_mock("EO_0"),
        create_indexed_mock("EO_1"),
        create_indexed_mock("EO_2"),
        create_indexed_mock("EO_3"),
    ]

    # Mock np.array calls used to build the expected_output list (4 calls)
    mock_np.array.side_effect = mock_expected_output_list

    # --- Setup Assertion Mocks ---

    mock_evaluate = mock_BatchRandomAccessTest_instance.evaluate
    mock_assertAllEqual = mock_BatchRandomAccessTest_instance.assertAllEqual

    # random_access.at returns a mock operation object
    mock_random_access_op = MagicMock(name="RandomAccessOp")
    mock_random_access.at.return_value = mock_random_access_op

    # Define the 11 expected return values for self.evaluate
    # These must match the corresponding element being compared in assertAllEqual
    evaluate_returns = [
        # Loop 1 (Batched Dataset Check, i=0..3)
        mock_expected_output_list[0],
        mock_expected_output_list[1],
        mock_expected_output_list[2],
        mock_expected_output_list[3],

        # Loop 2 (Shuffled Dataset Check, i=0..2, index 0 and 1)
        mock_expected_output_list[0].__getitem__(0),
        mock_expected_output_list[0].__getitem__(1),
        mock_expected_output_list[1].__getitem__(0),
        mock_expected_output_list[1].__getitem__(1),
        mock_expected_output_list[2].__getitem__(0),
        mock_expected_output_list[2].__getitem__(1),

        # Final Check (Remainder)
        mock_expected_output_list[3].__getitem__(0),
    ]
    mock_evaluate.side_effect = evaluate_returns

    # --- Execution ---
    mock_BatchRandomAccessTest_instance.testRandomAccessBatchWithShuffle()

    # --- Assertions ---

    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.from_tensor_slices.assert_called_once_with(
        [1, 2, 3, 4, 5, 6, 7]
    )
    mock_dataset.shuffle.assert_called_once_with(buffer_size=10, seed=2)
    mock_shuffle_dataset.batch.assert_called_once_with(2)

    # 2. Verify np.array calls (4 times for expected_output setup)
    assert mock_np.array.call_count == 4

    # 3. Verify random_access.at calls (11 total)
    assert mock_random_access.at.call_count == 11
    
    # Verify the specific calls to random_access.at
    expected_at_calls = []
    
    # Loop 1: Batched dataset access (i=0 to 3)
    for i in range(4):
        expected_at_calls.append(((mock_batch_dataset, i), {}))
        
    # Loop 2: Shuffled dataset access (i=0 to 2, indices 0, 1, 2, 3, 4, 5)
    for i in range(3):
        expected_at_calls.append(((mock_shuffle_dataset, i * 2), {}))
        expected_at_calls.append(((mock_shuffle_dataset, (i * 2) + 1), {}))
        
    # Final Check: Shuffled dataset access (index 6)
    expected_at_calls.append(((mock_shuffle_dataset, 6), {}))

    assert mock_random_access.at.call_args_list == expected_at_calls

    # 4. Verify
# Code added at 20251022-152724
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchGlobalShuffleTest

# Note: The fixture `mock_batchglobalshuffletest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchGlobalShuffleTest).

# Define parameter combinations to cover different batching scenarios.
# We ensure that the mocked output (shuffled_output) is different from the
# calculated expected list, satisfying both assertCountEqual and assertNotEqual.
TEST_PARAMS = [
    # dataset_range, batch_size, expected_list (calculated), shuffled_output (mocked), case_name
    (10, 2, list(range(10)), [9, 8, 7, 6, 5, 4, 3, 2, 1, 0], "perfectly_divisible"),
    (11, 3, list(range(9)), [8, 7, 6, 5, 4, 3, 2, 1, 0], "drop_remainder"),
    (5, 5, list(range(5)), [4, 3, 2, 1, 0], "full_batch"),
]

@pytest.mark.parametrize(
    "dataset_range, batch_size, expected_list, shuffled_output, case_name",
    TEST_PARAMS
)
@patch('sample_code.tensorflow.global_shuffle_op')
@patch('sample_code.tensorflow.dataset_ops')
def test_testBatch_execution_and_assertions(
    mock_dataset_ops: MagicMock,
    mock_global_shuffle_op: MagicMock,
    mock_batchglobalshuffletest_instance: MagicMock,
    dataset_range: int,
    batch_size: int,
    expected_list: list,
    shuffled_output: list,
    case_name: str
):
    """
    Tests the testBatch method, verifying the dataset chain construction,
    the calculation of the expected output, and the final assertions
    (assertCountEqual and assertNotEqual) using parameterized inputs.
    """
    # --- Setup Mocks ---

    # 1. Mock Dataset Chain stages
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_batched = MagicMock(name="DatasetBatched")
    mock_ds_prefetched = MagicMock(name="DatasetPrefetched")
    mock_ds_shuffled = MagicMock(name="DatasetShuffled")
    mock_ds_final = MagicMock(name="DatasetFinal")

    mock_dataset_ops.Dataset.range.return_value = mock_ds_start
    mock_ds_start.batch.return_value = mock_ds_batched
    mock_ds_batched.prefetch.return_value = mock_ds_prefetched
    mock_global_shuffle_op._global_shuffle.return_value = mock_ds_shuffled
    mock_ds_shuffled.unbatch.return_value = mock_ds_final

    # Mock AUTOTUNE constant
    mock_dataset_ops.AUTOTUNE = MagicMock(name="AUTOTUNE")

    # 2. Mock Output Retrieval and Assertions on the instance
    mock_getDatasetOutput = mock_batchglobalshuffletest_instance.getDatasetOutput
    mock_getDatasetOutput.return_value = shuffled_output

    mock_assertCountEqual = mock_batchglobalshuffletest_instance.assertCountEqual
    mock_assertNotEqual = mock_batchglobalshuffletest_instance.assertNotEqual

    # --- Execution ---
    mock_batchglobalshuffletest_instance.testBatch(dataset_range, batch_size)

    # --- Assertions ---

    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(dataset_range)

    # Verify batch call with drop_remainder=True
    mock_ds_start.batch.assert_called_once_with(batch_size, drop_remainder=True)

    # Verify prefetch call with AUTOTUNE
    mock_ds_batched.prefetch.assert_called_once_with(buffer_size=mock_dataset_ops.AUTOTUNE)

    # Verify global shuffle call
    mock_global_shuffle_op._global_shuffle.assert_called_once_with(mock_ds_prefetched)

    # Verify unbatch call
    mock_ds_shuffled.unbatch.assert_called_once()

    # 2. Verify Output Retrieval
    mock_getDatasetOutput.assert_called_once_with(
        mock_ds_final, requires_initialization=True
    )

    # 3. Verify Assertions
    # Calculate expected list based on method logic (to ensure test integrity)
    expected_calculated = list(range(0, (dataset_range // batch_size) * batch_size))
    
    # Verify assertCountEqual (checks same elements, regardless of order)
    mock_assertCountEqual.assert_called_once_with(shuffled_output, expected_calculated)
    
    # Verify assertNotEqual (checks that the order is different, simulating shuffle)
    mock_assertNotEqual.assert_called_once_with(shuffled_output, expected_calculated)
# Code added at 20251022-152738
import pytest
from unittest.mock import MagicMock, patch
from typing import Optional
from sample_code.tensorflow import BatchGlobalShuffleTest

# Note: The fixture `mock_batchglobalshuffletest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchGlobalShuffleTest).

# Define parameter combinations to cover both branches (reshuffle=True and reshuffle=False)
TEST_CASES = [
    # dataset_range, batch_size, reshuffle, seed, case_name, output_iteration_0, output_iteration_1, assert_method
    (10, 3, True, 42, "reshuffle_true_assert_not_equal", [8, 7, 6, 5, 4, 3, 2, 1, 0], [1, 0, 2, 3, 4, 5, 6, 7, 8], "assertNotEqual"),
    (10, 3, False, None, "reshuffle_false_assert_equal", [8, 7, 6, 5, 4, 3, 2, 1, 0], [8, 7, 6, 5, 4, 3, 2, 1, 0], "assertEqual"),
]

@pytest.mark.parametrize(
    "dataset_range, batch_size, reshuffle, seed, case_name, output_0, output_1, expected_assertion",
    TEST_CASES
)
@patch('sample_code.tensorflow.global_shuffle_op')
@patch('sample_code.tensorflow.dataset_ops')
def test_testReshuffleRepeatEpochs_coverage(
    mock_dataset_ops: MagicMock,
    mock_global_shuffle_op: MagicMock,
    mock_batchglobalshuffletest_instance: MagicMock,
    dataset_range: int,
    batch_size: int,
    reshuffle: bool,
    seed: Optional[int],
    case_name: str,
    output_0: list,
    output_1: list,
    expected_assertion: str
):
    """
    Tests the testReshuffleRepeatEpochs method, verifying the dataset chain
    construction and covering both branches of the reshuffle logic (assertNotEqual
    when reshuffle=True, assertEqual when reshuffle=False).
    """
    # --- Setup Dataset Chain Mocks ---

    # Define mocks for the chained dataset operations
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_batched = MagicMock(name="DatasetBatched")
    mock_ds_prefetched = MagicMock(name="DatasetPrefetched")
    mock_ds_shuffled = MagicMock(name="DatasetShuffled")
    mock_ds_repeated = MagicMock(name="DatasetRepeated")
    mock_ds_final = MagicMock(name="DatasetFinal")

    mock_dataset_ops.Dataset.range.return_value = mock_ds_start
    mock_ds_start.batch.return_value = mock_ds_batched
    mock_ds_batched.prefetch.return_value = mock_ds_prefetched
    mock_global_shuffle_op._global_shuffle.return_value = mock_ds_shuffled
    mock_ds_shuffled.repeat.return_value = mock_ds_repeated
    mock_ds_repeated.unbatch.return_value = mock_ds_final

    # Mock AUTOTUNE constant
    mock_dataset_ops.AUTOTUNE = MagicMock(name="AUTOTUNE")

    # --- Setup Output and Assertions ---

    # Combine the two iteration outputs for the total output list
    mock_total_output = output_0 + output_1

    mock_getDatasetOutput = mock_batchglobalshuffletest_instance.getDatasetOutput
    mock_getDatasetOutput.return_value = mock_total_output

    mock_assertCountEqual = mock_batchglobalshuffletest_instance.assertCountEqual
    mock_assertNotEqual = mock_batchglobalshuffletest_instance.assertNotEqual
    mock_assertEqual = mock_batchglobalshuffletest_instance.assertEqual

    # --- Execution ---
    mock_batchglobalshuffletest_instance.testReshuffleRepeatEpochs(
        dataset_range, batch_size, reshuffle, seed
    )

    # --- Assertions (Chain Verification) ---

    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(dataset_range)
    mock_ds_start.batch.assert_called_once_with(batch_size, drop_remainder=True)
    mock_ds_batched.prefetch.assert_called_once_with(buffer_size=mock_dataset_ops.AUTOTUNE)

    # 2. Verify global shuffle call parameters
    mock_global_shuffle_op._global_shuffle.assert_called_once_with(
        mock_ds_prefetched, seed=seed, reshuffle_each_iteration=reshuffle
    )

    # 3. Verify repeat and unbatch
    mock_ds_shuffled.repeat.assert_called_once_with(2)
    mock_ds_repeated.unbatch.assert_called_once()

    # 4. Verify Output Retrieval
    mock_getDatasetOutput.assert_called_once_with(
        mock_ds_final, requires_initialization=True
    )

    # --- Assertions (Logic Verification) ---

    # Calculate expected list based on method logic
    expected_len_per_iteration = (dataset_range // batch_size) * batch_size
    expected_list = list(range(expected_len_per_iteration)) * 2

    # 5. Verify assertCountEqual (always called)
    mock_assertCountEqual.assert_called_once_with(mock_total_output, expected_list)

    # 6. Verify conditional assertion based on 'reshuffle'
    if expected_assertion == "assertNotEqual":
        mock_assertNotEqual.assert_called_once_with(output_0, output_1)
        mock_assertEqual.assert_not_called()
    else:
        mock_assertEqual.assert_called_once_with(output_0, output_1)
        mock_assertNotEqual.assert_not_called()
# Code added at 20251022-152755
import pytest
from unittest.mock import MagicMock, patch
from sample_code.tensorflow import BatchGlobalShuffleTest

# Define a mock exception class to simulate the TensorFlow error
class MockFailedPreconditionError(Exception):
    """Mock class for errors.FailedPreconditionError."""
    pass

@patch('sample_code.tensorflow.global_shuffle_op')
@patch('sample_code.tensorflow.errors')
@patch('sample_code.tensorflow.dataset_ops')
def test_testNoDropRemainder_raises_expected_error(
    mock_dataset_ops: MagicMock,
    mock_errors: MagicMock,
    mock_global_shuffle_op: MagicMock,
    mock_batchglobalshuffletest_instance: MagicMock
):
    """
    Tests the testNoDropRemainder method ensures that attempting global shuffling
    on a dataset batched with drop_remainder=False correctly triggers a
    FailedPreconditionError, verifying the dataset chain and assertion setup.
    """
    # --- Setup Constants and Mocks ---

    DATASET_RANGE = 10
    BATCH_SIZE = 3

    # 1. Mock the expected exception type
    mock_errors.FailedPreconditionError = MockFailedPreconditionError

    # 2. Mock Dataset Chain stages
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_batched = MagicMock(name="DatasetBatched")
    mock_ds_prefetched = MagicMock(name="DatasetPrefetched")
    mock_ds_shuffled = MagicMock(name="DatasetShuffled")

    mock_dataset_ops.Dataset.range.return_value = mock_ds_start
    mock_ds_start.batch.return_value = mock_ds_batched
    mock_ds_batched.prefetch.return_value = mock_ds_prefetched
    mock_global_shuffle_op._global_shuffle.return_value = mock_ds_shuffled

    # Mock AUTOTUNE constant
    mock_autotune = MagicMock(name="AUTOTUNE")
    mock_dataset_ops.AUTOTUNE = mock_autotune

    # 3. Mock Assertion and Output Retrieval
    mock_assertRaisesRegex = mock_batchglobalshuffletest_instance.assertRaisesRegex
    mock_getDatasetOutput = mock_batchglobalshuffletest_instance.getDatasetOutput

    # Define the expected regex pattern
    expected_error_message = (
        "does not support global shuffling with `drop_remainder=False`."
    )
    
    # Simulate the error being raised when output is retrieved inside the context manager
    mock_getDatasetOutput.side_effect = MockFailedPreconditionError(expected_error_message)

    # --- Execution ---

    # The test should pass because the error is caught by the mocked assertRaisesRegex context
    mock_batchglobalshuffletest_instance.testNoDropRemainder(DATASET_RANGE, BATCH_SIZE)

    # --- Assertions ---

    # 1. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(DATASET_RANGE)

    # Verify batch call with drop_remainder=False
    mock_ds_start.batch.assert_called_once_with(BATCH_SIZE, drop_remainder=False)

    # Verify prefetch call
    mock_ds_batched.prefetch.assert_called_once_with(buffer_size=mock_autotune)

    # 2. Verify global shuffle call
    mock_global_shuffle_op._global_shuffle.assert_called_once_with(mock_ds_prefetched)

    # 3. Verify Assertion Context Setup
    mock_assertRaisesRegex.assert_called_once()

    call_args = mock_assertRaisesRegex.call_args[0]

    # Arg 0: Expected Exception Type
    assert call_args[0] is MockFailedPreconditionError

    # Arg 1: Expected Regex Pattern
    assert call_args[1] == expected_error_message

    # 4. Verify Output Retrieval (The call that triggered the error)
    mock_getDatasetOutput.assert_called_once_with(
        mock_ds_shuffled, requires_initialization=True
    )
# Code added at 20251022-152808
import pytest
from unittest.mock import MagicMock, patch
from typing import Callable
from sample_code.tensorflow import BatchGlobalShuffleCheckpointTest

# Note: The fixture `mock_BatchGlobalShuffleCheckpointTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchGlobalShuffleCheckpointTest).

# Define parameter combinations to cover different batching scenarios and symbolic checkpoint states
TEST_CASES = [
    # dataset_range, batch_size, symbolic_checkpoint, expected_num_outputs, case_name
    (10, 3, True, 9, "symbolic_true_partial_batch"),  # 10 // 3 = 3 batches * 3 size = 9 outputs
    (8, 4, False, 8, "symbolic_false_full_batch"),    # 8 // 4 = 2 batches * 4 size = 8 outputs
]

@pytest.mark.parametrize(
    "dataset_range, batch_size, symbolic_checkpoint, expected_num_outputs, case_name",
    TEST_CASES
)
@patch('sample_code.tensorflow.global_shuffle_op')
@patch('sample_code.tensorflow.options_lib')
@patch('sample_code.tensorflow.dataset_ops')
def test_testBatch_full_coverage(
    mock_dataset_ops: MagicMock,
    mock_options_lib: MagicMock,
    mock_global_shuffle_op: MagicMock,
    mock_BatchGlobalShuffleCheckpointTest_instance: MagicMock,
    dataset_range: int,
    batch_size: int,
    symbolic_checkpoint: bool,
    expected_num_outputs: int,
    case_name: str
):
    """
    Tests the testBatch method, ensuring the internal dataset builder closure is
    correctly constructed with all chained operations and options, and that the
    external verify_fn is called with the correct parameters and calculated num_outputs.
    """
    # --- Setup Mocks for Dataset Chain ---

    # Define mocks for each stage of the dataset chain
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_batched = MagicMock(name="DatasetBatched")
    mock_ds_prefetched = MagicMock(name="DatasetPrefetched")
    mock_ds_shuffled = MagicMock(name="DatasetShuffled")
    mock_ds_unbatched = MagicMock(name="DatasetUnbatched")
    mock_ds_final = MagicMock(name="DatasetFinal")

    # Mock the chain links
    mock_dataset_ops.Dataset.range.return_value = mock_ds_start
    mock_ds_start.batch.return_value = mock_ds_batched
    mock_ds_batched.prefetch.return_value = mock_ds_prefetched
    mock_global_shuffle_op._global_shuffle.return_value = mock_ds_shuffled
    mock_ds_shuffled.unbatch.return_value = mock_ds_unbatched
    mock_ds_unbatched.with_options.return_value = mock_ds_final

    # Mock AUTOTUNE constant
    mock_autotune = MagicMock(name="AUTOTUNE")
    mock_dataset_ops.AUTOTUNE = mock_autotune

    # Mock Options setup
    mock_options = MagicMock(name="OptionsInstance")
    mock_options_lib.Options.return_value = mock_options

    # Mock the external verification function
    mock_verify_fn: MagicMock = MagicMock(name="verify_fn")

    # --- Execution 1: Call testBatch and verify verify_fn setup ---
    mock_BatchGlobalShuffleCheckpointTest_instance.testBatch(
        mock_verify_fn, dataset_range, batch_size, symbolic_checkpoint
    )

    # 1. Verify verify_fn call arguments
    mock_verify_fn.assert_called_once()
    call_args, call_kwargs = mock_verify_fn.call_args

    # Arg 0: self
    assert call_args[0] is mock_BatchGlobalShuffleCheckpointTest_instance
    # Arg 1: _build_dataset closure (must be callable)
    dataset_fn: Callable = call_args[1]
    assert callable(dataset_fn)
    # Arg 2: num_outputs calculation
    assert call_kwargs['num_outputs'] == expected_num_outputs
    # Arg 3: assert_items_equal flag
    assert call_kwargs['assert_items_equal'] is True

    # --- Execution 2: Execute the closure and verify internal chain ---
    
    # Reset mocks that track calls inside the closure
    mock_dataset_ops.Dataset.range.reset_mock()
    mock_options_lib.Options.reset_mock()

    result_dataset = dataset_fn()

    # 2. Verify Options creation and configuration
    mock_options_lib.Options.assert_called_once()
    assert mock_options.experimental_symbolic_checkpoint == symbolic_checkpoint

    # 3. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(dataset_range)
    mock_ds_start.batch.assert_called_once_with(batch_size, drop_remainder=True)
    mock_ds_batched.prefetch.assert_called_once_with(buffer_size=mock_autotune)
    mock_global_shuffle_op._global_shuffle.assert_called_once_with(mock_ds_prefetched, seed=42)
    mock_ds_shuffled.unbatch.assert_called_once()
    mock_ds_unbatched.with_options.assert_called_once_with(mock_options)

    # 4. Verify the final returned object
    assert result_dataset is mock_ds_final
# Code added at 20251022-152820
import pytest
from unittest.mock import MagicMock, patch
from typing import Callable
from sample_code.tensorflow import BatchGlobalShuffleCheckpointTest

# Note: The fixture `mock_BatchGlobalShuffleCheckpointTest_instance` is assumed to be available
# from the preceding context, providing a MagicMock(spec=BatchGlobalShuffleCheckpointTest).

# Define parameter combinations to cover both branches (reshuffle=True and reshuffle=False)
TEST_CASES = [
    # dataset_range, batch_size, reshuffle_each_iteration, symbolic_checkpoint, expected_num_outputs, assert_items_equal, case_name
    (10, 3, True, True, 9, True, "reshuffle_true_symbolic_true"),
    (8, 4, False, False, 8, False, "reshuffle_false_symbolic_false"),
]

@pytest.mark.parametrize(
    "dataset_range, batch_size, reshuffle_each_iteration, symbolic_checkpoint, expected_num_outputs, assert_items_equal, case_name",
    TEST_CASES
)
@patch('sample_code.tensorflow.global_shuffle_op')
@patch('sample_code.tensorflow.options_lib')
@patch('sample_code.tensorflow.dataset_ops')
def test_testReshuffleEachIteration_coverage(
    mock_dataset_ops: MagicMock,
    mock_options_lib: MagicMock,
    mock_global_shuffle_op: MagicMock,
    mock_BatchGlobalShuffleCheckpointTest_instance: MagicMock,
    dataset_range: int,
    batch_size: int,
    reshuffle_each_iteration: bool,
    symbolic_checkpoint: bool,
    expected_num_outputs: int,
    assert_items_equal: bool,
    case_name: str
):
    """
    Tests the testReshuffleEachIteration method, ensuring the internal dataset 
    builder closure is correctly constructed with all chained operations and options, 
    and that the external verify_fn is called with the correct parameters, 
    covering both branches of `reshuffle_each_iteration`.
    """
    # --- Setup Mocks for Dataset Chain ---

    # Define mocks for each stage of the dataset chain
    mock_ds_start = MagicMock(name="DatasetStart")
    mock_ds_batched = MagicMock(name="DatasetBatched")
    mock_ds_prefetched = MagicMock(name="DatasetPrefetched")
    mock_ds_shuffled = MagicMock(name="DatasetShuffled")
    mock_ds_unbatched = MagicMock(name="DatasetUnbatched")
    mock_ds_final = MagicMock(name="DatasetFinal")

    # Mock the chain links
    mock_dataset_ops.Dataset.range.return_value = mock_ds_start
    mock_ds_start.batch.return_value = mock_ds_batched
    mock_ds_batched.prefetch.return_value = mock_ds_prefetched
    mock_global_shuffle_op._global_shuffle.return_value = mock_ds_shuffled
    mock_ds_shuffled.unbatch.return_value = mock_ds_unbatched
    mock_ds_unbatched.with_options.return_value = mock_ds_final

    # Mock AUTOTUNE constant
    mock_autotune = MagicMock(name="AUTOTUNE")
    mock_dataset_ops.AUTOTUNE = mock_autotune

    # Mock Options setup
    mock_options = MagicMock(name="OptionsInstance")
    mock_options_lib.Options.return_value = mock_options

    # Mock the external verification function
    mock_verify_fn: MagicMock = MagicMock(name="verify_fn")

    # --- Execution 1: Call testReshuffleEachIteration and verify verify_fn setup ---
    mock_BatchGlobalShuffleCheckpointTest_instance.testReshuffleEachIteration(
        mock_verify_fn, dataset_range, batch_size, reshuffle_each_iteration, symbolic_checkpoint
    )

    # 1. Verify verify_fn call arguments
    mock_verify_fn.assert_called_once()
    call_args, call_kwargs = mock_verify_fn.call_args

    # Arg 1: _build_dataset closure (must be callable)
    dataset_fn: Callable = call_args[1]
    assert callable(dataset_fn)
    
    # Arg 2: num_outputs calculation
    assert call_kwargs['num_outputs'] == expected_num_outputs
    
    # Arg 3: assert_items_equal flag (controlled by reshuffle_each_iteration)
    assert call_kwargs['assert_items_equal'] == assert_items_equal

    # --- Execution 2: Execute the closure and verify internal chain ---
    
    # Reset mocks that track calls inside the closure
    mock_dataset_ops.Dataset.range.reset_mock()
    mock_options_lib.Options.reset_mock()

    result_dataset = dataset_fn()

    # 2. Verify Options creation and configuration
    mock_options_lib.Options.assert_called_once()
    assert mock_options.experimental_symbolic_checkpoint == symbolic_checkpoint

    # 3. Verify Dataset Chain construction
    mock_dataset_ops.Dataset.range.assert_called_once_with(dataset_range)
    mock_ds_start.batch.assert_called_once_with(batch_size, drop_remainder=True)
    mock_ds_batched.prefetch.assert_called_once_with(buffer_size=mock_autotune)
    
    # Verify global shuffle call parameters
    mock_global_shuffle_op._global_shuffle.assert_called_once_with(
        mock_ds_prefetched, seed=42, reshuffle_each_iteration=reshuffle_each_iteration
    )
    
    mock_ds_shuffled.unbatch.assert_called_once()
    mock_ds_unbatched.with_options.assert_called_once_with(mock_options)

    # 4. Verify the final returned object
    assert result_dataset is mock_ds_final
