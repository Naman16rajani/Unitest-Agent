# Generated tests for class: BatchCheckpointTest
# Source file: sample_code/tensorflow.py
# Generated on: 2025-07-21 23:08:37

# Tests for method: _build_dataset
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
from types import ModuleType

# Create a dummy `tensorflow` module structure to allow patching
# without actually installing tensorflow. This simulates `import tensorflow.data`.
# This needs to be done before any potential `tensorflow` imports by the method itself.
mock_tf = ModuleType('tensorflow')
sys.modules['tensorflow'] = mock_tf
mock_tf_data = ModuleType('tensorflow.data')
mock_tf.data = mock_tf_data
sys.modules['tensorflow.data'] = mock_tf_data

# The class containing the _build_dataset method
# We assume `tensorflow.data` is imported as `dataset_ops` or used directly
# as `tensorflow.data` in the original module where this class resides.
# For testing purposes, we will directly use `tensorflow.data` as the target for patching.
class ClassUnderTest:
    def _build_dataset(
        self,
        multiplier=15.0,
        tensor_slice_len=2,
        batch_size=2,
        num_parallel_calls=None,
        options=None,
    ):
        components = (
            np.arange(tensor_slice_len),
            np.array([[1, 2, 3]]) * np.arange(tensor_slice_len)[:, np.newaxis],
            np.array(multiplier) * np.arange(tensor_slice_len),
        )

        # In a real scenario, this would likely be `import tensorflow.data as dataset_ops`
        # at the module level, then `dataset_ops.Dataset`.
        # For the purpose of this test, we directly refer to `tensorflow.data.Dataset`
        # which will be the mocked object due to the `patch` decorator.
        import tensorflow.data # This import will now resolve to our mock module
        dataset = tensorflow.data.Dataset.from_tensor_slices(components)
        dataset = dataset.batch(batch_size, num_parallel_calls=num_parallel_calls)
        if options:
            dataset = dataset.with_options(options)
        return dataset

@pytest.fixture
def mock_tf_data_dataset():
    """
    Fixture to mock tensorflow.data.Dataset and its chained methods.
    """
    with patch('tensorflow.data.Dataset') as mock_dataset_class:
        # Mock the instance returned by from_tensor_slices
        mock_dataset_instance = MagicMock()
        # Ensure batch and with_options return the mock instance for chaining
        mock_dataset_instance.batch.return_value = mock_dataset_instance
        mock_dataset_instance.with_options.return_value = mock_dataset_instance

        # Make from_tensor_slices a class method that returns the mock instance
        mock_dataset_class.from_tensor_slices.return_value = mock_dataset_instance
        yield mock_dataset_class, mock_dataset_instance

@pytest.fixture
def instance_under_test():
    """
    Fixture to provide an instance of the class under test.
    """
    return ClassUnderTest()

def test__build_dataset_default_parameters(instance_under_test, mock_tf_data_dataset):
    """
    Test _build_dataset with default parameters.
    Verifies correct calls to from_tensor_slices, batch, and no call to with_options.
    """
    mock_dataset_class, mock_dataset_instance = mock_tf_data_dataset

    expected_components = (
        np.array([0, 1]),
        np.array([[0, 0, 0], [1, 2, 3]]),
        np.array([0.0, 15.0])
    )

    result = instance_under_test._build_dataset()

    # Verify from_tensor_slices was called with correct components
    mock_dataset_class.from_tensor_slices.assert_called_once()
    actual_components = mock_dataset_class.from_tensor_slices.call_args[0][0]
    for i in range(len(expected_components)):
        np.testing.assert_array_equal(actual_components[i], expected_components[i])

    # Verify batch was called with default batch_size and num_parallel_calls
    mock_dataset_instance.batch.assert_called_once_with(2, num_parallel_calls=None)

    # Verify with_options was not called
    mock_dataset_instance.with_options.assert_not_called()

    # Verify the final dataset object is returned
    assert result is mock_dataset_instance

def test__build_dataset_custom_parameters_no_options(instance_under_test, mock_tf_data_dataset):
    """
    Test _build_dataset with custom parameters but no options.
    Verifies correct calls to from_tensor_slices and batch, and no call to with_options.
    """
    mock_dataset_class, mock_dataset_instance = mock_tf_data_dataset

    multiplier = 10.0
    tensor_slice_len = 5
    batch_size = 3
    num_parallel_calls = 4

    expected_components = (
        np.arange(tensor_slice_len),
        np.array([[1, 2, 3]]) * np.arange(tensor_slice_len)[:, np.newaxis],
        np.array(multiplier) * np.arange(tensor_slice_len),
    )

    result = instance_under_test._build_dataset(
        multiplier=multiplier,
        tensor_slice_len=tensor_slice_len,
        batch_size=batch_size,
        num_parallel_calls=num_parallel_calls,
        options=None
    )

    # Verify from_tensor_slices was called with correct components
    mock_dataset_class.from_tensor_slices.assert_called_once()
    actual_components = mock_dataset_class.from_tensor_slices.call_args[0][0]
    for i in range(len(expected_components)):
        np.testing.assert_array_equal(actual_components[i], expected_components[i])

    # Verify batch was called with custom parameters
    mock_dataset_instance.batch.assert_called_once_with(batch_size, num_parallel_calls=num_parallel_calls)

    # Verify with_options was not called
    mock_dataset_instance.with_options.assert_not_called()

    assert result is mock_dataset_instance

def test__build_dataset_with_options(instance_under_test, mock_tf_data_dataset):
    """
    Test _build_dataset when options are provided.
    Verifies that with_options is called with the provided options.
    """
    mock_dataset_class, mock_dataset_instance = mock_tf_data_dataset

    options = MagicMock()
    result = instance_under_test._build_dataset(options=options)

    # Verify from_tensor_slices and batch were called (default parameters)
    mock_dataset_class.from_tensor_slices.assert_called_once()
    mock_dataset_instance.batch.assert_called_once()

    # Verify with_options was called with the provided options
    mock_dataset_instance.with_options.assert_called_once_with(options)

    assert result is mock_dataset_instance

def test__build_dataset_zero_tensor_slice_len(instance_under_test, mock_tf_data_dataset):
    """
    Test _build_dataset with tensor_slice_len = 0 (edge case).
    """
    mock_dataset_class, mock_dataset_instance = mock_tf_data_dataset

    tensor_slice_len = 0
    multiplier = 10.0
    batch_size = 1

    expected_components = (
        np.arange(tensor_slice_len), # []
        np.array([[1, 2, 3]]) * np.arange(tensor_slice_len)[:, np.newaxis], # []
        np.array(multiplier) * np.arange(tensor_slice_len), # []
    )

    result = instance_under_test._build_dataset(
        tensor_slice_len=tensor_slice_len,
        multiplier=multiplier,
        batch_size=batch_size
    )

    mock_dataset_class.from_tensor_slices.assert_called_once()
    actual_components = mock_dataset_class.from_tensor_slices.call_args[0][0]
    for i in range(len(expected_components)):
        np.testing.assert_array_equal(actual_components[i], expected_components[i])

    mock_dataset_instance.batch.assert_called_once_with(batch_size, num_parallel_calls=None)
    mock_dataset_instance.with_options.assert_not_called()
    assert result is mock_dataset_instance

def test__build_dataset_large_tensor_slice_len(instance_under_test, mock_tf_data_dataset):
    """
    Test _build_dataset with a larger tensor_slice_len.
    """
    mock_dataset_class, mock_dataset_instance = mock_tf_data_dataset

    tensor_slice_len = 100
    batch_size = 10
    num_parallel_calls = 8

    expected_components = (
        np.arange(tensor_slice_len),
        np.array([[1, 2, 3]]) * np.arange(tensor_slice_len)[:, np.newaxis],
        np.array(15.0) * np.arange(tensor_slice_len),
    )

    result = instance_under_test._build_dataset(
        tensor_slice_len=tensor_slice_len,
        batch_size=batch_size,
        num_parallel_calls=num_parallel_calls
    )

    mock_dataset_class.from_tensor_slices.assert_called_once()
    actual_components = mock_dataset_class.from_tensor_slices.call_args[0][0]
    for i in range(len(expected_components)):
        np.testing.assert_array_equal(actual_components[i], expected_components[i])

    mock_dataset_instance.batch.assert_called_once_with(batch_size, num_parallel_calls=num_parallel_calls)
    assert result is mock_dataset_instance

def test__build_dataset_batch_size_one(instance_under_test, mock_tf_data_dataset):
    """
    Test _build_dataset with batch_size = 1.
    """
    mock_dataset_class, mock_dataset_instance = mock_tf_data_dataset

    batch_size = 1
    result = instance_under_test._build_dataset(batch_size=batch_size)

    mock_dataset_instance.batch.assert_called_once_with(batch_size, num_parallel_calls=None)
    assert result is mock_dataset_instance

def test__build_dataset_num_parallel_calls_explicit_none(instance_under_test, mock_tf_data_dataset):
    """
    Test _build_dataset with num_parallel_calls explicitly set to None.
    """
    mock_dataset_class, mock_dataset_instance = mock_tf_data_dataset

    num_parallel_calls = None
    result = instance_under_test._build_dataset(num_parallel_calls=num_parallel_calls)

    mock_dataset_instance.batch.assert_called_once_with(2, num_parallel_calls=None)
    assert result is mock_dataset_instance

def test__build_dataset_num_parallel_calls_zero(instance_under_test, mock_tf_data_dataset):
    """
    Test _build_dataset with num_parallel_calls set to 0.
    """
    mock_dataset_class, mock_dataset_instance = mock_tf_data_dataset

    num_parallel_calls = 0
    result = instance_under_test._build_dataset(num_parallel_calls=num_parallel_calls)

    mock_dataset_instance.batch.assert_called_once_with(2, num_parallel_calls=0)
    assert result is mock_dataset_instance

def test__build_dataset_multiplier_zero(instance_under_test, mock_tf_data_dataset):
    """
    Test _build_dataset with multiplier = 0.
    """
    mock_dataset_class, mock_dataset_instance = mock_tf_data_dataset

    multiplier = 0.0
    tensor_slice_len = 3
    expected_components = (
        np.arange(tensor_slice_len),
        np.array([[1, 2, 3]]) * np.arange(tensor_slice_len)[:, np.newaxis],
        np.array(multiplier) * np.arange(tensor_slice_len), # [0.0, 0.0, 0.0]
    )

    result = instance_under_test._build_dataset(multiplier=multiplier, tensor_slice_len=tensor_slice_len)

    mock_dataset_class.from_tensor_slices.assert_called_once()
    actual_components = mock_dataset_class.from_tensor_slices.call_args[0][0]
    for i in range(len(expected_components)):
        np.testing.assert_array_equal(actual_components[i], expected_components[i])

    assert result is mock_dataset_instance

# Tests for method: test
import pytest
from unittest.mock import MagicMock, patch
import sys
from types import ModuleType

# --- Setup for mocking options_lib ---
# Create a dummy module for `options_lib` in sys.modules.
# This simulates `import options_lib` and allows patching `options_lib.Options`.
# In a real TensorFlow environment, you would import the actual module
# (e.g., `from tensorflow.python.data.experimental.ops import options as options_lib`)
# and patch it using its full import path.
mock_options_lib_module = ModuleType('options_lib')
sys.modules['options_lib'] = mock_options_lib_module

# Add a dummy Options class to the mock module. This class will be replaced by MagicMock.
class DummyOptions:
    pass
mock_options_lib_module.Options = DummyOptions
# --- End setup for mocking options_lib ---

# Define the class containing the 'test' method.
# In a real scenario, you would import the actual class that contains this method.
class MyClassUnderTest:
    def _build_dataset(self, *args, **kwargs):
        # This method will be mocked by the test
        pass

    def test(self, verify_fn, symbolic_checkpoint, num_parallel_calls):
        tensor_slice_len = 8
        batch_size = 2
        # This call will now correctly refer to the patched options_lib.Options
        options = sys.modules['options_lib'].Options()
        options.experimental_symbolic_checkpoint = symbolic_checkpoint
        num_outputs = tensor_slice_len // batch_size
        verify_fn(
            self,
            lambda: self._build_dataset(
                15.0, tensor_slice_len, batch_size, num_parallel_calls, options
            ),
            num_outputs,
        )

class TestMyTestClassMethod:

    @pytest.fixture
    def mock_self(self):
        """Fixture for the 'self' argument of the test method."""
        mock_instance = MagicMock(spec=MyClassUnderTest)
        # Ensure _build_dataset is a mock callable attribute
        mock_instance._build_dataset = MagicMock()
        return mock_instance

    @pytest.fixture
    def mock_verify_fn(self):
        """Fixture for the 'verify_fn' argument."""
        return MagicMock()

    @patch('options_lib.Options') # Patch the Options class within the mocked options_lib module
    def test_test_method_symbolic_checkpoint_true(
        self, mock_options_class, mock_self, mock_verify_fn
    ):
        # Arrange
        symbolic_checkpoint = True
        num_parallel_calls = 5
        expected_tensor_slice_len = 8
        expected_batch_size = 2
        expected_num_outputs = expected_tensor_slice_len // expected_batch_size  # 4

        mock_options_instance = mock_options_class.return_value
        mock_build_dataset_return_value = MagicMock()
        mock_self._build_dataset.return_value = mock_build_dataset_return_value

        # Act
        MyClassUnderTest.test(
            mock_self,
            mock_verify_fn,
            symbolic_checkpoint,
            num_parallel_calls,
        )

        # Assert
        mock_options_class.assert_called_once_with()
        assert mock_options_instance.experimental_symbolic_checkpoint is symbolic_checkpoint

        mock_verify_fn.assert_called_once()
        assert mock_verify_fn.call_args[0][0] is mock_self

        lambda_arg = mock_verify_fn.call_args[0][1]
        assert callable(lambda_arg)
        
        # Call the lambda to trigger _build_dataset
        returned_from_lambda = lambda_arg()
        mock_self._build_dataset.assert_called_once_with(
            15.0,
            expected_tensor_slice_len,
            expected_batch_size,
            num_parallel_calls,
            mock_options_instance,
        )
        assert returned_from_lambda is mock_build_dataset_return_value
        assert mock_verify_fn.call_args[0][2] == expected_num_outputs

    @patch('options_lib.Options')
    def test_test_method_symbolic_checkpoint_false_none_parallel_calls(
        self, mock_options_class, mock_self, mock_verify_fn
    ):
        # Arrange
        symbolic_checkpoint = False
        num_parallel_calls = None
        expected_tensor_slice_len = 8
        expected_batch_size = 2
        expected_num_outputs = expected_tensor_slice_len // expected_batch_size  # 4

        mock_options_instance = mock_options_class.return_value
        mock_build_dataset_return_value = MagicMock()

# Tests for method: _sparse
import pytest
import unittest.mock
from unittest.mock import MagicMock, patch

# Define the class containing the _sparse method for testing purposes.
# In a real project, you would import this class from its module.
class MyClassUnderTest:
    def _sparse(self, i):
        # This import statement within the method is crucial for the patching to work correctly.
        # unittest.mock.patch needs to know the exact path where the object is looked up.
        from tensorflow import sparse_tensor
        return sparse_tensor.SparseTensorValue(
            indices=[[0]], values=(i * [1]), dense_shape=[1]
        )

class TestSparseMethod:

    @pytest.fixture
    def mock_sparse_tensor_value(self):
        """
        Fixture to mock the SparseTensorValue constructor.
        The patch target 'tensorflow.sparse_tensor.SparseTensorValue' assumes
        that `from tensorflow import sparse_tensor` is used, and then `sparse_tensor.SparseTensorValue` is accessed.
        Since MyClassUnderTest is defined in this test file, this path correctly targets the dependency.
        """
        with patch('tensorflow.sparse_tensor.SparseTensorValue') as mock_constructor:
            yield mock_constructor

    def test__sparse_positive_integer_input(self, mock_sparse_tensor_value):
        """
        Test _sparse with a positive integer input for 'i'.
        Verifies that SparseTensorValue is called with the correct arguments,
        especially `values` as a list of ones repeated 'i' times.
        """
        instance = MyClassUnderTest()
        i = 5
        expected_values = [1, 1, 1, 1, 1]  # 5 * [1]

        result = instance._sparse(i)

        mock_sparse_tensor_value.assert_called_once_with(
            indices=[[0]],
            values=expected_values,
            dense_shape=[1]
        )
        assert result == mock_sparse_tensor_value.return_value

    def test__sparse_zero_integer_input(self, mock_sparse_tensor_value):
        """
        Test _sparse with zero as input for 'i'.
        Verifies that `values` becomes an empty list.
        """
        instance = MyClassUnderTest()
        i = 0
        expected_values = []  # 0 * [1]

        result = instance._sparse(i)

        mock_sparse_tensor_value.assert_called_once_with(
            indices=[[0]],
            values=expected_values,
            dense_shape=[1]
        )
        assert result == mock_sparse_tensor_value.return_value

    def test__sparse_negative_integer_input(self, mock_sparse_tensor_value):
        """
        Test _sparse with a negative integer input for 'i'.
        Verifies that `values` becomes an empty list, as sequence repetition by negative number results in empty list.
        """
        instance = MyClassUnderTest()
        i = -3
        expected_values = []  # -3 * [1]

        result = instance._sparse(i)

        mock_sparse_tensor_value.assert_called_once_with(
            indices=[[0]],
            values=expected_values,
            dense_shape=[1]
        )
        assert result == mock_sparse_tensor_value.return_value

    def test__sparse_large_integer_input(self, mock_sparse_tensor_value):
        """
        Test _sparse with a large integer input for 'i' to check scalability of list creation.
        """
        instance = MyClassUnderTest()
        i = 1000
        expected_values = [1] * 1000

        result = instance._sparse(i)

        mock_sparse_tensor_value.assert_called_once_with(
            indices=[[0]],
            values=expected_values,
            dense_shape=[1]
        )
        assert result == mock_sparse_tensor_value.return_value

    def test__sparse_float_input_raises_type_error(self):
        """
        Test _sparse with a float input for 'i'.
        Verifies that a TypeError is raised because floats cannot be used for sequence repetition.
        """
        instance = MyClassUnderTest()
        i = 2.5

        with pytest.raises(TypeError) as excinfo:
            instance._sparse(i)
        assert "can't multiply sequence by non-int of type 'float'" in str(excinfo.value)

    def test__sparse_list_input_raises_type_error(self):
        """
        Test _sparse with a list input for 'i'.
        Verifies that a TypeError is raised because lists cannot be used for sequence repetition.
        """
        instance = MyClassUnderTest()
        i = [1, 2]

        with pytest.raises(TypeError) as excinfo:
            instance._sparse(i)
        assert "can't multiply sequence by non-int of type 'list'" in str(excinfo.value)

    def test__sparse_string_input_raises_type_error(self):
        """
        Test _sparse with a string input for 'i'.
        Verifies that a TypeError is raised because strings cannot be used for sequence repetition.
        """
        instance = MyClassUnderTest()
        i = "hello"

        with pytest.raises(TypeError) as excinfo:
            instance._sparse(i)
        assert "can't multiply sequence by non-int of type 'str'" in str(excinfo.value)

    def test__sparse_none_input_raises_type_error(self):
        """
        Test _sparse with None as input for 'i'.
        Verifies that a TypeError is raised because None cannot be used for sequence repetition.
        """
        instance = MyClassUnderTest()
        i = None

        with pytest.raises(TypeError) as excinfo:
            instance._sparse(i)
        assert "can't multiply sequence by non-int of type 'NoneType'" in str(excinfo.value)

# Tests for method: _build_dataset_sparse
import pytest
from unittest.mock import MagicMock, patch

# Define a dummy class that contains the method to be tested.
# In a real scenario, you would import the actual class:
# from your_module import YourClass

class MyClass:
    def __init__(self):
        # _sparse is an instance method, so it should be mocked on the instance.
        self._sparse = MagicMock(name='_sparse_method')

    def _build_dataset_sparse(self, batch_size=5):
        # This is the original method code.
        # It assumes `dataset_ops` is available in the current scope,
        # typically from `import tensorflow.data.dataset_ops as dataset_ops`
        # or `from tensorflow.data import dataset_ops`.
        # We will mock `tensorflow.data.dataset_ops.Dataset` directly.
        import tensorflow.data.dataset_ops as dataset_ops # This line is added for the method to run in a test env without actual TF import issues
                                                          # In a real scenario, this import would be at the module level.
                                                          # The patch will intercept `dataset_ops.Dataset`.
        return dataset_ops.Dataset.range(10).map(self._sparse).batch(batch_size)


@patch('tensorflow.data.dataset_ops.Dataset')
class TestBuildDatasetSparse:

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_Dataset):
        # Mock the chained calls: .range().map().batch()
        self.mock_range_obj = MagicMock(name='mock_range_obj')
        self.mock_map_obj = MagicMock(name='mock_map_obj')
        self.mock_batch_obj = MagicMock(name='mock_batch_obj')

        mock_Dataset.range.return_value = self.mock_range_obj
        self.mock_range_obj.map.return_value = self.mock_map_obj
        self.mock_map_obj.batch.return_value = self.mock_batch_obj

        self.instance = MyClass()

    def test_build_dataset_sparse_default_batch_size(self, mock_Dataset):
        # Test with default batch_size (5)
        result = self.instance._build_dataset_sparse()

        # Assertions for chained calls
        mock_Dataset.range.assert_called_once_with(10)
        self.mock_range_obj.map.assert_called_once_with(self.instance._sparse)
        self.mock_map_obj.batch.assert_called_once_with(5) # Default batch_size
        assert result is self.mock_batch_obj

    def test_build_dataset_sparse_custom_batch_size(self, mock_Dataset):
        # Test with a custom batch_size
        custom_batch_size = 32
        result = self.instance._build_dataset_sparse(batch_size=custom_batch_size)

        # Assertions for chained calls
        mock_Dataset.range.assert_called_once_with(10)
        self.mock_range_obj.map.assert_called_once_with(self.instance._sparse)
        self.mock_map_obj.batch.assert_called_once_with(custom_batch_size)
        assert result is self.mock_batch_obj

    def test_build_dataset_sparse_zero_batch_size(self, mock_Dataset):
        # Edge case: batch_size = 0 (though tf.data.Dataset.batch might handle this differently,
        # we test if the value is passed correctly)
        result = self.instance._build_dataset_sparse(batch_size=0)

        mock_Dataset.range.assert_called_once_with(10)
        self.mock_range_obj.map.assert_called_once_with(self.instance._sparse)
        self.mock_map_obj.batch.assert_called_once_with(0)
        assert result is self.mock_batch_obj

    def test_build_dataset_sparse_large_batch_size(self, mock_Dataset):
        # Boundary value: large batch_size
        large_batch_size = 1000
        result = self.instance._build_dataset_sparse(batch_size=large_batch_size)

        mock_Dataset.range.assert_called_once_with(10)
        self.mock_range_obj.map.assert_called_once_with(self.instance._sparse)
        self.mock_map_obj.batch.assert_called_once_with(large_batch_size)
        assert result is self.mock_batch_obj

    def test_build_dataset_sparse_sparse_method_is_called_by_map(self, mock_Dataset):
        # Ensure that self._sparse is passed to map correctly
        self.instance._build_dataset_sparse()

        # The mock for self._sparse itself is not called by `map` directly,
        # but rather passed as a callable. The actual call happens when the dataset is iterated.
        # We assert that it was passed as an argument.
        self.mock_range_obj.map.assert_called_once_with(self.instance._sparse)
        # We don't expect self._sparse to be called during dataset construction.
        self.instance._sparse.assert_not_called()

# Tests for method: testSparse
import pytest
from unittest.mock import MagicMock, patch

# Assume testSparse is a method of a class, e.g., TensorflowTestClass
# In a real scenario, you would import the actual class:
# from tensorflow.python.data.experimental import checkpoint_test_base as test_base
# class TensorflowTestClass(test_base.CheckpointTestBase):
#     # ... other methods and attributes
#     def testSparse(self, verify_fn):
#         verify_fn(self, self._build_dataset_sparse, num_outputs=2)

# For the purpose of this test, we create a minimal dummy class
# that mimics the structure required by testSparse.
class DummyTestClass:
    def testSparse(self, verify_fn):
        verify_fn(self, self._build_dataset_sparse, num_outputs=2)


class TestTestSparseMethod:

    def test_testSparse_calls_verify_fn_with_correct_arguments(self):
        """
        Test that testSparse correctly calls verify_fn with self,
        self._build_dataset_sparse, and num_outputs=2.
        """
        # Arrange
        # Create an instance of our dummy class to act as 'self'
        instance = DummyTestClass()

        # Mock the _build_dataset_sparse attribute of the instance
        # This simulates the method/attribute that would exist on the real 'self' object
        instance._build_dataset_sparse = MagicMock(name='_build_dataset_sparse_mock')

        # Mock the verify_fn that will be passed as an argument to testSparse
        mock_verify_fn = MagicMock(name='verify_fn_mock')

        # Act
        instance.testSparse(mock_verify_fn)

        # Assert
        # Verify that verify_fn was called exactly once with the expected arguments
        mock_verify_fn.assert_called_once_with(
            instance,  # The 'self' instance itself
            instance._build_dataset_sparse,  # The mocked _build_dataset_sparse attribute
            num_outputs=2
        )

    # The testSparse method is very simple, consisting of a single function call.
    # It does not contain any branches, loops, or error handling logic within itself.
    # Therefore, a single test case covering the correct invocation of verify_fn
    # is sufficient to achieve 100% code coverage for the testSparse method.
    # No specific edge cases or error conditions are handled by testSparse's logic.

# Tests for method: _build_dataset_nested_sparse
import pytest
import unittest.mock
from unittest.mock import MagicMock, patch

# Assume the method _build_dataset_nested_sparse belongs to a class
# For testing purposes, we'll create a placeholder class.
# In a real scenario, you would import the actual class, e.g.,
# from your_module import YourClass

# We need to mock tensorflow.python.data.ops.dataset_ops.Dataset
# This path might vary slightly depending on the exact TensorFlow version
# and how dataset_ops is exposed. This is a common internal path.
# If tensorflow.data.Dataset is the public API, you might patch that instead.
# For this example, we'll use the more internal path as it's often where
# Dataset is directly defined.

# Define a dummy class to hold the method for testing purposes
class DummyClass:
    def _build_dataset_nested_sparse(self):
        # This implementation is copied directly from the problem description
        # We need to import dataset_ops here or mock it globally if it's
        # a module-level import in the real code.
        # For this test, we'll assume dataset_ops is accessible or mocked.
        import tensorflow.python.data.ops.dataset_ops as dataset_ops
        return dataset_ops.Dataset.range(10).map(self._sparse).batch(5).batch(2)

    # _sparse is an internal method that needs to exist for the call
    # but its implementation doesn't matter for this test.
    def _sparse(self, element):
        pass


@patch('tensorflow.python.data.ops.dataset_ops.Dataset', autospec=True)
class TestBuildDatasetNestedSparse:

    def test__build_dataset_nested_sparse_happy_path(self, mock_Dataset):
        # Arrange
        instance = DummyClass()
        instance._sparse = MagicMock(name='_sparse_method')

        # Mock the chained calls: range().map().batch().batch()
        mock_dataset_instance = MagicMock(name='mock_dataset_instance')
        mock_Dataset.range.return_value = mock_dataset_instance

        # Ensure map and batch return the mock_dataset_instance itself for chaining
        mock_dataset_instance.map.return_value = mock_dataset_instance
        mock_dataset_instance.batch.return_value = mock_dataset_instance

        # Act
        result = instance._build_dataset_nested_sparse()

        # Assert
        # 1. Check Dataset.range was called correctly
        mock_Dataset.range.assert_called_once_with(10)

        # 2. Check map was called correctly
        mock_dataset_instance.map.assert_called_once_with(instance._sparse)

        # 3. Check batch was called twice with the correct arguments in order
        expected_batch_calls = [
            unittest.mock.call(5),
            unittest.mock.call(2)
        ]
        mock_dataset_instance.batch.assert_has_calls(expected_batch_calls)
        assert mock_dataset_instance.batch.call_count == 2

        # 4. Ensure the final result is the chained mock object
        assert result is mock_dataset_instance

    def test__build_dataset_nested_sparse_range_failure(self, mock_Dataset):
        # Arrange
        instance = DummyClass()
        instance._sparse = MagicMock(name='_sparse_method')

        # Simulate an error when range is called
        mock_Dataset.range.side_effect = RuntimeError("Range operation failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Range operation failed"):
            instance._build_dataset_nested_sparse()

        mock_Dataset.range.assert_called_once_with(10)
        # No further calls should happen if range fails
        assert not mock_Dataset.range.return_value.map.called
        assert not mock_Dataset.range.return_value.batch.called

    def test__build_dataset_nested_sparse_map_failure(self, mock_Dataset):
        # Arrange
        instance = DummyClass()
        instance._sparse = MagicMock(name='_sparse_method')

        mock_dataset_instance = MagicMock(name='mock_dataset_instance')
        mock_Dataset.range.return_value = mock_dataset_instance
        mock_dataset_instance.map.side_effect = ValueError("Map transformation failed")

        # Act & Assert
        with pytest.raises(ValueError, match="Map transformation failed"):
            instance._build_dataset_nested_sparse()

        mock_Dataset.range.assert_called_once_with(10)
        mock_dataset_instance.map.assert_called_once_with(instance._sparse)
        # No batch calls should happen if map fails
        assert not mock_dataset_instance.batch.called

    def test__build_dataset_nested_sparse_first_batch_failure(self, mock_Dataset):
        # Arrange
        instance = DummyClass()
        instance._sparse = MagicMock(name='_sparse_method')

        mock_dataset_instance = MagicMock(name='mock_dataset_instance')
        mock_Dataset.range.return_value = mock_dataset_instance
        mock_dataset_instance.map.return_value = mock_dataset_instance

        # Simulate failure on the first batch call (batch(5))
        mock_dataset_instance.batch.side_effect = [
            TypeError("Batch size must be positive"), # For batch(5)
            MagicMock() # This won't be called
        ]

        # Act & Assert
        with pytest.raises(TypeError, match="Batch size must be positive"):
            instance._build_dataset_nested_sparse()

        mock_Dataset.range.assert_called_once_with(10)
        mock_dataset_instance.map.assert_called_once_with(instance._sparse)
        # Check that the first batch call was made
        mock_dataset_instance.batch.assert_called_once_with(5)
        assert mock_dataset_instance.batch.call_count == 1 # Only the first call happened

    def test__build_dataset_nested_sparse_second_batch_failure(self, mock_Dataset):
        # Arrange
        instance = DummyClass()
        instance._sparse = MagicMock(name='_sparse_method')

        mock_dataset_instance = MagicMock(name='mock_dataset_instance')
        mock_Dataset.range.return_value = mock_dataset_instance
        mock_dataset_instance.map.return_value = mock_dataset_instance

        # Simulate failure on the second batch call (batch(2))
        mock_dataset_instance.batch.side_effect = [
            mock_dataset_instance, # For batch(5)
            IndexError("Batch index out of bounds") # For batch(2)
        ]

        # Act & Assert
        with pytest.raises(IndexError, match="Batch index out of bounds"):
            instance._build_dataset_nested_sparse()

        mock_Dataset.range.assert_called_once_with(10)
        mock_dataset_instance.map.assert_called_once_with(instance._sparse)
        # Check that both batch calls were attempted
        expected_batch_calls = [
            unittest.mock.call(5),
            unittest.mock.call(2)
        ]
        mock_dataset_instance.batch.assert_has_calls(expected_batch_calls)
        assert mock_dataset_instance.batch.call_count == 2

# Tests for method: testNestedSparse
import pytest
from unittest.mock import MagicMock, patch

# Assume the class containing testNestedSparse is part of a module like 'tensorflow'.
# For the purpose of testing, we will create a mockable structure for 'self'.
# In a real scenario, you would import the actual class, e.g.:
# from tensorflow.python.data.experimental.kernel_tests.checkpoint_test_base import CheckpointTestBase

# Define a dummy class to represent 'self' and its expected attributes/methods
# as used by testNestedSparse.
class MockSelf:
    def __init__(self):
        # Mock the internal method/attribute that testNestedSparse accesses
        self._build_dataset_nested_sparse = MagicMock(name='_build_dataset_nested_sparse_mock')

    # The method under test
    def testNestedSparse(self, verify_fn):
        verify_fn(self, self._build_dataset_nested_sparse, num_outputs=1)

def test_testNestedSparse_normal_execution():
    """
    Test that testNestedSparse correctly calls verify_fn with the expected arguments.
    """
    mock_self = MockSelf()
    mock_verify_fn = MagicMock(name='verify_fn_mock')

    # Call the method under test
    mock_self.testNestedSparse(mock_verify_fn)

    # Assert that verify_fn was called exactly once
    mock_verify_fn.assert_called_once()

    # Assert the arguments passed to verify_fn
    # The first argument should be the instance of self (mock_self)
    # The second argument should be self._build_dataset_nested_sparse (mock_self._build_dataset_nested_sparse)
    # The third argument should be num_outputs=1
    mock_verify_fn.assert_called_once_with(
        mock_self,
        mock_self._build_dataset_nested_sparse,
        num_outputs=1
    )

def test_testNestedSparse_verify_fn_raises_exception():
    """
    Test that an exception raised by verify_fn is propagated.
    """
    mock_self = MockSelf()
    mock_verify_fn = MagicMock(name='verify_fn_mock')
    mock_verify_fn.side_effect = ValueError("Test exception from verify_fn")

    # Expect the ValueError to be raised
    with pytest.raises(ValueError, match="Test exception from verify_fn"):
        mock_self.testNestedSparse(mock_verify_fn)

    # Ensure verify_fn was still called once before the exception
    mock_verify_fn.assert_called_once_with(
        mock_self,
        mock_self._build_dataset_nested_sparse,
        num_outputs=1
    )

