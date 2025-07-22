# Generated tests for class: BatchRandomAccessTest
# Source file: sample_code/tensorflow.py
# Generated on: 2025-07-21 23:12:04

# Tests for method: testBasic


# Tests for method: testRandomAccessBatchWithShuffle
import pytest
import unittest.mock
import numpy as np

# Import the actual modules/classes that the original method would interact with.
# These imports are for the *test file* to be able to patch them.
# The original method itself would have these imports.
# We're simulating the environment where the original test method runs.
# If tensorflow is not installed, these imports will fail.
# For the purpose of generating the test code, we assume these are valid paths.
try:
    import tensorflow.data.dataset_ops
    import tensorflow.data.experimental.random_access
except ImportError:
    # Create dummy modules for patching if TensorFlow is not available,
    # to allow the test code to be generated without import errors.
    # In a real setup, TensorFlow would be installed.
    class MockDatasetOps:
        class Dataset:
            @classmethod
            def from_tensor_slices(cls, data):
                pass
    tensorflow = unittest.mock.MagicMock()
    tensorflow.data = unittest.mock.MagicMock()
    tensorflow.data.dataset_ops = MockDatasetOps()
    tensorflow.data.experimental = unittest.mock.MagicMock()
    tensorflow.data.experimental.random_access = unittest.mock.MagicMock()


# Define a dummy class to house the method under test.
# In a real scenario, you would import the actual class containing this method.
class MyTestClass:
    def assertAllEqual(self, expected: np.ndarray, actual: np.ndarray):
        # This method is part of the original test class (e.g., tf.test.TestCase).
        # We will mock this method in the pytest test to verify its calls.
        pass

    def evaluate(self, tensor: unittest.mock.MagicMock):
        # This method is part of the original test class.
        # We will mock this method in the pytest test.
        # It typically evaluates a TensorFlow tensor to a NumPy array.
        return tensor # Default behavior, will be overridden by mock

    def testRandomAccessBatchWithShuffle(self):
        # The original method code provided by the user.
        # Note: The original code uses `dataset_ops` and `random_access` directly.
        # This implies they are either globally available or imported at the module level.
        # For the purpose of patching, we will target their full paths.
        dataset = tensorflow.data.dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7])
        shuffle_dataset = dataset.shuffle(buffer_size=10, seed=2)
        batch_dataset = shuffle_dataset.batch(2)

        expected_output = [
            np.array([5, 2], dtype=np.int32),
            np.array([4, 7], dtype=np.int32),
            np.array([1, 3], dtype=np.int32),
            np.array([6], dtype=np.int32),
        ]
        for i in range(4):
            self.assertAllEqual(
                expected_output[i], self.evaluate(tensorflow.data.experimental.random_access.at(batch_dataset, i))
            )

        # Checks the order is consistent with shuffle dataset.
        for i in range(3):
            self.assertAllEqual(
                expected_output[i][0],
                self.evaluate(tensorflow.data.experimental.random_access.at(shuffle_dataset, i * 2)),
            )
            self.assertAllEqual(
                expected_output[i][1],
                self.evaluate(tensorflow.data.experimental.random_access.at(shuffle_dataset, (i * 2) + 1)),
            )

        # Checks the remainder is the last element in shuffled dataset.
        self.assertAllEqual(
            expected_output[3][0], self.evaluate(tensorflow.data.experimental.random_access.at(shuffle_dataset, 6))
        )


@unittest.mock.patch('tensorflow.data.experimental.random_access.at')
@unittest.mock.patch('tensorflow.data.dataset_ops.Dataset.from_tensor_slices')
def test_testRandomAccessBatchWithShuffle_normal_scenario(
    mock_from_tensor_slices: unittest.mock.MagicMock, mock_random_access_at: unittest.mock.MagicMock
):
    """
    Test the testRandomAccessBatchWithShuffle method for its normal execution path,
    verifying all internal calls and assertions.
    """
    # Setup mocks for the dataset creation chain
    mock_dataset = unittest.mock.MagicMock()
    mock_shuffle_dataset = unittest.mock.MagicMock()
    mock_batch_dataset = unittest.mock.MagicMock()

    mock_from_tensor_slices.return_value = mock_dataset
    mock_dataset.shuffle.return_value = mock_shuffle_dataset
    mock_shuffle_dataset.batch.return_value = mock_batch_dataset

    # Instantiate MyTestClass and mock its internal methods (assertAllEqual, evaluate)
    test_instance = MyTestClass()
    test_instance.assertAllEqual = unittest.mock.MagicMock()
    test_instance.evaluate = unittest.mock.MagicMock()

    # Define the expected outputs as per the original method's logic
    expected_output = [
        np.array([5, 2], dtype=np.int32),
        np.array([4, 7], dtype=np.int32),
        np.array([1, 3], dtype=np.int32),
        np.array([6], dtype=np.int32),
    ]

    # Configure mock_random_access_at's side_effect
    # It will return a unique mock tensor for each call, which evaluate will then process.

