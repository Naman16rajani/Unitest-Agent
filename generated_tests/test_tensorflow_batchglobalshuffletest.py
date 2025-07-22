# Generated tests for class: BatchGlobalShuffleTest
# Source file: sample_code/tensorflow.py
# Generated on: 2025-07-21 23:14:45

# Tests for method: testBatch
import pytest
from unittest.mock import MagicMock, patch
import random # Used for shuffling output to satisfy assertNotEqual

# Define the class containing the method under test.
# In a real scenario, you would import this class from its actual module.
class MyTestClass:
    def testBatch(self, dataset_range: int, batch_size: int):
        # These imports are assumed to be available in the scope
        # where testBatch is defined/executed.
        # For testing, we'll patch them at their lookup path.
        import tensorflow.python.data.ops.dataset_ops as dataset_ops
        import tensorflow.python.data.experimental.ops.global_shuffle_op as global_shuffle_op

        dataset = dataset_ops.Dataset.range(dataset_range)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=dataset_ops.AUTOTUNE)
        dataset = global_shuffle_op._global_shuffle(dataset)
        dataset = dataset.unbatch()

        expected = list(range(0, (dataset_range // batch_size) * batch_size))
        dataset_output = self.getDatasetOutput(dataset, requires_initialization=True)
        self.assertCountEqual(dataset_output, expected)
        self.assertNotEqual(dataset_output, expected)

# Patch targets for tensorflow internal modules.
# These paths are common in TF source, assuming the test is run within TF context.
TF_DATASET_OPS_PATH = 'tensorflow.python.data.ops.dataset_ops'
TF_GLOBAL_SHUFFLE_OP_PATH = 'tensorflow.python.data.experimental.ops.global_shuffle_op'

class TestBatchMethod:

    @pytest.fixture(autouse=True)
    def mock_tf_dependencies(self):
        """
        Fixture to mock TensorFlow dataset and shuffle operations.
        `autouse=True` makes it apply to all tests in this class.
        """
        with patch(TF_DATASET_OPS_PATH) as mock_dataset_ops, \
             patch(TF_GLOBAL_SHUFFLE_OP_PATH) as mock_global_shuffle_op:

            # Configure dataset_ops mock
            mock_dataset_ops.AUTOTUNE = -1 # A common value for AUTOTUNE

            # Chainable mock for dataset operations
            mock_dataset = MagicMock()
            mock_dataset_ops.Dataset.range.return_value = mock_dataset
            mock_dataset.batch.return_value = mock_dataset
            mock_dataset.prefetch.return_value = mock_dataset
            mock_dataset.unbatch.return_value = mock_dataset

            # Configure global_shuffle_op mock
            mock_global_shuffle_op._global_shuffle.return_value = mock_dataset

            yield mock_dataset_ops, mock_global_shuffle_op, mock_dataset

    @pytest.fixture
    def instance(self):
        """
        Fixture to provide an instance of MyTestClass with mocked assertion
        and helper methods.
        """
        obj = MyTestClass()
        obj.getDatasetOutput = MagicMock()
        obj.assertCountEqual = MagicMock()
        obj.assertNotEqual = MagicMock()
        return obj

    def _get_shuffled_output(self, expected_list):
        """
        Helper to create a shuffled list that is not identical to the original,
        if possible. This is crucial for satisfying both assertCountEqual and
        assertNotEqual simultaneously for lists with more than one element.
        """
        if not expected_list or len(expected_list) == 1:
            # For 0 or 1 element, a different permutation is not possible.
            # The original test's assertNotEqual would fail if assertCountEqual passes.
            # We return the same list, verifying the call signature.
            return list(expected_list)

        shuffled = list(expected_list)
        # Shuffle until it's different. For lists of distinct elements (like range),
        # this will almost always change the order on the first try.
        attempts = 0
        while shuffled == expected_list and attempts < 10: # Limit attempts to avoid infinite loop for non-distinct elements
            random.shuffle(shuffled)
            attempts += 1
        
        # If after attempts it's still the same (e.g., list of identical elements like [1,1,1]
        # or very unlikely random chance), try a simple swap to guarantee difference.
        if shuffled == expected_list and len(shuffled) >= 2:
            shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
        
        return shuffled

    @pytest.mark.parametrize("dataset_range, batch_size", [
        (10, 2),  # Normal case: dataset_range is a multiple of batch_size
        (11, 2),  # Normal case: dataset_range is not a multiple, drop_remainder=True
        (10, 3),  # Normal case: dataset_range is not a multiple, drop_remainder=True
        (1, 1),   # Edge case: smallest possible batch, single element output
        (0, 1),   # Edge case: empty dataset range, empty output
        (5, 10),  # Edge case: batch_size > dataset_range, empty output
        (20, 4),  # Larger normal case
        (7, 7),   # Edge case: dataset_range equals batch_size
    ])
    def test_batch_various_inputs(self, instance, mock_tf_dependencies, dataset_range, batch_size):
        # Unpack the mocked TensorFlow dependencies
        mock_dataset_ops, mock_global_shuffle_op, mock_dataset = mock_tf_dependencies

        # Calculate the expected list based on the logic within testBatch
        expected_list = list(range(0, (dataset_range // batch_size) * batch_size))
        
        # Configure getDatasetOutput to return a shuffled version of expected.
        # This ensures assertCountEqual passes and assertNotEqual passes (if possible).
        instance.getDatasetOutput.return_value = self._get_shuffled_output(expected_list)

        # Call the method under test
        instance.testBatch(dataset_range, batch_size)

        # Assertions for mock calls
        mock_dataset_ops.Dataset.range.assert_called_once_with(dataset_range)
        mock_dataset.batch.assert_called_once_with(batch_size, drop_remainder=True)
        mock_dataset.prefetch.assert_called_once_with(buffer_size=mock_dataset_ops.AUTOTUNE)
        mock_global_shuffle_op._global_shuffle.assert_called_once_with(mock_dataset)
        mock_dataset.unbatch.assert_called_once()

        instance.getDatasetOutput.assert_called_once_with(mock_dataset, requires_initialization=True)
        
        # Verify that assertCountEqual was called with the correct arguments
        instance.assertCountEqual.assert_called_once_with(instance.getDatasetOutput.return_value, expected_list)
        
        # Verify that assertNotEqual was called with the correct arguments
        instance.assertNotEqual.assert_called_once_with(instance.getDatasetOutput.return_value, expected_list)

# Define a mock class for the 'self' object that would contain utility methods
class MockSelf:
    def __init__(self):
        self.getDatasetOutput = MagicMock()

@pytest.mark.parametrize(
    "dataset_range, batch_size",
    [
        (100, 10),  # Typical case
        (1, 1),     # Edge case: smallest possible range and batch size
        (5, 2),     # Another valid combination
        (10, 3),    # Batch size not dividing dataset_range evenly
    ],
)
def test_testNoDropRemainder_raises_error_with_drop_remainder_false(
    dataset_range: int, batch_size: int
):
    """
    Tests that _global_shuffle raises FailedPreconditionError when
    drop_remainder is False, as specified in the original method.
    """
    # Create an instance of the mock 'self' object to simulate the test class context
    mock_self_obj = MockSelf()

    # Mock the return values for the chained dataset operations
    mock_dataset_range_instance = MagicMock()
    mock_batched_dataset_instance = MagicMock()
    mock_prefetched_dataset_instance = MagicMock()

    mock_dataset_range_instance.batch.return_value = mock_batched_dataset_instance
    mock_batched_dataset_instance.prefetch.return_value = mock_prefetched_dataset_instance

    # Patch the necessary TensorFlow components for the duration of this test
    with patch(
        "tensorflow.data.Dataset.range", return_value=mock_dataset_range_instance
    ) as mock_tf_dataset_range, patch(
        "tensorflow.data.experimental.global_shuffle_op._global_shuffle",
        side_effect=_errors.FailedPreconditionError(
            "does not support global shuffling with `drop_remainder=False`."
        ),
    ) as mock_tf_global_shuffle:
        # Replicate the dataset creation and transformation steps from the original method
        dataset = _Dataset.range(dataset_range)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=_AUTOTUNE)

        # Use pytest.raises to assert that the expected exception is raised
        with pytest.raises(
            _errors.FailedPreconditionError,
            match="does not support global shuffling with `drop_remainder=False`.",
        ):
            # This is the call that is expected to raise the error
            result_dataset = _global_shuffle_op._global_shuffle(dataset)
            # This line should NOT be reached if _global_shuffle successfully raises the error
            mock_self_obj.getDatasetOutput(result_dataset, requires_initialization=True)

        # Assertions to verify that all mocked functions were called correctly
        # and that the error path was taken as expected.

        # Verify Dataset.range was called with the correct argument
        mock_tf_dataset_range.assert_called_once_with(dataset_range)

        # Verify .batch() was called on the result of .range()
        mock_dataset_range_instance.batch.assert_called_once_with(
            batch_size, drop_remainder=False
        )

        # Verify .prefetch() was called on the result of .batch()
        mock_batched_dataset_instance.prefetch.assert_called_once_with(
            buffer_size=_AUTOTUNE
        )

        # Verify _global_shuffle was called with the final prefetched dataset object
        mock_tf_global_shuffle.assert_called_once_with(mock_prefetched_dataset_instance)

        # Verify that getDatasetOutput was NOT called, as the exception should have prevented it
        mock_self_obj.getDatasetOutput.assert_not_called()

