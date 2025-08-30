import time
import random
import math
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class DataProcessor:
    """
    Complex data processing class with various methods and time-consuming operations
    """

    def __init__(self, dataset_size: int, processing_mode: str = "standard",
                 enable_caching: bool = True, config: Dict[str, Any] = None):
        """
        Complex constructor with conditional logic, loops, and function calls
        """
        print(f"Initializing DataProcessor with {dataset_size} records...")

        self.dataset_size = dataset_size
        self.processing_mode = processing_mode
        self.enable_caching = enable_caching
        self.cache = {}
        self.processed_data = []
        self.metrics = {"processing_time": 0, "operations_count": 0}

        # Complex initialization logic with if-else and loops
        if config is None:
            config = self._generate_default_config()

        self.config = config

        # Initialize processing parameters based on mode
        if processing_mode == "fast":
            self.batch_size = 1000
            self.precision = 0.1
        elif processing_mode == "accurate":
            self.batch_size = 100
            self.precision = 0.001
        else:  # standard mode
            self.batch_size = 500
            self.precision = 0.01

        # Initialize data structures with loops
        for i in range(min(dataset_size, 10000)):
            if i % 1000 == 0:
                self.cache[f"checkpoint_{i}"] = self._create_checkpoint(i)

        # Validate configuration
        self._validate_configuration()

        print(f"DataProcessor initialized successfully in {processing_mode} mode")

    def _generate_default_config(self) -> Dict[str, Any]:
        """Helper method called during initialization"""
        return {
            "max_iterations": 1000,
            "threshold": 0.95,
            "enable_logging": True,
            "parallel_processing": False
        }

    def _create_checkpoint(self, index: int) -> Dict[str, Any]:
        """Helper method for creating checkpoints"""
        return {
            "index": index,
            "timestamp": time.time(),
            "status": "initialized"
        }

    def _validate_configuration(self):
        """Validate the configuration parameters"""
        required_keys = ["max_iterations", "threshold"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

    def process_data(self, data: List[float]) -> List[float]:
        """
        Main processing method that calls other methods
        """
        print("Starting data processing...")
        start_time = time.time()

        # Preprocess data
        cleaned_data = self._preprocess_data(data)

        # Apply transformations
        transformed_data = self._apply_transformations(cleaned_data)

        # Perform complex calculations
        result = self._perform_calculations(transformed_data)

        # Post-process results
        final_result = self._postprocess_results(result)

        processing_time = time.time() - start_time
        self.metrics["processing_time"] += processing_time
        self.metrics["operations_count"] += 1

        print(f"Data processing completed in {processing_time:.2f} seconds")
        return final_result

    def _preprocess_data(self, data: List[float]) -> List[float]:
        """
        Preprocessing method with nested function
        """

        def remove_outliers(values: List[float], threshold: float = 2.0) -> List[float]:
            """Nested function to remove outliers"""
            if not values:
                return values

            mean_val = sum(values) / len(values)
            std_dev = math.sqrt(sum((x - mean_val) ** 2 for x in values) / len(values))

            return [x for x in values if abs(x - mean_val) <= threshold * std_dev]

        def normalize_data(values: List[float]) -> List[float]:
            """Nested function to normalize data"""
            if not values:
                return values

            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [0.5] * len(values)

            return [(x - min_val) / (max_val - min_val) for x in values]

        print("Preprocessing data...")

        # Remove outliers using nested function
        cleaned_data = remove_outliers(data)

        # Normalize data using nested function
        normalized_data = normalize_data(cleaned_data)

        return normalized_data

    def _apply_transformations(self, data: List[float]) -> List[float]:
        """Apply mathematical transformations"""
        print("Applying transformations...")

        transformed = []
        for value in data:
            # Apply different transformations based on processing mode
            if self.processing_mode == "fast":
                result = value * 2
            elif self.processing_mode == "accurate":
                result = math.sin(value) + math.cos(value * 2)
            else:
                result = math.log(abs(value) + 1) if value != 0 else 0

            transformed.append(result)

        return transformed

    def time_consuming_analysis(self, data: List[float]) -> Dict[str, float]:
        """
        Time-consuming method that simulates heavy computation
        """
        print("Starting time-consuming analysis...")
        start_time = time.time()

        results = {
            "mean": 0,
            "variance": 0,
            "correlation_matrix": 0,
            "complexity_score": 0
        }

        # Simulate time-consuming operations
        for i in range(self.config["max_iterations"]):
            if i % 100 == 0:
                print(f"Progress: {i}/{self.config['max_iterations']}")

            # Simulate heavy computation with sleep
            time.sleep(0.001)  # Simulate processing time

            # Perform calculations
            if data:
                temp_result = sum(x * math.sin(i * 0.01) for x in data[:min(len(data), 100)])
                results["complexity_score"] += temp_result / self.config["max_iterations"]

        # Calculate final statistics
        if data:
            results["mean"] = sum(data) / len(data)
            mean_val = results["mean"]
            results["variance"] = sum((x - mean_val) ** 2 for x in data) / len(data)

        analysis_time = time.time() - start_time
        print(f"Time-consuming analysis completed in {analysis_time:.2f} seconds")

        return results

    def _perform_calculations(self, data: List[float]) -> List[float]:
        """Perform complex calculations calling other methods"""
        print("Performing calculations...")

        # Call time-consuming analysis
        analysis_results = self.time_consuming_analysis(data)

        # Apply results to modify data
        complexity_factor = analysis_results.get("complexity_score", 1.0)

        calculated_data = []
        for value in data:
            new_value = value * complexity_factor + analysis_results.get("mean", 0)
            calculated_data.append(new_value)

        return calculated_data

    def _postprocess_results(self, data: List[float]) -> List[float]:
        """Final post-processing step"""
        print("Post-processing results...")

        # Cache results if enabled
        if self.enable_caching:
            cache_key = f"result_{len(data)}_{time.time()}"
            self.cache[cache_key] = data.copy()

        return data

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return self.metrics.copy()

    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        print("Cache cleared")


class AdvancedDataProcessor(DataProcessor):
    """
    Advanced data processor that inherits from DataProcessor and overrides methods
    """

    def __init__(self, dataset_size: int, processing_mode: str = "advanced",
                 enable_caching: bool = True, config: Dict[str, Any] = None,
                 ml_enabled: bool = True):
        """
        Enhanced constructor that calls parent constructor
        """
        print("Initializing AdvancedDataProcessor...")

        # Call parent constructor
        super().__init__(dataset_size, processing_mode, enable_caching, config)

        self.ml_enabled = ml_enabled
        self.model_accuracy = 0.0
        self.advanced_cache = {}

        # Additional initialization for advanced features
        if ml_enabled:
            self._initialize_ml_components()

    def _initialize_ml_components(self):
        """Initialize machine learning components"""
        print("Initializing ML components...")
        self.model_accuracy = random.uniform(0.85, 0.99)
        print(f"ML model initialized with accuracy: {self.model_accuracy:.3f}")

    def process_data(self, data: List[float]) -> List[float]:
        """
        Override parent's process_data method with advanced features
        """
        print("Starting advanced data processing...")
        start_time = time.time()

        # Use parent's preprocessing
        cleaned_data = self._preprocess_data(data)

        # Apply advanced ML-based processing
        if self.ml_enabled:
            ml_processed_data = self._apply_ml_processing(cleaned_data)
        else:
            ml_processed_data = cleaned_data

        # Call parent's calculation method
        calculated_data = super()._perform_calculations(ml_processed_data)

        # Apply advanced post-processing
        final_result = self._advanced_postprocessing(calculated_data)

        processing_time = time.time() - start_time
        self.metrics["processing_time"] += processing_time
        self.metrics["operations_count"] += 1

        print(f"Advanced data processing completed in {processing_time:.2f} seconds")
        return final_result

    def _apply_ml_processing(self, data: List[float]) -> List[float]:
        """Apply machine learning-based processing"""
        print("Applying ML processing...")

        # Simulate ML model inference time
        time.sleep(0.1)

        # Apply ML transformations
        ml_data = []
        for i, value in enumerate(data):
            # Simulate ML prediction
            prediction_factor = self.model_accuracy * math.sin(i * 0.1)
            ml_enhanced_value = value * (1 + prediction_factor * 0.1)
            ml_data.append(ml_enhanced_value)

        return ml_data

    def _advanced_postprocessing(self, data: List[float]) -> List[float]:
        """Advanced post-processing with additional features"""
        print("Applying advanced post-processing...")

        # Call parent's post-processing
        processed_data = super()._postprocess_results(data)

        # Apply additional advanced processing
        enhanced_data = []
        for value in processed_data:
            # Apply advanced mathematical operations
            enhanced_value = value * math.tanh(value) + math.log(abs(value) + 1)
            enhanced_data.append(enhanced_value)

        # Store in advanced cache
        if self.enable_caching:
            cache_key = f"advanced_{len(data)}_{time.time()}"
            self.advanced_cache[cache_key] = enhanced_data.copy()

        return enhanced_data

    def time_consuming_analysis(self, data: List[float]) -> Dict[str, float]:
        """
        Override parent's time-consuming analysis with ML enhancements
        """
        print("Starting advanced time-consuming analysis...")

        # Call parent's analysis
        parent_results = super().time_consuming_analysis(data)

        # Add ML-specific analysis
        ml_results = self._ml_analysis(data)

        # Combine results
        combined_results = {**parent_results, **ml_results}
        return combined_results

    def _ml_analysis(self, data: List[float]) -> Dict[str, float]:
        """ML-specific analysis"""
        print("Performing ML analysis...")
        start_time = time.time()

        ml_results = {
            "ml_accuracy": self.model_accuracy,
            "prediction_confidence": 0.0,
            "feature_importance": 0.0
        }

        # Simulate ML analysis time
        for i in range(100):
            time.sleep(0.002)  # Simulate ML computation
            ml_results["prediction_confidence"] += random.uniform(0.001, 0.01)

        ml_results["feature_importance"] = ml_results["prediction_confidence"] * 0.5

        analysis_time = time.time() - start_time
        print(f"ML analysis completed in {analysis_time:.2f} seconds")

        return ml_results


def complex_data_processing_pipeline(dataset_sizes: List[int], use_advanced: bool = False):
    """
    Main function that uses the classes above with subfunctions
    """
    print("=" * 60)
    print("STARTING COMPLEX DATA PROCESSING PIPELINE")
    print("=" * 60)

    def generate_sample_data(size: int) -> List[float]:
        """Subfunction to generate sample data"""
        print(f"Generating {size} sample data points...")
        return [random.gauss(0, 1) for _ in range(size)]

    def validate_results(results: List[float]) -> bool:
        """Subfunction to validate processing results"""
        if not results:
            return False

        # Check for NaN or infinite values
        for value in results:
            if math.isnan(value) or math.isinf(value):
                return False

        return True

    def generate_report(processor, results: List[float], dataset_size: int) -> Dict[str, Any]:
        """Subfunction to generate processing report"""
        metrics = processor.get_metrics()

        report = {
            "dataset_size": dataset_size,
            "result_count": len(results),
            "processing_time": metrics.get("processing_time", 0),
            "operations_count": metrics.get("operations_count", 0),
            "validation_passed": validate_results(results),
            "processor_type": type(processor).__name__
        }

        if results:
            report["result_statistics"] = {
                "min": min(results),
                "max": max(results),
                "mean": sum(results) / len(results),
                "range": max(results) - min(results)
            }

        return report

    # Main processing logic
    all_reports = []

    for size in dataset_sizes:
        print(f"\n--- Processing dataset of size {size} ---")

        # Generate sample data using subfunction
        sample_data = generate_sample_data(size)

        # Choose processor type
        if use_advanced:
            processor = AdvancedDataProcessor(
                dataset_size=size,
                processing_mode="advanced",
                enable_caching=True,
                ml_enabled=True
            )
        else:
            processor = DataProcessor(
                dataset_size=size,
                processing_mode="accurate",
                enable_caching=True
            )

        # Process the data
        try:
            results = processor.process_data(sample_data)

            # Generate report using subfunction
            report = generate_report(processor, results, size)
            all_reports.append(report)

            print(f"Processing completed successfully for dataset size {size}")
            print(f"Validation: {'PASSED' if report['validation_passed'] else 'FAILED'}")

        except Exception as e:
            print(f"Error processing dataset size {size}: {str(e)}")
            error_report = {
                "dataset_size": size,
                "error": str(e),
                "processor_type": type(processor).__name__
            }
            all_reports.append(error_report)

        finally:
            # Cleanup
            processor.clear_cache()

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)

    total_processing_time = sum(r.get("processing_time", 0) for r in all_reports)
    successful_runs = sum(1 for r in all_reports if r.get("validation_passed", False))

    print(f"Total datasets processed: {len(all_reports)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Processor type used: {'Advanced' if use_advanced else 'Standard'}")

    return all_reports


# Example usage and demonstration
if __name__ == "__main__":
    # Test with different dataset sizes
    test_sizes = [100, 500, 1000]

    print("Testing Standard DataProcessor:")
    standard_reports = complex_data_processing_pipeline(test_sizes, use_advanced=False)

    print("\n" + "=" * 80 + "\n")

    print("Testing Advanced DataProcessor:")
    advanced_reports = complex_data_processing_pipeline(test_sizes, use_advanced=True)

    # Compare performance
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    for i, size in enumerate(test_sizes):
        if i < len(standard_reports) and i < len(advanced_reports):
            std_time = standard_reports[i].get("processing_time", 0)
            adv_time = advanced_reports[i].get("processing_time", 0)

            print(f"Dataset size {size}:")
            print(f"  Standard: {std_time:.2f}s")
            print(f"  Advanced: {adv_time:.2f}s")
            print(f"  Difference: {adv_time - std_time:+.2f}s")
            print()
