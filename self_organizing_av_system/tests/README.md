# Self-Organizing AV System Tests

This directory contains test scripts for verifying the functionality of the Self-Organizing Audio-Visual System.

## Test Files

- `test_full_pipeline.py`: Tests the full processing pipeline from sensory input to action output, ensuring structural plasticity changes are properly synchronized across the entire system.
- `test_structural_sync.py`: Tests synchronization between the structural plasticity component and multimodal state.
- `test_system.py`: Basic system functionality tests.
- `run_tests.py`: A simple test runner to execute all tests.

## Running Tests

You can run the tests using any of the following methods:

```bash
# Run all tests in test_full_pipeline.py
python test_full_pipeline.py

# Run a specific test
python -m unittest test_full_pipeline.TestFullPipeline.test_full_pipeline_with_resizing

# Use the test runner
python run_tests.py
```

## Test Coverage

The tests verify:

1. **Structural Plasticity Synchronization**: The system's handling of dynamic neural growth and pruning.
2. **Multimodal State Consistency**: Ensuring the multimodal state size stays in sync with structural changes.
3. **Action Generation**: Testing that actions remain consistent after network structure changes.
4. **Pixel Output**: Testing direct pixel output consistency after resizing.
5. **Temporal Prediction**: Testing temporal prediction capabilities after structural changes.
6. **Robustness**: Testing system robustness with multiple resize operations.
7. **Adaptability**: Testing the system's ability to learn new patterns after resizing.

## Implementation Notes

The test setup uses mocks for the visual and audio processors to simplify testing and avoid external dependencies, while testing the actual system integration with the structural plasticity component.

When testing the system's resizing capabilities, note that there's a `max_size` parameter in the configuration that limits growth beyond a certain point (default: 100 neurons). 