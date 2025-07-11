# Tests

This directory contains comprehensive unit tests for the MLOps project.

## Test Structure

### `test_data.py`
Tests for the data processing module (`src/data.py`):

- **TestMyDataset**: Tests for the PyTorch dataset class
  - Dataset creation and loading
  - Label mapping and encoding
  - Error handling for invalid inputs
  - Edge cases (empty files, missing columns)

- **TestPreprocessData**: Tests for data preprocessing function
  - Train/test/eval split creation
  - Label encoding
  - Data validation
  - Logging functionality

- **TestCreateHFDatasets**: Tests for Hugging Face dataset creation
  - Tokenizer integration
  - Dataset creation with different parameters
  - Error handling

- **TestDataIntegration**: Integration tests for the complete data pipeline

### `test_model.py`
Tests for the model module (`src/model.py`):

- **TestClassifier**: Tests for the PyTorch Classifier model
  - Model creation and architecture
  - Forward pass functionality
  - Gradient flow and training
  - Device handling (CPU/GPU)
  - Model saving and loading
  - Parameter counting
  - Dropout behavior

- **TestCreateHFModel**: Tests for Hugging Face model creation
  - Model instantiation with different pretrained models
  - Parameter validation
  - Error handling

- **TestModelIntegration**: Integration tests for model functionality
  - Complete training steps
  - Evaluation procedures

- **TestModelEdgeCases**: Edge case testing
  - Empty batches
  - Very long texts
  - Special characters
  - Mixed case text

## Running Tests

### Basic pytest commands:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data.py
pytest tests/test_model.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test classes
pytest tests/test_data.py::TestMyDataset
pytest tests/test_model.py::TestClassifier
```

### Test Markers:
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run both unit and integration tests
pytest -m "unit or integration"
```

### For CI/CD workflows:
```bash
# Run tests with coverage and XML report
pytest --cov=src --cov-report=xml --cov-report=term-missing

# Run tests with JUnit XML report
pytest --junitxml=test-results.xml

# Run tests in parallel (if pytest-xdist is installed)
pytest -n auto
```

## Test Coverage

The tests cover:

1. **Data Module**:
   - ✅ Dataset creation and loading
   - ✅ Data preprocessing and splitting
   - ✅ Label encoding and mapping
   - ✅ Error handling and validation
   - ✅ Hugging Face dataset integration

2. **Model Module**:
   - ✅ PyTorch model architecture
   - ✅ Forward pass functionality
   - ✅ Training and evaluation
   - ✅ Model saving and loading
   - ✅ Hugging Face model creation
   - ✅ Edge cases and error handling

## Test Dependencies

The tests require:
- `pytest`
- `pytest-cov` (for coverage reports)
- `pandas`
- `numpy`
- `torch`
- `transformers`
- `unittest.mock`

## Best Practices

1. **Isolation**: Each test is independent and doesn't rely on other tests
2. **Fixtures**: Use pytest fixtures for common test data and setup
3. **Mocking**: External dependencies (like Hugging Face models) are mocked
4. **Temporary files**: Tests use temporary files/directories that are cleaned up
5. **Error handling**: Tests verify proper error handling for edge cases
6. **Integration**: Include integration tests for complete workflows

## Adding New Tests

When adding new functionality:

1. Create corresponding test classes
2. Use descriptive test method names
3. Test both success and failure cases
4. Use appropriate fixtures for test data
5. Mock external dependencies
6. Add integration tests for complete workflows

## Continuous Integration

These tests are designed to work with CI/CD workflows like `tests.yaml`:

```yaml
# Example workflow step
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest --cov=src --cov-report=xml --cov-report=term-missing
    pytest --junitxml=test-results.xml

- name: Run unit tests only
  run: |
    pytest -m unit --cov=src --cov-report=xml

- name: Run integration tests only
  run: |
    pytest -m integration --cov=src --cov-report=xml
```

## Coverage Reports

Generate coverage reports:
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

This will create:
- HTML coverage report in `htmlcov/`
- Terminal output with missing lines
- XML report for CI integration 