# Tests for Road-Sense Project

This directory contains unit tests for the Road-Sense project.

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_augmentations.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run with coverage report:
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run specific test class:
```bash
pytest tests/test_augmentations.py::TestTrainingAugmentation
```

### Run specific test function:
```bash
pytest tests/test_augmentations.py::TestTrainingAugmentation::test_training_augmentation_without_bbox
```

## Test Structure

- `test_augmentations.py` - Tests for data augmentation pipelines
- `test_kitti_utils.py` - Tests for KITTI utility functions
- More test files will be added as features are developed

## Writing New Tests

Follow pytest conventions:
- Test files should start with `test_`
- Test classes should start with `Test`
- Test functions should start with `test_`
- Use fixtures for reusable test data
- Use descriptive test names

Example:
```python
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

class TestMyFeature:
    def test_basic_functionality(self, sample_data):
        assert sample_data["key"] == "value"
```

## Dependencies

Make sure pytest is installed:
```bash
pip install pytest pytest-cov
```

## Continuous Integration

These tests should be run automatically in CI/CD pipeline before merging any changes.
