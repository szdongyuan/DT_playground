---
name: test-running
description: Run Python tests with pytest, analyze failures, generate coverage reports, and configure test environments. Use when executing tests, checking coverage, debugging test failures, validating changes, or working with pytest commands and configuration.
---

# Test Running

## Overview

Comprehensive guide for executing Python tests using pytest, analyzing test results, generating coverage reports, and configuring test environments. Covers common pytest commands, coverage analysis, and CI/CD integration.

## Workflow

```
Configure environment → Execute tests → Analyze results → Generate coverage report → Fix failures
```

## Prerequisites

Ensure required packages are installed:

```bash
pip install pytest pytest-cov pytest-asyncio pytest-xdist
```

## Instructions

### Step 1: Configure Test Environment

#### pytest.ini Configuration

Create `pytest.ini` in project root:

```ini
[pytest]
# Test discovery
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output options
addopts = -v --tb=short --strict-markers

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests

# Async mode
asyncio_mode = auto

# Warnings
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

#### pyproject.toml Configuration

Alternative configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/migrations/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
fail_under = 80
show_missing = true
```

### Step 2: Execute Tests

#### Basic Commands

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest -v` | Verbose output |
| `pytest -vv` | More verbose output |
| `pytest -q` | Quiet output |
| `pytest -x` | Stop on first failure |
| `pytest --pdb` | Enter debugger on failure |

#### Run Specific Tests

```bash
# Run tests in a specific file
pytest tests/test_module.py

# Run tests in a specific directory
pytest tests/unit/

# Run a specific test class
pytest tests/test_module.py::TestClassName

# Run a specific test function
pytest tests/test_module.py::TestClassName::test_method

# Run tests matching a keyword expression
pytest -k "login"
pytest -k "login and not admin"
pytest -k "test_user or test_admin"
```

#### Run Tests by Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run all except slow tests
pytest -m "not slow"

# Run unit OR integration tests
pytest -m "unit or integration"

# Combine markers
pytest -m "integration and not slow"
```

#### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto          # Auto-detect CPU count
pytest -n 4             # Use 4 workers
pytest -n auto --dist loadfile  # Group by file
```

### Step 3: Analyze Test Results

#### Output Formats

| Option | Description |
|--------|-------------|
| `--tb=short` | Short traceback |
| `--tb=long` | Long traceback |
| `--tb=line` | One line per failure |
| `--tb=no` | No traceback |
| `--tb=native` | Python standard traceback |

#### Failed Test Analysis

```bash
# Show only failed tests
pytest --last-failed     # or --lf
pytest --failed-first    # or --ff

# Show local variables in traceback
pytest --showlocals      # or -l

# Show N slowest tests
pytest --durations=10
pytest --durations=0     # Show all durations
```

#### JUnit XML Report

```bash
# Generate JUnit XML for CI/CD
pytest --junitxml=results.xml
```

#### HTML Report

```bash
# Generate HTML report (requires pytest-html)
pip install pytest-html
pytest --html=report.html --self-contained-html
```

### Step 4: Generate Coverage Reports

#### Run with Coverage

```bash
# Basic coverage
pytest --cov=src

# Coverage with report
pytest --cov=src --cov-report=term

# Detailed terminal report
pytest --cov=src --cov-report=term-missing

# HTML coverage report
pytest --cov=src --cov-report=html

# XML coverage report (for CI/CD)
pytest --cov=src --cov-report=xml

# Multiple reports
pytest --cov=src --cov-report=term-missing --cov-report=html
```

#### Coverage Analysis

| Metric | Description | Target |
|--------|-------------|--------|
| **Line Coverage** | % of lines executed | 80%+ |
| **Branch Coverage** | % of branches taken | 70%+ |
| **Function Coverage** | % of functions called | 90%+ |

#### Interpreting Coverage Report

```
Name                      Stmts   Miss Branch BrPart  Cover   Missing
----------------------------------------------------------------------
src/module.py                50      5     12      2    88%   23, 45-47
src/utils.py                 30      0      6      0   100%
----------------------------------------------------------------------
TOTAL                        80      5     18      2    92%
```

| Column | Meaning |
|--------|---------|
| **Stmts** | Total statements |
| **Miss** | Statements not executed |
| **Branch** | Total branch points |
| **BrPart** | Branches partially covered |
| **Cover** | Coverage percentage |
| **Missing** | Line numbers not covered |

#### Identify Uncovered Code

```bash
# Show missing lines in terminal
pytest --cov=src --cov-report=term-missing

# Generate HTML with line highlighting
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Step 5: Handle Test Failures

#### Debugging Strategies

| Strategy | Command/Action |
|----------|----------------|
| **Stop on first failure** | `pytest -x` |
| **Enter debugger** | `pytest --pdb` |
| **Show local variables** | `pytest -l` |
| **Run failed only** | `pytest --lf` |
| **Increase verbosity** | `pytest -vvv` |

#### Common Failure Patterns

| Pattern | Cause | Solution |
|---------|-------|----------|
| `AssertionError` | Assertion failed | Check expected vs actual values |
| `AttributeError` | Missing attribute | Check mock setup, object state |
| `ImportError` | Module not found | Check PYTHONPATH, package structure |
| `fixture not found` | Missing fixture | Check conftest.py, fixture scope |
| `Timeout` | Test took too long | Add timeout marker or optimize |

#### Re-run Strategies

```bash
# Run only failed tests from last run
pytest --lf

# Run failed tests first, then all
pytest --ff

# Clear cache and run all
pytest --cache-clear

# Run with step-by-step failure (stop after each)
pytest -x --pdb
```

### Step 6: Advanced Configuration

#### Environment-Specific Settings

```python
# conftest.py
import os
import pytest

def pytest_configure(config):
    """Set up test environment."""
    os.environ["TESTING"] = "true"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up and tear down test environment."""
    # Setup
    print("\nSetting up test environment...")

    yield

    # Teardown
    print("\nTearing down test environment...")
```

#### Custom Markers

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow running tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "smoke: smoke tests")

def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless explicitly requested."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="run slow tests")
```

#### Timeout Configuration

```bash
# Install pytest-timeout
pip install pytest-timeout

# Run with timeout
pytest --timeout=30

# Per-test timeout
@pytest.mark.timeout(10)
def test_slow_operation():
    ...
```

### Step 7: CI/CD Integration

#### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --junitxml=results.xml

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        files: coverage.xml
```

#### GitLab CI Example

```yaml
test:
  stage: test
  script:
    - pip install -r requirements.txt
    - pytest --cov=src --cov-report=xml --junitxml=report.xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      junit: report.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

## Quick Reference

### Common Command Combinations

| Purpose | Command |
|---------|---------|
| **Quick check** | `pytest -q` |
| **Verbose with coverage** | `pytest -v --cov=src` |
| **Debug mode** | `pytest -x --pdb -l` |
| **CI/CD run** | `pytest --cov=src --cov-report=xml --junitxml=results.xml` |
| **Fast feedback** | `pytest --lf -x` |
| **Full report** | `pytest -v --cov=src --cov-report=html --durations=10` |

### Pytest Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | Some tests failed |
| 2 | Test execution interrupted |
| 3 | Internal error |
| 4 | Command line usage error |
| 5 | No tests collected |

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Tests not discovered | Check naming conventions (test_*.py) |
| Fixtures not found | Check conftest.py location |
| Import errors | Add project root to PYTHONPATH |
| Async tests failing | Install pytest-asyncio, add asyncio_mode |
| Coverage missing lines | Check source= in coverage config |

### Project Structure Recommendation

```
project/
├── src/
│   └── package/
│       ├── __init__.py
│       └── module.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_module.py
│   └── integration/
│       ├── __init__.py
│       └── test_module_integration.py
├── pytest.ini
├── pyproject.toml
└── requirements-dev.txt
```

## Examples

### Example 1: Running Tests During Development

```bash
# Initial run - see what we have
pytest --collect-only

# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# After making changes, run only affected tests
pytest --lf

# Before commit, run full suite
pytest -v --cov=src
```

### Example 2: Investigating a Failure

```bash
# Run the failing test with debug info
pytest tests/test_module.py::test_failing -vvv -l

# Enter debugger at failure
pytest tests/test_module.py::test_failing --pdb

# Check if it's an isolation issue
pytest tests/test_module.py::test_failing --forked
```

### Example 3: Coverage Improvement

```bash
# Generate HTML report to see uncovered lines
pytest --cov=src --cov-report=html

# Open htmlcov/index.html and identify gaps

# Run specific uncovered code paths
pytest -v -k "edge_case" --cov=src
```

## Integration with Other Skills

| Skill | Relationship |
|-------|--------------|
| `test_designing` | Use BEFORE writing tests to design test cases |
| `test_writing` | Use BEFORE this skill to write test code |
| `code_implementing_features` | Run tests after implementing features |
| `code_fixing_bugs` | Run tests to verify bug fixes |
