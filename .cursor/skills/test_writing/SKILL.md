---
name: test_writing
description: Guide for writing Python tests using pytest with advanced mock techniques, fixtures, and parametrization. Use when writing unit tests, integration tests, or when the user needs help with pytest and mocking.
---

# Test Writing

## Overview

Comprehensive guide for writing Python tests using pytest framework with advanced mocking techniques. Covers test structure, fixtures, parametrization, and mocking external dependencies.

## Workflow

```
Create test file → Set up fixtures → Write test cases → Add mocks → Verify tests pass
```

## Prerequisites

Ensure required packages are installed:

```bash
pip install pytest pytest-cov pytest-asyncio responses
```

## Instructions

### Step 1: Create Test File Structure

#### Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| Test file | `test_<module>.py` | `test_user_service.py` |
| Test function | `test_<action>_<scenario>` | `test_login_success` |
| Test class | `Test<ClassName>` | `TestUserService` |
| Fixture | `<resource>_fixture` or descriptive | `db_session`, `mock_client` |

#### Directory Structure

```
project/
├── src/
│   └── module.py
└── tests/
    ├── __init__.py
    ├── conftest.py          # Shared fixtures
    ├── unit/
    │   ├── __init__.py
    │   └── test_module.py
    └── integration/
        ├── __init__.py
        └── test_module_integration.py
```

### Step 2: Write Basic Tests

#### Test Function Template

```python
def test_<action>_<scenario>():
    """
    Test that <expected behavior> when <condition>.
    """
    # Arrange - Set up test data and conditions
    input_data = ...
    expected = ...
    
    # Act - Execute the code under test
    result = function_under_test(input_data)
    
    # Assert - Verify the result
    assert result == expected
```

#### Test Class Template

```python
class TestClassName:
    """Tests for ClassName."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.instance = ClassName()
    
    def teardown_method(self):
        """Clean up after each test method."""
        pass
    
    def test_method_success(self):
        """Test method succeeds with valid input."""
        result = self.instance.method(valid_input)
        assert result == expected
```

### Step 3: Use Fixtures

Fixtures provide reusable test data and setup.

#### Basic Fixture

```python
import pytest

@pytest.fixture
def sample_user():
    """Provide a sample user for testing."""
    return {
        "id": 1,
        "name": "Test User",
        "email": "test@example.com"
    }

def test_user_display(sample_user):
    result = format_user(sample_user)
    assert "Test User" in result
```

#### Fixture with Cleanup

```python
@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    yield file_path  # Test runs here
    # Cleanup happens automatically with tmp_path
```

#### Fixture Scopes

| Scope | Lifetime | Use Case |
|-------|----------|----------|
| `function` (default) | Each test | Most cases |
| `class` | Per test class | Shared class state |
| `module` | Per test file | Expensive setup |
| `session` | Entire test run | Database connection |

```python
@pytest.fixture(scope="module")
def database_connection():
    """Create database connection once per module."""
    conn = create_connection()
    yield conn
    conn.close()
```

#### Shared Fixtures in conftest.py

```python
# tests/conftest.py
import pytest

@pytest.fixture
def api_client():
    """Shared API client fixture."""
    return APIClient(base_url="http://test.local")

@pytest.fixture
def authenticated_user():
    """Shared authenticated user fixture."""
    return User(id=1, token="test-token")
```

### Step 4: Parametrized Tests

Run the same test with multiple inputs.

#### Basic Parametrization

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (0, 0),
    (-1, -2),
])
def test_double(input, expected):
    assert double(input) == expected
```

#### Multiple Parameters

```python
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

#### Parametrize with IDs

```python
@pytest.mark.parametrize("email,valid", [
    ("user@example.com", True),
    ("invalid-email", False),
    ("", False),
    ("user@domain", False),
], ids=["valid_email", "no_at_sign", "empty", "no_tld"])
def test_validate_email(email, valid):
    assert validate_email(email) == valid
```

### Step 5: Mocking with unittest.mock

#### Basic Mock

```python
from unittest.mock import Mock, MagicMock

def test_with_mock():
    # Create a mock object
    mock_service = Mock()
    mock_service.get_data.return_value = {"key": "value"}
    
    # Use the mock
    result = mock_service.get_data()
    
    # Verify
    assert result == {"key": "value"}
    mock_service.get_data.assert_called_once()
```

#### Patch Decorator

```python
from unittest.mock import patch

@patch("module.external_function")
def test_with_patch(mock_func):
    mock_func.return_value = "mocked result"
    
    result = function_that_calls_external()
    
    assert result == "expected based on mocked result"
    mock_func.assert_called_once()
```

#### Patch as Context Manager

```python
from unittest.mock import patch

def test_with_context_manager():
    with patch("module.external_function") as mock_func:
        mock_func.return_value = "mocked"
        
        result = function_under_test()
        
        assert result == "expected"
```

#### Patch Object Attribute

```python
from unittest.mock import patch

@patch.object(MyClass, "method_name")
def test_patch_object(mock_method):
    mock_method.return_value = "mocked"
    
    obj = MyClass()
    result = obj.method_name()
    
    assert result == "mocked"
```

### Step 6: Advanced Mocking

#### side_effect for Multiple Returns

```python
from unittest.mock import Mock

def test_multiple_returns():
    mock = Mock()
    mock.method.side_effect = [1, 2, 3]
    
    assert mock.method() == 1
    assert mock.method() == 2
    assert mock.method() == 3
```

#### side_effect for Exceptions

```python
from unittest.mock import Mock

def test_exception():
    mock = Mock()
    mock.method.side_effect = ValueError("error message")
    
    with pytest.raises(ValueError, match="error message"):
        mock.method()
```

#### side_effect with Function

```python
from unittest.mock import Mock

def custom_side_effect(x):
    if x < 0:
        raise ValueError("negative")
    return x * 2

def test_custom_side_effect():
    mock = Mock()
    mock.process.side_effect = custom_side_effect
    
    assert mock.process(5) == 10
    with pytest.raises(ValueError):
        mock.process(-1)
```

#### Mock Verification

```python
from unittest.mock import Mock, call

def test_mock_verification():
    mock = Mock()
    
    mock.method("arg1")
    mock.method("arg2", key="value")
    
    # Verify calls
    mock.method.assert_called()
    mock.method.assert_called_with("arg2", key="value")
    assert mock.method.call_count == 2
    
    # Verify call order
    mock.method.assert_has_calls([
        call("arg1"),
        call("arg2", key="value")
    ])
```

#### PropertyMock

```python
from unittest.mock import patch, PropertyMock

def test_property_mock():
    with patch.object(MyClass, "my_property", new_callable=PropertyMock) as mock_prop:
        mock_prop.return_value = "mocked value"
        
        obj = MyClass()
        assert obj.my_property == "mocked value"
```

### Step 7: Async Testing

#### pytest-asyncio Setup

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

#### AsyncMock

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_async_mock():
    mock_client = AsyncMock()
    mock_client.fetch.return_value = {"data": "value"}
    
    result = await mock_client.fetch()
    
    assert result == {"data": "value"}
    mock_client.fetch.assert_awaited_once()
```

#### Patching Async Functions

```python
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
@patch("module.async_external_call", new_callable=AsyncMock)
async def test_patched_async(mock_call):
    mock_call.return_value = "mocked"
    
    result = await function_that_awaits_external()
    
    assert result == "expected"
```

### Step 8: Mocking External Dependencies

#### HTTP API Mock with responses

```python
import responses
import requests

@responses.activate
def test_api_call():
    responses.add(
        responses.GET,
        "https://api.example.com/data",
        json={"key": "value"},
        status=200
    )
    
    result = requests.get("https://api.example.com/data")
    
    assert result.json() == {"key": "value"}
```

#### Database Mock

```python
from unittest.mock import Mock, patch

@patch("module.db_session")
def test_database_operation(mock_session):
    mock_query = Mock()
    mock_query.filter.return_value.first.return_value = User(id=1, name="Test")
    mock_session.query.return_value = mock_query
    
    result = get_user_by_id(1)
    
    assert result.name == "Test"
```

#### File System Mock

```python
from unittest.mock import patch, mock_open

def test_file_read():
    mock_content = "file content"
    
    with patch("builtins.open", mock_open(read_data=mock_content)):
        result = read_file("any_path.txt")
    
    assert result == mock_content
```

#### Environment Variables Mock

```python
from unittest.mock import patch
import os

@patch.dict(os.environ, {"API_KEY": "test-key", "DEBUG": "true"})
def test_with_env_vars():
    assert os.environ["API_KEY"] == "test-key"
    assert os.environ["DEBUG"] == "true"
```

#### Time Mock

```python
from unittest.mock import patch
from datetime import datetime

@patch("module.datetime")
def test_with_frozen_time(mock_datetime):
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
    
    result = get_current_timestamp()
    
    assert result == "2024-01-01 12:00:00"
```

## Quick Reference

### Common Assertions

| Assertion | Purpose |
|-----------|---------|
| `assert x == y` | Equality |
| `assert x != y` | Inequality |
| `assert x is None` | None check |
| `assert x is not None` | Not None |
| `assert x in collection` | Membership |
| `assert isinstance(x, Type)` | Type check |
| `pytest.raises(Error)` | Exception expected |
| `pytest.approx(x)` | Float comparison |

### Mock Assertion Methods

| Method | Purpose |
|--------|---------|
| `assert_called()` | Called at least once |
| `assert_called_once()` | Called exactly once |
| `assert_called_with(*args, **kwargs)` | Last call arguments |
| `assert_called_once_with(*args, **kwargs)` | Single call with arguments |
| `assert_not_called()` | Never called |
| `assert_has_calls(calls)` | Specific call sequence |
| `assert_awaited()` | Awaited (async) |

### Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.skip` | Skip test |
| `@pytest.mark.skipif(condition)` | Conditional skip |
| `@pytest.mark.xfail` | Expected failure |
| `@pytest.mark.asyncio` | Async test |
| `@pytest.mark.parametrize` | Multiple inputs |
| `@pytest.mark.slow` | Custom marker |

## Integration with Other Skills

| Skill | Relationship |
|-------|--------------|
| `test_designing` | Use BEFORE this skill to design tests |
| `test_running` | Use AFTER this skill to execute tests |
| `code_implementing_features` | Write tests after implementing features |
| `code_fixing_bugs` | Write regression tests after fixing bugs |
