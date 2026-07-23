---
name: test-designing
description: Design test cases using equivalence partitioning, boundary value analysis, scenario coverage, and dependency planning. Use when planning tests, designing test cases, analyzing what to test, defining coverage, or preparing mock strategies.
---

# Test Designing

## Overview

Systematic approach to designing effective test cases before writing test code. This skill helps identify what to test, which testing methods to apply, and how to plan mock strategies for dependencies.

## Workflow

```
Analyze code → Identify test points → Select design methods → Design test cases → Plan mock strategy
```

## Instructions

### Step 1: Analyze the Code Under Test

Before designing tests, understand what you're testing:

#### Analysis Checklist

| Aspect | Questions to Answer |
|--------|---------------------|
| **Purpose** | What does this code do? |
| **Inputs** | What parameters/data does it accept? |
| **Outputs** | What does it return or produce? |
| **Side effects** | Does it modify state, call APIs, write files? |
| **Dependencies** | What external services/modules does it use? |
| **Edge cases** | What unusual inputs might it receive? |

#### Code Classification

| Type | Characteristics | Test Focus |
|------|-----------------|------------|
| **Pure function** | No side effects, same input = same output | Input/output validation |
| **Stateful** | Modifies internal state | State transitions |
| **I/O dependent** | File, database, network operations | Mock external resources |
| **Async** | Uses async/await | Async behavior, timing |

### Step 2: Choose Test Type

#### Unit Test vs Integration Test

| Criteria | Unit Test | Integration Test |
|----------|-----------|------------------|
| **Scope** | Single function/class | Multiple components |
| **Dependencies** | All mocked | Real or partially mocked |
| **Speed** | Fast (< 100ms) | Slower (100ms - seconds) |
| **Purpose** | Verify logic correctness | Verify component interaction |
| **When to use** | Business logic, algorithms | API endpoints, DB operations |

**Decision Guide:**

```
Does it have external dependencies?
├── No → Unit test
└── Yes → Is testing the integration important?
          ├── Yes → Integration test
          └── No → Unit test with mocks
```

### Step 3: Apply Test Case Design Methods

#### Method 1: Equivalence Partitioning

Divide input domain into groups where all values in a group should behave the same.

**Process:**
1. Identify input parameters
2. Divide each parameter into valid and invalid partitions
3. Select one representative value from each partition

**Example:**

```python
# Function: validate_age(age: int) -> bool
# Returns True if age is valid (0-150)

# Partitions:
# - Invalid: age < 0 (e.g., -1)
# - Valid: 0 <= age <= 150 (e.g., 25)
# - Invalid: age > 150 (e.g., 200)

# Test cases:
# test_negative_age: age = -1 → False
# test_valid_age: age = 25 → True
# test_too_old: age = 200 → False
```

#### Method 2: Boundary Value Analysis

Test at the edges of equivalence partitions where bugs commonly occur.

**Process:**
1. Identify boundaries for each partition
2. Test values at, just below, and just above boundaries

**Boundary Types:**

| Boundary | Test Values |
|----------|-------------|
| Minimum | min, min-1, min+1 |
| Maximum | max, max-1, max+1 |
| Empty | empty, single item |
| Type limits | 0, -1, MAX_INT |

**Example:**

```python
# Function: validate_age(age: int) -> bool
# Valid range: 0-150

# Boundary test cases:
# test_age_negative_one: age = -1 → False (below min)
# test_age_zero: age = 0 → True (at min)
# test_age_one: age = 1 → True (above min)
# test_age_149: age = 149 → True (below max)
# test_age_150: age = 150 → True (at max)
# test_age_151: age = 151 → False (above max)
```

#### Method 3: Scenario Coverage

Test realistic user workflows and use cases.

**Process:**
1. Identify user scenarios/workflows
2. Map each scenario to test steps
3. Include both happy path and error paths

**Scenario Categories:**

| Category | Description | Examples |
|----------|-------------|----------|
| **Happy path** | Normal successful flow | User registers successfully |
| **Error path** | Expected failures | Invalid input rejected |
| **Edge case** | Unusual but valid | Empty list, max items |
| **Recovery** | Error handling | Network retry, rollback |

**Example:**

```python
# Scenario: User login workflow

# Happy path:
# test_login_success: valid credentials → logged in

# Error paths:
# test_login_wrong_password: correct email, wrong password → error
# test_login_user_not_found: unknown email → error
# test_login_account_locked: locked account → specific error

# Edge cases:
# test_login_case_insensitive_email: Email vs email → works
# test_login_whitespace_trimmed: "  email  " → works
```

#### Method 4: Error Guessing

Use experience to predict likely bugs.

**Common Bug Patterns:**

| Category | Typical Bugs |
|----------|--------------|
| **Null/None** | Null pointer, undefined access |
| **Empty** | Empty string, empty list, zero |
| **Boundary** | Off-by-one, overflow, underflow |
| **Type** | Wrong type, type coercion |
| **Concurrency** | Race conditions, deadlocks |
| **State** | Invalid state transitions |

### Step 4: Plan Mock Strategy

Identify what needs to be mocked and how:

#### Dependency Analysis

| Dependency Type | Mock Approach | Example |
|-----------------|---------------|---------|
| **Database** | Mock repository/ORM | `patch('db.session')` |
| **HTTP API** | Mock client/responses | `responses` library |
| **File system** | Mock file operations | `patch('builtins.open')` |
| **Time** | Mock datetime/time | `freezegun` or `patch` |
| **Random** | Mock random functions | `patch('random.randint')` |
| **Environment** | Mock env variables | `patch.dict(os.environ)` |

#### Mock Decision Matrix

```
Is the dependency:
├── External service (API, DB) → Always mock in unit tests
├── Slow operation → Mock for speed
├── Non-deterministic (time, random) → Mock for reproducibility
├── Has side effects (email, payment) → Mock to avoid real actions
└── Internal module → Usually don't mock (test integration)
```

### Step 5: Document Test Design

Create a test design document using the template.

Store project test-design documents under `docs/plans/` as local working artifacts by default. Do not stage, commit, push, or include them in a pull request unless the user explicitly asks to publish a specific plan.

**Output format:**

```markdown
## Test Design: [Feature/Module Name]

### Code Analysis
- Purpose: [What it does]
- Inputs: [Parameters]
- Dependencies: [External dependencies]

### Test Cases

| ID | Category | Description | Input | Expected Output |
|----|----------|-------------|-------|-----------------|
| TC001 | Happy path | [Description] | [Input] | [Output] |
| TC002 | Boundary | [Description] | [Input] | [Output] |

### Mock Strategy
- [Dependency]: [Mock approach]
```

## Test Coverage Guidelines

### Coverage Targets

| Test Type | Recommended Coverage |
|-----------|---------------------|
| Unit tests | 80%+ line coverage |
| Integration tests | Critical paths covered |
| Combined | 70-90% overall |

### What Must Be Tested

| Priority | What to Test |
|----------|--------------|
| **High** | Business logic, data validation, security |
| **Medium** | Error handling, edge cases |
| **Low** | Simple getters/setters, logging |

### What Not to Test

- Framework code (Django, Flask internals)
- Third-party libraries
- Generated code
- Simple property accessors

## Quick Reference

### Test Case Design Checklist

- [ ] Identified all input parameters
- [ ] Applied equivalence partitioning
- [ ] Tested boundary values
- [ ] Covered happy path scenarios
- [ ] Covered error scenarios
- [ ] Identified edge cases
- [ ] Planned mock strategy
- [ ] Documented test design

### Common Test Patterns

| Pattern | When to Use |
|---------|-------------|
| **Given-When-Then** | Behavior verification |
| **Arrange-Act-Assert** | State verification |
| **Setup-Exercise-Verify-Teardown** | Complex fixtures |

## Examples

### Example 1: Designing Tests for a Calculator Function

**Code:**
```python
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

**Test Design:**

| Method | Test Cases |
|--------|------------|
| **Equivalence** | Positive/positive, negative/positive, zero/positive |
| **Boundary** | Very small divisor, very large numbers |
| **Error** | Division by zero |

### Example 2: Designing Tests for User Service

**Code:**
```python
class UserService:
    def __init__(self, db, email_client):
        self.db = db
        self.email_client = email_client

    def register(self, email, password):
        # Validates, saves to DB, sends email
```

**Test Design:**

| Test Type | Mock Strategy |
|-----------|---------------|
| Unit test | Mock both `db` and `email_client` |
| Integration | Real `db`, mock `email_client` |

**Test Cases:**
- Happy path: Valid registration
- Validation: Invalid email format, weak password
- Duplicate: Email already exists
- Failure: DB error, email send failure

## Integration with Other Skills

| Skill | Relationship |
|-------|--------------|
| `test_writing` | Use AFTER this skill to write actual tests |
| `test_running` | Use AFTER test_writing to execute tests |
| `code_implementing_features` | Use this skill AFTER implementing features |
| `code_fixing_bugs` | Use this skill when adding regression tests |
