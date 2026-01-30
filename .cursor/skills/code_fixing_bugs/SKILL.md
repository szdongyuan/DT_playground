---
name: code_fixing_bugs
description: Guide for fixing bugs with systematic debugging and root cause analysis. Use when fixing bugs, debugging issues, resolving defects, or when the user reports a problem that needs to be fixed.
---

# Fixing Bugs

## Overview

Standard workflow for fixing bugs after the preparation phase (issue created, branch created). Guides through reproduction, severity assessment, root cause analysis, fix implementation, and verification stages.

## Language Requirement

**IMPORTANT: All code comments MUST be written entirely in English.**

This includes:
- Inline comments
- Block comments
- Docstrings and function documentation
- TODO/FIXME annotations

Rationale:
- Ensures code is accessible to all contributors
- Maintains consistency across the codebase
- Aligns with common open-source practices

## Workflow

```
1. Reproduce → 2. Assess severity → 3. Analyze root cause → 4. Implement fix → 5. Test → 5.5. Regression test → 6. Self-review → 7. Document & commit
```

## Prerequisites

Before using this skill, ensure:

- [ ] Bug issue has been created (recommended)
- [ ] Development branch has been created (bugfix or hotfix)
- [ ] Currently on the correct branch

**Verify with:**
```bash
git branch --show-current
```

## Instructions

### Step 1: Reproduce the Bug

Before fixing, ensure the bug can be consistently reproduced:

#### Reproduction Checklist

| Item | Description |
|------|-------------|
| **Steps to reproduce** | Exact sequence of actions |
| **Expected behavior** | What should happen |
| **Actual behavior** | What actually happens |
| **Environment** | OS, browser, version, etc. |
| **Frequency** | Always, sometimes, rarely |

#### Gather Evidence

Collect relevant information:

- Error messages and stack traces
- Console/application logs
- Screenshots or screen recordings
- Network requests (if applicable)
- Database state (if applicable)

**If bug cannot be reproduced:**
> "I cannot reproduce this bug with the provided information. Could you provide:
> 1. Exact steps to trigger the issue
> 2. Any error messages you see
> 3. Your environment details (browser, OS, version)"

### Step 2: Assess Severity

Evaluate the bug's severity to determine the appropriate response:

#### Severity Classification

| Severity | Definition | Response | Branch Strategy |
|----------|------------|----------|-----------------|
| **Critical** | System crash, data loss, security vulnerability | Immediate | Hotfix from main |
| **High** | Core functionality broken, no workaround | Same day | Hotfix or bugfix |
| **Medium** | Feature impaired but workaround exists | This week | Standard bugfix |
| **Low** | Minor issue, cosmetic defect | Next iteration | Standard bugfix |

#### Severity Indicators

| Severity | Indicators |
|----------|------------|
| **Critical** | Production down, data corruption, security breach, payment failures |
| **High** | Login broken, core feature unusable, blocking multiple users |
| **Medium** | Feature partially working, edge case failures, degraded performance |
| **Low** | Typos, minor UI issues, rare edge cases, cosmetic problems |

### Step 2.5: Hotfix Flow (Critical/High Only)

For Critical or High severity bugs requiring immediate attention:

#### Hotfix Branch Creation

```bash
# Create hotfix branch directly from main/master
git fetch origin
git checkout -b hotfix/BUG_ID_description origin/main
```

#### Hotfix Principles

| Principle | Description |
|-----------|-------------|
| **Minimal changes** | Only fix the specific issue |
| **Fast verification** | Quick smoke test, not full regression |
| **Immediate deploy** | Deploy as soon as verified |
| **Backport later** | Sync changes to develop after deploy |

#### Post-Hotfix Sync

After hotfix is deployed, sync to develop branch:

```bash
git checkout develop
git merge hotfix/BUG_ID_description
git push origin develop
```

### Step 3: Root Cause Analysis

Perform systematic analysis to identify the true cause:

#### Analysis Methods

| Method | When to Use | How |
|--------|-------------|-----|
| **Binary search** | Large codebase, unclear location | Narrow down by halves |
| **Log tracing** | Runtime errors, data flow issues | Follow the execution path |
| **Breakpoint debugging** | Logic errors, state issues | Step through code |
| **Git bisect** | Regression, worked before | Find the breaking commit |
| **Rubber duck** | Complex logic, stuck | Explain the problem aloud |

#### Common Root Cause Types

| Category | Common Causes | Example |
|----------|---------------|---------|
| **Logic errors** | Wrong condition, off-by-one, incorrect algorithm | `if (x > 0)` should be `if (x >= 0)` |
| **Null/undefined** | Missing null checks, uninitialized variables | `user.name` when user is null |
| **Type errors** | Wrong type, implicit conversion | String "5" + 1 = "51" |
| **Race conditions** | Async timing, concurrent access | Data loaded before render |
| **State management** | Stale state, incorrect updates | Redux state not updating |
| **API/Integration** | Wrong endpoint, bad request format | 404 or 500 errors |
| **Configuration** | Wrong environment, missing config | Dev config in production |
| **Dependencies** | Version conflicts, breaking changes | Library update broke code |

#### Root Cause Documentation

Document your findings:

```markdown
## Bug Analysis

**Symptom:** [What user sees]

**Root Cause:** [Technical explanation]

**Location:** [File and line number]

**Why it happened:** [How this bug was introduced]
```

#### Complex Bug Analysis - Switch to Plan Mode

**Recommend switching to Plan mode when:**

- Bug involves multiple components or systems
- Root cause is unclear after initial analysis
- Multiple potential causes need investigation
- Fix may have significant side effects

**Suggest to user:**
> "This bug appears to involve multiple components. I recommend switching to Plan mode to discuss the analysis approach and potential solutions before making changes. Would you like to switch?"

### Step 4: Implement Fix

Apply the fix with minimal changes:

#### Fix Guidelines

| Guideline | Description |
|-----------|-------------|
| **Minimal change** | Only modify what's necessary to fix the bug |
| **Root cause fix** | Fix the actual cause, not just symptoms |
| **No refactoring** | Resist urge to clean up unrelated code |
| **Preserve behavior** | Don't change existing functionality |
| **Add guards** | Prevent similar issues in the future |
| **English comments** | All comments and docstrings must be in English |

#### Implementation Workflow

1. **Locate the issue** - Find the exact code causing the bug
2. **Understand context** - Read surrounding code
3. **Plan the fix** - Decide on minimal changes needed
4. **Make changes** - Edit files carefully
5. **Check for errors** - Verify no syntax/lint errors

#### Mode Selection for Complex Fixes

| Complexity | Indicators | Recommended Mode |
|------------|------------|------------------|
| **Simple** | Single location, clear fix | Agent mode |
| **Moderate** | Multiple files, clear approach | Agent mode |
| **Complex** | Architectural implications, multiple solutions | Plan mode |

### Step 5: Test Fix

Verify the bug is fixed and no new issues introduced:

#### Automated Tests

Run existing tests to ensure no regression:

| Test Type | Command Example | Purpose |
|-----------|-----------------|---------|
| Unit tests | `pytest`, `npm test` | Verify individual components |
| Integration tests | `pytest tests/integration/` | Verify component interaction |
| Lint check | `ruff check`, `eslint` | Code quality |

**Note:** All existing tests must pass before proceeding.

#### Manual Verification

Verify the specific bug is fixed:

1. **Reproduce original issue** - Follow the original steps
2. **Confirm fix** - Issue should no longer occur
3. **Test variations** - Try related scenarios
4. **Check side effects** - Verify affected features still work

**Ask user to test:**

> "I've implemented the fix. Please verify:
> 
> 1. **Original issue** - Follow the reproduction steps. Is the bug fixed?
> 2. **Related scenarios** - Test similar functionality
> 3. **Side effects** - Check that related features still work correctly
> 
> Let me know the results."

### Step 5.5: Regression Test (Optional)

Consider adding a test to prevent this bug from recurring:

#### When to Add Regression Tests

| Add Test | Skip Test |
|----------|-----------|
| Bug could recur easily | One-time configuration issue |
| Critical functionality | External dependency issue |
| Complex logic involved | Simple typo fix |
| Edge case not covered | Already covered by existing tests |

#### Regression Test Template

```python
def test_bug_ISSUE_ID_description():
    """
    Regression test for BUG-123: Description of the bug.
    
    This test ensures the bug does not recur.
    """
    # Setup: Create the conditions that triggered the bug
    
    # Action: Perform the action that previously failed
    
    # Assert: Verify the correct behavior
```

**Ask user:**
> "Would you like to add a regression test for this bug? This would help prevent similar issues in the future."

### Step 6: Self-Review

Before committing, review the changes:

#### Bug Fix Review Checklist

| Aspect | Check |
|--------|-------|
| **Root cause fixed** | Does the fix address the actual cause? |
| **Minimal changes** | Only necessary changes made? |
| **No new bugs** | Could the fix introduce new issues? |
| **Edge cases** | Are boundary conditions handled? |
| **Error handling** | Are errors caught gracefully? |
| **No debug code** | Removed console.log, print statements? |
| **Tests pass** | All automated tests passing? |

#### Agent-Assisted Review

Request agent to review:
> "Please review the bug fix and check for any issues."

Agent will check:
- Fix addresses root cause
- No unintended side effects
- Code quality maintained
- Best practices followed

### Step 7: Documentation and Commit

Complete the fix with proper documentation:

#### Update Bug Issue

If there's an associated issue, update it with:
- Root cause explanation
- Fix description
- Any follow-up actions needed

#### Commit Message Format

Use conventional commit format for bug fixes:

```
fix(scope): short description

Fixes #ISSUE_ID

Root cause: [Brief explanation]
Solution: [What was changed]
```

**Examples:**
```
fix(auth): handle null user in login validation

Fixes #BUG2026012801

Root cause: User object could be null when session expired
Solution: Added null check before accessing user properties
```

#### Pre-Commit Verification

Verify ready for commit:

- [ ] Bug is confirmed fixed
- [ ] All tests pass
- [ ] No new issues introduced
- [ ] Code reviewed
- [ ] Commit message follows format

#### Prompt User for Commit

**Ask user:**

> "The bug fix is complete and verified. Would you like to commit your changes?
> 
> Changes summary:
> - [List of modified files]
> - Root cause: [Brief explanation]
> - Fix: [What was changed]
> 
> If ready, I can help you commit using the `git_pushing_changes` workflow."

## Quick Reference

### Severity Decision Guide

```
Is production affected?
├── Yes → Is data at risk or system down?
│         ├── Yes → Critical (Hotfix immediately)
│         └── No → High (Hotfix today)
└── No → Is core functionality broken?
         ├── Yes → Medium (Fix this week)
         └── No → Low (Next iteration)
```

### Complete Workflow Summary

| Phase | Key Actions | Mode |
|-------|-------------|------|
| **Reproduce** | Confirm bug, gather evidence | Agent |
| **Assess** | Determine severity | Agent |
| **Hotfix** | Create hotfix branch (if Critical/High) | Agent |
| **Analyze** | Find root cause | Agent/Plan |
| **Fix** | Implement minimal fix | Agent |
| **Test** | Verify fix, run tests | Agent |
| **Regression** | Add test (optional) | Agent |
| **Review** | Self-review changes | Agent |
| **Commit** | Document and commit | Agent |

## Examples

### Example 1: Simple Bug Fix

**Context:** Button click not responding

**Workflow:**
1. Reproduce: Click button, nothing happens
2. Assess: Medium severity (feature impaired)
3. Analyze: Event handler typo `onCLick` instead of `onClick`
4. Fix: Correct the typo
5. Test: Button now works
6. Commit: `fix(ui): correct button click handler typo`

### Example 2: Critical Bug (Hotfix)

**Context:** Users cannot login after deployment

**Workflow:**
1. Reproduce: Login returns 500 error
2. Assess: Critical (core functionality broken)
3. Create hotfix branch from main
4. Analyze: Environment variable missing in production
5. Fix: Add missing configuration
6. Quick test: Login works
7. Deploy immediately
8. Backport to develop
9. Post-mortem: Add config validation

### Example 3: Complex Bug (Plan Mode)

**Context:** Data inconsistency in reports

**Workflow:**
1. Reproduce: Some reports show wrong totals
2. Assess: High severity
3. Analyze: Multiple potential causes identified
4. Switch to Plan mode: Discuss race condition vs calculation error
5. In Plan mode: Determine it's a race condition
6. Return to Agent mode: Implement locking mechanism
7. Add regression test
8. Commit with detailed explanation

## Integration with Other Skills

| Skill | Relationship |
|-------|--------------|
| `git_submitting_issues` | Used BEFORE this skill to create bug issue |
| `git_creating_branch` | Used BEFORE this skill to create bugfix/hotfix branch |
| `git_pushing_changes` | Used AFTER this skill to commit changes |
| `git_creating_pr` | Used AFTER committing to create pull request |

## Communication Patterns

### Suggesting Plan Mode for Complex Bugs

```
"This bug involves [complexity factors]. To ensure we fix the root cause 
correctly, I recommend switching to Plan mode to:
- Analyze potential causes
- Discuss fix approaches
- Identify potential side effects

Would you like to switch to Plan mode?"
```

### After Fix Implementation

```
"I've implemented the fix for [bug description]. Here's a summary:

Root cause: [Explanation]
Fix: [What was changed]

Please verify:
1. The original issue is resolved
2. Related functionality still works
3. No new issues appear

Let me know the test results."
```

### Ready to Commit

```
"The bug fix is verified and ready to commit. Would you like to:
1. Commit these changes?
2. Add a regression test first?
3. Review the changes again?
```
