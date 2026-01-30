---
name: code_implementing_features
description: Guide for implementing new features after creating feature issue and local branch. Use when developing new functionality, making code changes, or when the user is ready to start coding a feature.
---

# Implementing Features

## Overview

Standard workflow for implementing new features after the preparation phase (issue created, branch created). Guides through analysis, implementation, testing, and completion stages with appropriate mode selection based on change complexity.

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
1. Assess complexity → 2. Analyze requirements → 3. Implement changes → 4. Test → 4.5. Self-review → 5. Document & commit
```

## Prerequisites

Before using this skill, ensure:

- [ ] Feature issue has been created (optional but recommended)
- [ ] Local development branch has been created
- [ ] Currently on the correct development branch

**Verify with:**
```bash
git branch --show-current
```

## Instructions

### Step 1: Assess Change Complexity

Evaluate the scope of changes to determine the appropriate approach:

| Complexity | Indicators | Recommended Mode |
|------------|------------|------------------|
| **Small** | Single file, < 50 lines, clear implementation | Agent mode (direct implementation) |
| **Medium** | 2-5 files, clear requirements, moderate logic | Agent mode with brief planning |
| **Large** | Multiple files, architectural decisions, unclear requirements | Plan mode (collaborative planning) |
| **Complex** | Cross-cutting concerns, API changes, data model changes | Plan mode (comprehensive analysis) |

### Step 2: Mode Selection

#### For Large/Complex Changes - Switch to Plan Mode

**Recommend switching to Plan mode when:**

- Requirements are ambiguous or incomplete
- Multiple valid implementation approaches exist
- Architectural decisions are needed
- Changes affect multiple modules or systems
- User needs to understand trade-offs before proceeding

**Before switching modes (recommended):**
- If the user intent, constraints, or the approach/design is still unclear, use `design_brainstorming` first to clarify requirements and explore 2-3 approaches with trade-offs.
- After the design is clear enough, proceed with Plan mode (for large/complex) or Agent mode (for small/medium).

**Suggest to user:**
> "This feature involves [complexity factors]. I recommend switching to Plan mode to analyze the requirements and discuss implementation approaches before coding. Would you like to switch?"

**In Plan mode, discuss:**

1. **Requirements clarification**
   - What exactly should the feature do?
   - What are the expected inputs and outputs?
   - Are there edge cases to consider?

2. **Impact analysis**
   - Which files/modules will be affected?
   - Are there dependencies to consider?
   - Will this require database/API changes?

3. **Implementation approach**
   - What are the possible approaches?
   - What are the trade-offs of each?
   - Which approach best fits the project?

4. **Test strategy**
   - What tests are needed?
   - How will we verify the feature works?

#### For Small/Medium Changes - Direct Implementation

For straightforward changes, proceed directly in Agent mode:

1. Confirm understanding of the requirement
2. Identify files to modify
3. Make targeted changes
4. Verify changes work as expected

### Step 3: Implementation Phase

During implementation, follow these practices:

#### Code Change Guidelines

| Practice | Description |
|----------|-------------|
| **Incremental changes** | Make small, focused changes |
| **Read before edit** | Always read existing code before modifying |
| **Maintain style** | Follow existing code conventions |
| **Add comments** | Explain complex logic (in English only) |
| **Handle errors** | Add appropriate error handling |
| **English comments** | All comments and docstrings must be in English |

#### Implementation Workflow

1. **Read existing code** - Understand current implementation
2. **Plan modifications** - Identify exact changes needed
3. **Make changes** - Edit files incrementally
4. **Check for errors** - Verify no syntax/lint errors introduced
5. **Review changes** - Ensure changes match requirements

#### Long-Running Development

For features requiring extended development time:

**Branch Synchronization:**
- Periodically sync with main branch to avoid merge conflicts
- Recommended: sync at least once per day for active development

```bash
# Sync commands
git fetch origin
git rebase origin/main
# or
git merge origin/main
```

**Incremental Commits:**
- Make WIP (Work In Progress) commits for significant milestones
- Use descriptive commit messages even for intermediate states
- Example: `feat(auth): WIP - add login form structure`

### Step 4: Testing Phase

After completing implementation, guide user through testing:

#### Automated Tests

If the project has existing tests, run them before manual testing:

| Test Type | Command Example | When to Run |
|-----------|-----------------|-------------|
| Unit tests | `pytest`, `npm test` | After implementation |
| Lint check | `ruff check`, `eslint` | Before commit |

**Note:** Fix any failing tests before proceeding to manual testing.

#### Manual Testing Checklist

**Ask user to test:**

> "Implementation is complete. Please test the following:
> 
> 1. **Basic functionality** - Does the feature work as expected?
> 2. **Edge cases** - How does it handle unusual inputs?
> 3. **Error handling** - Does it fail gracefully?
> 4. **Integration** - Does it work with existing features?
> 5. **Regression** - Are existing features still working?"

#### If Tests Fail - Root Cause Analysis

When issues are found, perform systematic debugging:

1. **Reproduce the issue**
   - What are the exact steps to reproduce?
   - What is the expected vs actual behavior?

2. **Gather evidence**
   - Check error messages/logs
   - Review related code
   - Identify the failing component

3. **Analyze root cause**
   - Where does the issue originate?
   - Is it a logic error, data issue, or integration problem?
   - What assumptions were incorrect?

4. **Fix and verify**
   - Make targeted fixes
   - Re-test the specific issue
   - Verify no new issues introduced

5. **Document findings**
   - What was the issue?
   - What was the fix?
   - Any lessons learned?

#### Test Results Summary

| Result | Action |
|--------|--------|
| All tests pass | Proceed to Step 4.5 |
| Issues found | Debug, fix, and re-test |
| Requirements unclear | Return to Plan mode for clarification |
| Design flaw discovered | Discuss redesign approach |

### Step 4.5: Code Self-Review

Before committing, perform a self-review of changes:

#### Review Checklist

| Aspect | Check |
|--------|-------|
| **Logic** | Does the code do what it's supposed to? |
| **Edge cases** | Are boundary conditions handled? |
| **Error handling** | Are errors caught and handled gracefully? |
| **Security** | No hardcoded secrets? Input validated? |
| **Performance** | No obvious inefficiencies? |
| **Readability** | Is the code clear and well-commented? |
| **Dead code** | No unused imports, variables, or debug code? |

#### Agent-Assisted Review

Request agent to review:
> "Please review the changes I made and check for any issues."

Agent will check:
- Code style consistency
- Potential bugs or logic errors
- Security concerns
- Best practices violations

### Step 5: Documentation and Commit

When testing and self-review are complete:

#### Documentation Check

If the feature affects user-facing functionality, consider updating:

| Document | When to Update |
|----------|----------------|
| README | New setup steps, usage changes |
| API docs | Endpoint changes, new parameters |
| CHANGELOG | All user-visible changes |
| Code comments | Complex logic explanations |

**Ask user:**
> "Does this feature require documentation updates? (README, API docs, CHANGELOG, etc.)"

#### Pre-Commit Verification

Verify the feature is ready for commit:

- [ ] All planned functionality implemented
- [ ] Feature tested and working correctly
- [ ] No known bugs or issues
- [ ] Code follows project conventions
- [ ] No debugging code left behind
- [ ] No sensitive data exposed

#### Prompt User for Commit

**Ask user:**

> "The feature implementation is complete and verified. Would you like to commit your changes?
> 
> Changes summary:
> - [List of modified files]
> - [Brief description of changes]
> 
> If ready, I can help you commit using the `git_pushing_changes` workflow."

## Quick Reference

### Mode Selection Guide

```
Is the change straightforward with clear requirements?
├── Yes → Agent mode (direct implementation)
└── No → Consider Plan mode
         ├── Multiple approaches? → Plan mode
         ├── Unclear requirements? → Plan mode
         ├── Architectural impact? → Plan mode
         └── Cross-cutting concerns? → Plan mode
```

### Complete Workflow Summary

| Phase | Key Actions | Mode |
|-------|-------------|------|
| **Assess** | Evaluate complexity | Agent |
| **Plan** | Analyze requirements (if needed) | Plan |
| **Implement** | Make code changes (sync branch if long-running) | Agent |
| **Test** | Run automated tests + manual verification | Agent |
| **Debug** | Fix issues (if any) | Agent |
| **Self-Review** | Code quality check before commit | Agent |
| **Document** | Update docs if needed | Agent |
| **Commit** | Push changes to branch | Agent |

## Examples

### Example 1: Small Change (Direct Implementation)

**Context:** Add a simple utility function

**Workflow:**
1. User requests: "Add a function to format dates"
2. Agent assesses: Small change, single file, clear requirement
3. Agent implements directly in Agent mode
4. Agent asks user to test
5. User confirms working
6. Agent offers to commit

### Example 2: Large Feature (Plan Mode)

**Context:** Implement user notification system

**Workflow:**
1. User requests: "Add a notification system"
2. Agent assesses: Multiple components, architectural decisions needed
3. Agent suggests: "This feature has significant scope. I recommend switching to Plan mode to discuss the approach. Should we switch?"
4. In Plan mode:
   - Discuss notification types
   - Decide on storage approach
   - Plan API endpoints
   - Define data models
5. Return to Agent mode for implementation
6. Implement in phases
7. Test each phase
8. Final integration testing
9. Commit when complete

### Example 3: Issue Found During Testing

**Context:** Feature works but edge case fails

**Workflow:**
1. Implementation complete
2. User tests, reports: "Fails when input is empty"
3. Agent analyzes:
   - Reproduce: Empty input causes crash
   - Root cause: Missing null check
   - Fix: Add validation
4. Agent makes fix
5. User re-tests: Works correctly
6. Proceed to commit

## Integration with Other Skills

| Skill | Relationship |
|-------|--------------|
| `design_brainstorming` | Used BEFORE this skill when requirements/approach are unclear |
| `git_submitting_issues` | Used BEFORE this skill to create feature issue |
| `git_creating_branch` | Used BEFORE this skill to create dev branch |
| `git_pushing_changes` | Used AFTER this skill to commit changes |
| `git_creating_pr` | Used AFTER committing to create pull request |

## Communication Patterns

### Suggesting Plan Mode

```
"This feature involves [specific complexity factors]. To ensure we implement 
it correctly, I recommend switching to Plan mode to:
- Clarify requirements
- Discuss implementation approaches  
- Identify potential challenges

Would you like to switch to Plan mode for planning?"
```

### After Implementation

```
"I've completed the implementation. Here's a summary of changes:

[List changes]

Please test the following scenarios:
1. [Test case 1]
2. [Test case 2]
3. [Edge case]

Let me know the results, and we'll address any issues together."
```

### Ready to Commit

```
"All tests pass and the feature is verified working. Would you like to:
1. Commit these changes to your branch?
2. Make additional modifications?
3. Review the changes first?
```
