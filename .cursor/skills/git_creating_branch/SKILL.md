---
name: git_creating_branch
description: Create local personal branches with latest code from remote. Use when starting new feature work, bug fixes, or any development that needs a fresh branch, optionally linked to issues.
---

# Creating Local Personal Branch

## Overview

Standard workflow for creating local development branches. Ensures branches are created from the latest remote code and follow consistent naming conventions, with optional issue linking.

## Workflow

```
1. Fetch latest → 2. Identify base → 3. Determine branch name → 4. Create & switch
```

## Instructions

### Step 1: Fetch Latest Code

Before creating a branch, always fetch the latest code from remote:

```bash
git fetch origin
```

### Step 2: Identify Base Branch

Determine which branch to base your new branch on:

| Scenario | Base Branch |
|----------|-------------|
| New feature | `origin/develop` |
| Hotfix | `origin/develop` or release branch |
| Bugfix for specific version | Corresponding release branch |

### Step 3: Determine Branch Name

#### Branch Naming Format

```
{type}/{issue_id?}_{short_description}
```

#### Branch Types

| Type | Use For | Example |
|------|---------|---------|
| `feature` | New functionality | `feature/add_user_auth` |
| `bugfix` | Bug fixes | `bugfix/fix_login_error` |
| `hotfix` | Urgent production fixes | `hotfix/critical_db_issue` |
| `refactor` | Code refactoring | `refactor/extract_validation` |
| `docs` | Documentation updates | `docs/update_readme` |
| `test` | Test additions/changes | `test/add_unit_tests` |

#### With Issue Linking

When the work is related to an issue, include the issue ID in the branch name:

**Format:**
```
{type}/{issue_id}_{short_description}
```

**Examples:**
```
feature/FEAT2026012801_user_authentication
bugfix/BUG2026012802_login_validation_error
refactor/REFACTOR2026012803_extract_common_utils
```

#### Without Issue Linking

When there's no related issue:

**Examples:**
```
feature/add_export_function
bugfix/fix_null_pointer
refactor/optimize_query_performance
```

### Step 4: Create and Switch to Branch

Create the new branch from the latest remote code:

```bash
# Create branch from origin/develop
git checkout -b {branch_name} origin/develop
```

**Complete example with issue:**
```bash
git fetch origin
git checkout -b feature/FEAT2026012801_user_authentication origin/develop
```

**Complete example without issue:**
```bash
git fetch origin
git checkout -b feature/add_export_function origin/develop
```

## Quick Reference

### Commands Summary

```bash
# 1. Fetch latest
git fetch origin

# 2. Create branch (with issue)
git checkout -b feature/ISSUE123_description origin/develop

# 3. Create branch (without issue)
git checkout -b feature/description origin/develop

# 4. Verify branch
git branch --show-current
```

### Branch Name Rules

| Rule | Requirement |
|------|-------------|
| Characters | Lowercase letters, numbers, underscores, hyphens |
| Separator | Use `_` between issue ID and description |
| Path separator | Use `/` after branch type |
| Max length | Keep reasonably short (< 50 chars recommended) |
| No spaces | Replace spaces with underscores |

## Examples

### Example 1: Feature with Issue

**Context:** Creating a new user authentication feature linked to issue FEAT2026012801

**Commands:**
```bash
git fetch origin
git checkout -b feature/FEAT2026012801_user_authentication origin/develop
```

**Result:** Branch `feature/FEAT2026012801_user_authentication` created from latest `origin/develop`

### Example 2: Bugfix with Issue

**Context:** Fixing login validation error linked to issue BUG2026012802

**Commands:**
```bash
git fetch origin
git checkout -b bugfix/BUG2026012802_login_validation_error origin/develop
```

**Result:** Branch `bugfix/BUG2026012802_login_validation_error` created from latest `origin/develop`

### Example 3: Feature without Issue

**Context:** Adding a small utility function, no issue created

**Commands:**
```bash
git fetch origin
git checkout -b feature/add_string_utils origin/develop
```

**Result:** Branch `feature/add_string_utils` created from latest `origin/develop`

### Example 4: Hotfix from Release Branch

**Context:** Critical fix for production, branch from release

**Commands:**
```bash
git fetch origin
git checkout -b hotfix/BUG2026012803_fix_payment_crash origin/release/v2.0
```

**Result:** Branch `hotfix/BUG2026012803_fix_payment_crash` created from release branch

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Not fetching before branching | Always run `git fetch origin` first |
| Branching from local develop (outdated) | Use `origin/develop` not `develop` |
| Spaces in branch name | Replace with underscores |
| Missing branch type prefix | Always include `feature/`, `bugfix/`, etc. |
| Issue ID without underscore separator | Use `TYPE/ID_description` not `TYPE/IDdescription` |

## Verification Checklist

After creating a branch, verify:

- [ ] Ran `git fetch origin` before creating branch
- [ ] Branch created from remote base (`origin/develop`)
- [ ] Branch name includes type prefix
- [ ] Issue ID included if applicable
- [ ] Branch name uses underscores, no spaces
- [ ] Currently on the new branch (`git branch --show-current`)
