---
name: git_pushing_changes
description: Commit and push changes to remote personal branch. Use when committing code changes, pushing to remote, or syncing local work to personal branches. Prevents accidental pushes to main/master.
---

# Pushing Changes to Personal Branch

## Overview

Safe workflow for committing and pushing changes to personal development branches. Includes protection against accidental pushes to main/master branches.

## Language Requirement

**IMPORTANT: All commit messages MUST be written entirely in English.**

This includes:
- Commit type, scope, and description
- Commit body and additional details

Rationale:
- Ensures consistent git history
- Makes logs searchable and accessible to all contributors
- Aligns with common open-source practices

## Workflow

```
1. Verify branch → 2. Review changes → 3. Stage files → 4. Commit → 5. Push
```

## Instructions

### Step 1: Verify Current Branch

**CRITICAL**: Before any commit/push operation, verify you are NOT on a protected branch.

```bash
# Get current branch name
git branch --show-current
```

#### Protected Branches (DO NOT push directly)

| Branch | Purpose |
|--------|---------|
| `main` | Production code |
| `master` | Production code (legacy) |
| `develop` | Integration branch |
| `release/*` | Release candidates |

**If on protected branch:** Create a personal branch first using `git_creating_branch` skill.

### Step 2: Review Changes

Before committing, review what will be included:

```bash
# View all changes (staged and unstaged)
git status

# View detailed diff of unstaged changes
git diff

# View diff of already staged changes
git diff --cached
```

### Step 3: Stage Files

Choose appropriate staging approach:

| Scenario | Command |
|----------|---------|
| Stage all changes | `git add .` |
| Stage specific files | `git add path/to/file1 path/to/file2` |
| Stage by pattern | `git add "*.py"` |
| Interactive staging | `git add -p` (for partial file staging) |

**Warning:** Avoid staging sensitive files:
- `.env` files
- Credentials/secrets
- Local configuration

### Step 4: Create Commit

#### Commit Message Format

```
{type}({scope}): {description}

{body}
```

#### Commit Types

| Type | Use For | Example |
|------|---------|---------|
| `feat` | New feature | `feat(auth): add login validation` |
| `fix` | Bug fix | `fix(api): handle null response` |
| `docs` | Documentation | `docs(readme): update install steps` |
| `refactor` | Code refactoring | `refactor(utils): extract helper` |
| `test` | Test changes | `test(auth): add unit tests` |
| `chore` | Build/config changes | `chore(deps): update packages` |

#### Commit Command

```bash
git commit -m "type(scope): description"
```

**For multi-line messages:**
```bash
git commit -m "type(scope): short description" -m "Detailed body explaining the change"
```

### Step 5: Push to Remote

#### First Push (New Branch)

When pushing a new branch for the first time:

```bash
# Push and set upstream tracking
git push -u origin HEAD
```

#### Subsequent Pushes

For branches that already track remote:

```bash
git push
```

#### Force Push (Use with Caution)

Only use when you've rebased or amended commits:

```bash
# Safe force push (fails if remote has new commits)
git push --force-with-lease
```

**Warning:** Never force push to shared branches.

## Safety Checks

### Pre-Push Verification Script

Run these checks before pushing:

```bash
# 1. Verify not on protected branch
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" == "main" || "$BRANCH" == "master" || "$BRANCH" == "develop" ]]; then
    echo "ERROR: Cannot push directly to $BRANCH"
    exit 1
fi

# 2. Verify branch has remote tracking or will create one
git push -u origin HEAD
```

### Verification Checklist

Before each push, verify:

- [ ] Not on main/master/develop branch
- [ ] All intended changes are staged
- [ ] No sensitive files included
- [ ] Commit message follows format
- [ ] Tests pass locally (if applicable)

## Quick Reference

### Complete Workflow Commands

```bash
# 1. Check current branch
git branch --show-current

# 2. Review changes
git status
git diff

# 3. Stage changes
git add .

# 4. Commit
git commit -m "feat(module): add new feature"

# 5. Push (first time)
git push -u origin HEAD

# 5. Push (subsequent)
git push
```

### Abort/Undo Commands

| Situation | Command |
|-----------|---------|
| Unstage all files | `git reset HEAD` |
| Unstage specific file | `git reset HEAD path/to/file` |
| Discard unstaged changes | `git checkout -- path/to/file` |
| Amend last commit | `git commit --amend` |
| Undo last commit (keep changes) | `git reset --soft HEAD~1` |

## Examples

### Example 1: Standard Feature Commit

**Context:** Completed work on user authentication feature

**Commands:**
```bash
# Verify branch
git branch --show-current
# Output: feature/FEAT2026012801_user_authentication

# Review and stage
git status
git add .

# Commit
git commit -m "feat(auth): implement user login flow

Add login endpoint with JWT token validation.
Include password hashing and session management."

# Push
git push -u origin HEAD
```

### Example 2: Bugfix with Specific Files

**Context:** Fixed a validation error, only want to commit specific files

**Commands:**
```bash
# Verify branch
git branch --show-current
# Output: bugfix/BUG2026012802_login_validation

# Stage specific files
git add src/validators/login.py tests/test_login.py

# Commit
git commit -m "fix(auth): correct email validation regex

Handle edge cases for subdomains and plus addressing."

# Push
git push
```

### Example 3: Accidental Main Branch Detection

**Context:** Developer realizes they're on main branch

**Commands:**
```bash
# Check branch
git branch --show-current
# Output: main

# STOP! Create personal branch first
git fetch origin
git checkout -b feature/my_new_feature origin/main

# Now safe to commit and push
git add .
git commit -m "feat(module): add feature"
git push -u origin HEAD
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Non-English commit message | Write all commit messages in English only |
| Pushing to main/master | Always verify branch first with `git branch --show-current` |
| Forgot to stage changes | Use `git status` to verify staged files before commit |
| Committed sensitive files | Add to `.gitignore`, use `git reset` to unstage |
| Push rejected (no tracking) | Use `git push -u origin HEAD` for first push |
| Vague commit messages | Follow `type(scope): description` format |

## Integration with Other Skills

| Skill | When to Use |
|-------|-------------|
| `git_creating_branch` | Before starting work, to create personal branch |
| `git_submitting_issues` | When the work is related to an issue |
