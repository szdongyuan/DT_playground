---
name: git_creating_pr
description: Create pull requests from personal branches to target branches. Use when submitting code for review, merging features, or when the user asks to create a PR/pull request.
---

# Creating Pull Request

## Overview

Standard workflow for creating pull requests from personal development branches. Ensures all changes are pushed, PR information is complete, and follows consistent format for titles and descriptions.

## Language Requirement

**IMPORTANT: All PR content MUST be written entirely in English.**

This includes:
- PR title
- PR description (summary, changes, test plan, etc.)
- Comments and review responses

Rationale:
- Ensures consistent documentation
- Makes PRs accessible to all contributors
- Aligns with common open-source practices

## Workflow

```
1. Verify branch → 2. Ensure pushed → 3. Prepare PR info → 4. Create PR → 5. Verify
```

## Prerequisites

Before creating a PR:

- [ ] All changes committed locally
- [ ] Branch pushed to remote
- [ ] Not on protected branch (main/master/develop)
- [ ] GitHub CLI (`gh`) installed (recommended)

## Instructions

### Step 1: Verify Current Branch

**CRITICAL**: Ensure you are on a feature/bugfix branch, not a protected branch.

```bash
# Get current branch name
git branch --show-current
```

#### Protected Branches (PR targets, not sources)

| Branch | Purpose |
|--------|---------|
| `main` | Production code |
| `master` | Production code (legacy) |
| `develop` | Integration branch |
| `release/*` | Release candidates |

**If on protected branch:** You should NOT create a PR from here. Create a feature branch first using `git_creating_branch` skill.

### Step 2: Ensure All Changes Are Pushed

Before creating PR, verify all local commits are pushed to remote:

```bash
# Check for unpushed commits
git status

# If needed, push changes
git push -u origin HEAD
```

### Step 3: Prepare PR Information

#### PR Title Format

```
{type}({scope}): {short description}
```

| Type | Use For | Example |
|------|---------|---------|
| `feat` | New feature | `feat(auth): add user login` |
| `fix` | Bug fix | `fix(api): handle null response` |
| `docs` | Documentation | `docs(readme): update install guide` |
| `refactor` | Code refactoring | `refactor(utils): extract helpers` |
| `test` | Test changes | `test(auth): add unit tests` |
| `chore` | Build/config | `chore(deps): update dependencies` |

#### PR Description Template

```markdown
## Summary

[1-3 bullet points describing the changes]

- Added feature X
- Fixed issue Y
- Updated documentation for Z

## Related Issue

[Link to related issue if applicable]

Closes #123
Fixes #456

## Changes

[List of specific changes made]

- `file1.py`: Added new validation logic
- `file2.py`: Fixed error handling
- `tests/`: Added unit tests for new feature

## Test Plan

[How the changes were tested]

- [ ] Unit tests pass
- [ ] Manual testing completed
- [ ] Integration tests pass

## Screenshots

[If applicable, add screenshots or GIFs]

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Step 4: Create Pull Request

#### Method 1: Using GitHub CLI (Recommended)

```bash
# Basic PR creation (opens editor for description)
gh pr create --base main --title "feat(module): add new feature"

# PR with inline body
gh pr create --base main --title "feat(module): add feature" --body "$(cat <<'EOF'
## Summary

- Added new feature X
- Updated tests

## Test Plan

- [x] Unit tests pass
EOF
)"

# Interactive mode (prompts for all fields)
gh pr create

# Create as draft PR
gh pr create --draft --base main --title "WIP: feat(module): new feature"

# Assign reviewers
gh pr create --base main --title "feat(module): feature" --reviewer @username

# Add labels
gh pr create --base main --title "feat(module): feature" --label "enhancement"
```

#### Method 2: Using Git Commands

```bash
# Push branch first
git push -u origin HEAD

# Get the URL to create PR manually
git remote get-url origin
# Then visit: https://github.com/{owner}/{repo}/pull/new/{branch-name}
```

#### Common gh pr create Options

| Option | Description | Example |
|--------|-------------|---------|
| `--base` | Target branch | `--base main` |
| `--title` | PR title | `--title "feat: add login"` |
| `--body` | PR description | `--body "Description here"` |
| `--draft` | Create as draft | `--draft` |
| `--reviewer` | Add reviewers | `--reviewer @user1,@user2` |
| `--assignee` | Assign to users | `--assignee @me` |
| `--label` | Add labels | `--label "bug,priority:high"` |
| `--milestone` | Set milestone | `--milestone "v2.0"` |
| `--web` | Open in browser | `--web` |

### Step 5: Verify PR Creation

After creating the PR:

```bash
# View the PR in terminal
gh pr view

# Open PR in browser
gh pr view --web

# List your open PRs
gh pr list --author @me
```

## Quick Reference

### Complete Workflow Commands

```bash
# 1. Verify branch
git branch --show-current

# 2. Check status and push if needed
git status
git push -u origin HEAD

# 3. Create PR with gh CLI
gh pr create --base main --title "type(scope): description" --body "PR description"

# 4. Verify
gh pr view
```

### Minimal PR Creation

```bash
# Quick PR with interactive prompts
gh pr create
```

### Draft PR for Work in Progress

```bash
gh pr create --draft --base main --title "WIP: feat(module): work in progress"
```

## Examples

### Example 1: Feature PR with Full Description

**Context:** Completed user authentication feature on branch `feature/FEAT2026012801_user_authentication`

**Commands:**
```bash
# Verify branch
git branch --show-current
# Output: feature/FEAT2026012801_user_authentication

# Ensure pushed
git push -u origin HEAD

# Create PR
gh pr create --base main --title "feat(auth): implement user authentication" --body "$(cat <<'EOF'
## Summary

- Added JWT-based user authentication
- Implemented login and logout endpoints
- Added password hashing with bcrypt

## Related Issue

Closes #FEAT2026012801

## Changes

- `src/auth/`: New authentication module
- `src/routes/auth.py`: Login/logout endpoints
- `tests/test_auth.py`: Unit tests for auth

## Test Plan

- [x] Unit tests pass
- [x] Manual testing with Postman
- [x] Integration tests pass

## Checklist

- [x] Code follows project style guidelines
- [x] Self-review completed
- [x] Documentation updated
- [x] Tests added
EOF
)"

# View created PR
gh pr view --web
```

### Example 2: Quick Bugfix PR

**Context:** Fixed validation bug on branch `bugfix/BUG2026012802_login_validation`

**Commands:**
```bash
# Verify and push
git branch --show-current
git push

# Create simple PR
gh pr create --base main \
  --title "fix(auth): correct email validation regex" \
  --body "Fixes edge case for subdomains and plus addressing. Closes #BUG2026012802"

# View PR
gh pr view
```

### Example 3: Draft PR for Early Feedback

**Context:** Work in progress, want early review

**Commands:**
```bash
# Create draft PR
gh pr create --draft \
  --base main \
  --title "WIP: feat(dashboard): new analytics view" \
  --body "Early draft for feedback. Not ready for merge."

# Later, mark as ready for review
gh pr ready
```

### Example 4: PR with Reviewers and Labels

**Context:** Feature ready for specific team members to review

**Commands:**
```bash
gh pr create --base main \
  --title "feat(api): add rate limiting" \
  --reviewer @senior-dev,@team-lead \
  --assignee @me \
  --label "enhancement,needs-review" \
  --body "Implemented rate limiting for API endpoints."
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Non-English PR content | Write title, description, and comments in English only |
| Creating PR from main/master | Always work on feature branches |
| Unpushed local commits | Run `git push` before creating PR |
| Vague PR title | Use `type(scope): description` format |
| No description | Always include summary and test plan |
| Missing issue link | Use `Closes #123` or `Fixes #456` |
| Forgot to assign reviewers | Use `--reviewer` flag or add via web |

## Verification Checklist

Before submitting PR:

- [ ] On correct source branch (not main/master)
- [ ] All commits pushed to remote
- [ ] PR title follows `type(scope): description` format
- [ ] PR description includes summary
- [ ] Related issue linked (if applicable)
- [ ] Tests pass locally
- [ ] Reviewers assigned (if required)

## Integration with Other Skills

| Skill | When to Use |
|-------|-------------|
| `git_creating_branch` | Before starting work, to create feature branch |
| `git_pushing_changes` | Before creating PR, to push all changes |
| `git_submitting_issues` | To create issue before starting work |

## Additional Commands

### Managing Existing PRs

```bash
# List all open PRs
gh pr list

# Check PR status
gh pr status

# View specific PR
gh pr view 123

# Add comment to PR
gh pr comment 123 --body "Comment text"

# Merge PR
gh pr merge 123

# Close PR without merging
gh pr close 123

# Reopen closed PR
gh pr reopen 123
```

### Update PR After Feedback

```bash
# Make changes locally, commit and push
git add .
git commit -m "fix: address review comments"
git push

# PR automatically updates with new commits
```
