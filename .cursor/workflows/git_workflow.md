# Git Workflow Rules for AI Agent

> **Purpose**: This document defines the standard Git workflow that the AI agent must follow when making changes to this project.

## Workflow Overview

```mermaid
flowchart TD
    A[User requests modification] --> B[Read git_workflow.md]
    B --> C["Step 1: Sync code (git fetch)"]
    C --> D{Uncommitted changes or conflicts?}
    D -->|Yes| E[Preserve user changes and discuss if blocking]
    D -->|No| F["Step 2: Create task branch"]
    F --> G["Step 3: Confirm change plan with user"]
    G --> H{User approved?}
    H -->|No| I[Adjust plan]
    I --> G
    H -->|Yes| J["Step 4: Implement changes"]
    J --> K{Ask: Run tests?}
    K -->|Yes| L[Run tests]
    K -->|No| M["Step 5: Commit changes"]
    L --> M
    M --> N{Ask: Push to remote?}
    N -->|Yes| O["Step 6: Push and create PR"]
    N -->|No| P[Keep changes local]
```

---

## Step-by-Step Process

### Step 1: Sync Code

Before making any changes, ensure the local repository is up to date.

```bash
# Fetch latest changes from remote
git fetch origin

# Check current status
git status

# Create new work from the remote base, usually origin/develop
```

**If uncommitted changes exist:**
- Do not discard or overwrite them.
- If they are unrelated and do not block the new branch, keep them intact.
- If they block branch creation or editing, ask the user how to proceed.

**If merge conflicts exist:**
- Notify the user and assist in resolving conflicts before continuing.

---

### Step 2: Create Task Branch

Create a new branch following the naming convention.

**Branch Naming Format:**
```
{type}/{issue_id?}_{short_description}
```

**Branch Types:**
| Type | Use For |
|------|---------|
| `feature` | New functionality |
| `bugfix` | Bug fixes |
| `hotfix` | Urgent fixes |
| `refactor` | Refactoring |
| `docs` | Documentation updates |
| `test` | Test additions or changes |

**Examples:**
- `docs/TASK2026052401_cursor_rules_refresh`
- `feature/FEAT2026050601_label_file_preview`
- `bugfix/BUG2026050901_training_callback_error`

**Commands:**
```bash
# Create and switch to new branch from the remote develop base
git checkout -b {branch_name} origin/develop
```

Use the `git_creating_branch` skill for the detailed branch creation procedure.

---

### Step 3: Confirm Change Plan

Before implementing any changes, the AI agent must:

1. **Analyze the request** - Understand what the user wants to achieve
2. **Create a plan when needed** - Use Cursor Plan Mode for broad, ambiguous, or cross-cutting changes
3. **Present the plan** - Show the user:
   - Files to be modified
   - Summary of changes
   - Potential impacts
4. **Get user approval** - Wait for explicit confirmation before proceeding

**Do NOT proceed with code changes until the user explicitly approves the plan.**

---

### Step 4: Implement Changes

Follow the project's coding standards defined in [coding_standards.md](../rules/coding_standards.md):

- Adhere to PEP 8 style guide
- Use type hints
- Write docstrings in English
- Follow the import order convention
- Respect the MVC + Controller architecture
- Keep procedural task details in skills and durable project constraints in rules

---

### Step 5: Test (If Requested)

Ask the user: **"Do you want me to run tests for these changes?"**

If yes:
```bash
# Run project tests
pytest tests/

# Or run specific test file
pytest tests/test_<module>.py
```

For desktop PySide6 UI changes, prefer automated offscreen Qt tests or manual app verification. Browser tools are not the primary verification path for this desktop application.

---

### Step 6: Commit Changes

Use **Conventional Commits** format for all commit messages.
Commit messages must be written in English.

**Format:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, no logic change) |
| `refactor` | Code refactoring |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |

**Examples:**
```bash
git add .
git commit -m "feat(workflow): add audio augmentation node"
git commit -m "fix(training): resolve memory leak in data generator"
git commit -m "docs(readme): update installation instructions"
```

---

### Step 7: Push and Create Pull Request

Ask the user: **"Do you want me to push these changes and create a Pull Request?"**

If yes:

**Push to remote:**
```bash
git push -u origin <branch>
```

**Create Pull Request:**
- Target branch: `develop`
- Use GitHub CLI (`gh pr create`) when creating PRs from the agent
- Use the PR template below

---

## Pull Request Template

When creating a PR, include the following information:

```markdown
## Summary
<!-- Brief description of what this PR does -->

## Changes Made
<!-- List of specific changes -->
- 
- 
- 

## Files Modified
<!-- List of files that were changed -->
- 
- 

## Testing
<!-- Describe testing performed -->
- [ ] Unit tests passed
- [ ] Manual testing completed
- [ ] No tests required (documentation only)

## Related Issues
<!-- Link any related issues -->
Closes #

## Additional Notes
<!-- Any other relevant information -->
```

---

## Security Checks

Before committing, verify that the changes do NOT include:

- [ ] API keys or secrets
- [ ] Passwords or credentials
- [ ] Personal access tokens
- [ ] Sensitive file paths
- [ ] Private user data

If any sensitive information is detected, alert the user immediately.

---

## Protected Branches

Do not commit or push directly to protected long-lived branches:

- `main`
- `master`
- `develop`
- `release/*`

Use a personal/task branch and open a PR instead.

---

## Conflict Resolution

If conflicts are detected during sync or merge:

1. **Notify the user** about the conflicting files
2. **Show the conflict details** if possible
3. **Ask for guidance** on how to resolve:
   - Accept incoming changes
   - Keep current changes
   - Manual merge required
4. **Assist with resolution** as directed by the user

---

## Rollback Procedure

If changes need to be reverted:

```bash
# Revert last commit (keep changes staged)
git reset --soft HEAD~1

# Revert a specific commit
git revert <commit-hash>
```

Never discard uncommitted changes, run `git checkout --`, or run `git reset --hard`
unless the user explicitly requests that exact destructive operation after being warned.

---

## Quick Reference

| Action | Command |
|--------|---------|
| Sync code | `git fetch origin` |
| Create branch | `git checkout -b docs/TASK2026052401_description origin/develop` |
| Stage all | `git add .` |
| Commit | `git commit -m "type(scope): message"` |
| Push | `git push -u origin <branch>` |
| Switch branch | `git checkout <branch>` |
| Check status | `git status` |
| View log | `git log --oneline -10` |

---

## Related Documents

- [rules.mdc](../rules/rules.mdc) - Project rules entry point
- [architecture.md](../rules/architecture.md) - Project architecture documentation
- [coding_standards.md](../rules/coding_standards.md) - Coding standards
- [model_json_spec.md](../rules/model_json_spec.md) - Model JSON specification
- [git_creating_branch](../skills/git_creating_branch/SKILL.md) - Branch creation procedure
- [git_pushing_changes](../skills/git_pushing_changes/SKILL.md) - Commit and push procedure
- [git_creating_pr](../skills/git_creating_pr/SKILL.md) - Pull request creation procedure
