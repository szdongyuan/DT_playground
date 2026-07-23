# Skills Directory

## Required Preflight

Every task in this repository must start by reading this file and then reading the matching skill files under `.codex/skills/*/SKILL.md`.

This requirement applies to all tasks without exception, including GitHub issues, pull requests, branches, pushes, testing, bug fixes, feature implementation, planning, documentation, shell commands, refactoring, project maintenance, and exploratory code analysis.

Repository-local skills in `.codex/skills` take priority over generic tools, global skills, and plugin workflows when a local skill covers the requested task.

## Local Planning Artifact Policy

- Files under `docs/plans/` are local working artifacts by default.
- Do not stage, commit, push, or include `docs/plans/` in a pull request unless the user explicitly asks to publish a specific plan.
- Planning and test-design skills may still create or update these files locally.
- Before staging changes, explicitly verify that `docs/plans/` is excluded from the commit scope.

Minimum routing rules:

| Task | Required local skill files |
|------|----------------------------|
| GitHub issue creation or issue updates | `git_submitting_issues/SKILL.md`, plus `shell_powershell_commands/SKILL.md` when using `gh` |
| Branch creation | `git_creating_branch/SKILL.md`, plus `shell_powershell_commands/SKILL.md` when using GitHub/network commands |
| Commit or push workflow | `git_pushing_changes/SKILL.md`, plus `shell_powershell_commands/SKILL.md` when using GitHub/network commands |
| Pull request creation | `git_creating_pr/SKILL.md`, plus `shell_powershell_commands/SKILL.md` when using `gh` |
| Bug fixing | `code_fixing_bugs/SKILL.md` |
| Feature implementation | `code_implementing_features/SKILL.md` |
| Test design | `test_designing/SKILL.md` |
| Test writing | `test_writing/SKILL.md` |
| Test execution | `test_running/SKILL.md` |
| PowerShell or Windows shell work | `shell_powershell_commands/SKILL.md` |
| Skill creation or organization | `skill_creating_skills/SKILL.md` |

If multiple skills match a task, read all relevant skill files before taking action. If a local skill conflicts with a generic plugin workflow, follow the local skill unless the user explicitly requests a different process.

> **Purpose**: This directory contains reusable skill definitions that the AI agent can reference when performing specific tasks.

## What is a Skill?

A **Skill** is a documented set of techniques, patterns, or step-by-step instructions for handling specific types of tasks. Skills are more granular than workflows and focus on "how to do something" rather than "what process to follow".

## Rules vs Skills

- Rules in `.cursor/rules/` describe durable project facts, constraints, architecture, and standards.
- Skills in `.codex/skills/` describe task procedures: how to create branches, manage i18n, run tests, write tests, submit issues, or perform debugging.
- If guidance is mostly a repeatable step-by-step workflow, keep it in a skill and link to it from the relevant rule.

## Skill Document Format

Each skill document should follow this structure:

```markdown
# Skill Name

## Overview
Brief description of what this skill enables.

## When to Use
- Condition 1
- Condition 2

## Prerequisites
- Required knowledge or tools

## Steps / Techniques

### Technique 1: Name
Description and examples...

### Technique 2: Name
Description and examples...

## Examples
Concrete examples demonstrating the skill.

## Common Pitfalls
- Mistake 1 and how to avoid it
- Mistake 2 and how to avoid it

## Related Skills
- Link to related skill documents
```

## Available Skills

| Skill | Description | File |
|-------|-------------|------|
| code-fixing-bugs | Systematic debugging and root cause analysis | [code_fixing_bugs/SKILL.md](code_fixing_bugs/SKILL.md) |
| code-implementing-features | Feature/documentation implementation workflow | [code_implementing_features/SKILL.md](code_implementing_features/SKILL.md) |
| data-preparing-labels | Prepare supervised classification labels for audio datasets | [data_preparing_labels/SKILL.md](data_preparing_labels/SKILL.md) |
| design-brainstorming | Clarify vague requirements before implementation | [design_brainstorming/SKILL.md](design_brainstorming/SKILL.md) |
| git-creating-branch | Create local personal/task branches | [git_creating_branch/SKILL.md](git_creating_branch/SKILL.md) |
| git-creating-pr | Create pull requests | [git_creating_pr/SKILL.md](git_creating_pr/SKILL.md) |
| git-pushing-changes | Commit and push changes safely | [git_pushing_changes/SKILL.md](git_pushing_changes/SKILL.md) |
| git-submitting-issues | Create structured GitHub issues | [git_submitting_issues/SKILL.md](git_submitting_issues/SKILL.md) |
| project-managing-i18n | Manage gettext catalog updates | [project_managing_i18n/SKILL.md](project_managing_i18n/SKILL.md) |
| project-python-init | Initialize Python project scaffolding | [project_python_init/SKILL.md](project_python_init/SKILL.md) |
| shell-powershell-commands | Run PowerShell commands on Windows | [shell_powershell_commands/SKILL.md](shell_powershell_commands/SKILL.md) |
| skill-creating-skills | Create or organize project skills | [skill_creating_skills/SKILL.md](skill_creating_skills/SKILL.md) |
| test-designing | Design test cases and coverage scenarios | [test_designing/SKILL.md](test_designing/SKILL.md) |
| test-running | Run pytest and analyze test results | [test_running/SKILL.md](test_running/SKILL.md) |
| test-writing | Write pytest/unit tests | [test_writing/SKILL.md](test_writing/SKILL.md) |

## Adding New Skills

1. Create a folder named with the project convention, e.g. `category-action-target/`.
2. Add `SKILL.md` with YAML frontmatter and English instructions.
3. Keep procedural details in the skill; keep stable project constraints in rules.
4. Update this README to include the new skill in the table.

---

*This directory is designed for extensibility. Add skills as patterns emerge from repeated tasks.*
