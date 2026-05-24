# Skills Directory

> **Purpose**: This directory contains reusable skill definitions that the AI agent can reference when performing specific tasks.

## What is a Skill?

A **Skill** is a documented set of techniques, patterns, or step-by-step instructions for handling specific types of tasks. Skills are more granular than workflows and focus on "how to do something" rather than "what process to follow".

## Rules vs Skills

- Rules in `.cursor/rules/` describe durable project facts, constraints, architecture, and standards.
- Skills in `.cursor/skills/` describe task procedures: how to create branches, manage i18n, run tests, write tests, submit issues, or perform debugging.
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
| code_fixing_bugs | Systematic debugging and root cause analysis | [code_fixing_bugs/SKILL.md](code_fixing_bugs/SKILL.md) |
| code_implementing_features | Feature/documentation implementation workflow | [code_implementing_features/SKILL.md](code_implementing_features/SKILL.md) |
| data_preparing_labels | Prepare supervised classification labels for audio datasets | [data_preparing_labels/SKILL.md](data_preparing_labels/SKILL.md) |
| design_brainstorming | Clarify vague requirements before implementation | [design_brainstorming/SKILL.md](design_brainstorming/SKILL.md) |
| git_creating_branch | Create local personal/task branches | [git_creating_branch/SKILL.md](git_creating_branch/SKILL.md) |
| git_creating_pr | Create pull requests | [git_creating_pr/SKILL.md](git_creating_pr/SKILL.md) |
| git_pushing_changes | Commit and push changes safely | [git_pushing_changes/SKILL.md](git_pushing_changes/SKILL.md) |
| git_submitting_issues | Create structured GitHub issues | [git_submitting_issues/SKILL.md](git_submitting_issues/SKILL.md) |
| project_managing_i18n | Manage gettext catalog updates | [project_managing_i18n/SKILL.md](project_managing_i18n/SKILL.md) |
| project_python_init | Initialize Python project scaffolding | [project_python_init/SKILL.md](project_python_init/SKILL.md) |
| shell_powershell_commands | Run PowerShell commands on Windows | [shell_powershell_commands/SKILL.md](shell_powershell_commands/SKILL.md) |
| skill_creating_skills | Create or organize project skills | [skill_creating_skills/SKILL.md](skill_creating_skills/SKILL.md) |
| test_designing | Design test cases and coverage scenarios | [test_designing/SKILL.md](test_designing/SKILL.md) |
| test_running | Run pytest and analyze test results | [test_running/SKILL.md](test_running/SKILL.md) |
| test_writing | Write pytest/unit tests | [test_writing/SKILL.md](test_writing/SKILL.md) |

## Adding New Skills

1. Create a folder named with the project convention, e.g. `category_action_target/`.
2. Add `SKILL.md` with YAML frontmatter and English instructions.
3. Keep procedural details in the skill; keep stable project constraints in rules.
4. Update this README to include the new skill in the table.

---

*This directory is designed for extensibility. Add skills as patterns emerge from repeated tasks.*
