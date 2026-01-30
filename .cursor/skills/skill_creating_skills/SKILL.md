---
name: skill_creating_skills
description: Guide for creating new skills with standardized naming conventions and consistent structure. Use when creating, writing, or authoring a new skill, or when organizing skills by category.
---

# Creating Skills

This skill provides a standardized guide for creating new skills with consistent naming conventions, file structure, and content format.

## Naming Convention

### Format

```
{category}_{action}_{target}
```

- **category**: Domain or topic area (required prefix)
- **action**: What the skill does (verb or action noun)
- **target**: What the skill operates on (optional, for specificity)

### Rules

| Rule | Requirement |
|------|-------------|
| Language | **All skill content MUST be in English** |
| Characters | Lowercase letters, numbers, underscores only |
| Max length | 64 characters |
| Prefix | Category prefix is mandatory |
| Separator | Use underscores `_` between words |

### Category Prefixes

| Prefix | Domain | Examples |
|--------|--------|----------|
| `git_` | Git operations | `git_commit_messages`, `git_submitting_issues`, `git_branch_naming` |
| `code_` | Code operations | `code_review`, `code_formatting`, `code_refactoring` |
| `doc_` | Documentation | `doc_api_reference`, `doc_changelog`, `doc_readme` |
| `test_` | Testing | `test_unit_setup`, `test_integration`, `test_e2e` |
| `deploy_` | Deployment | `deploy_docker`, `deploy_kubernetes`, `deploy_ci_cd` |
| `db_` | Database | `db_migration`, `db_backup`, `db_schema_design` |
| `api_` | API related | `api_rest_design`, `api_graphql`, `api_authentication` |
| `util_` | Utilities | `util_file_processing`, `util_data_conversion` |
| `skill_` | Skill management | `skill_creating_skills`, `skill_organizing` |
| `project_` | Project setup | `project_structure`, `project_initialization` |
| `debug_` | Debugging | `debug_logging`, `debug_profiling` |
| `security_` | Security | `security_auth`, `security_encryption` |

### Naming Examples

**Good names:**
```
git_commit_messages       # Git + action + target
code_review               # Code + action
test_unit_setup           # Test + type + action
api_rest_design           # API + protocol + action
deploy_docker_compose     # Deploy + tool + detail
```

**Bad names:**
```
helper                    # No category prefix
my-skill                  # Uses hyphens instead of underscores
Code_Review               # Uses uppercase
gitcommitmessages         # No separators
```

## Language Requirement

**All skills MUST be written in English.** This ensures:

- Universal readability across teams
- Consistency in skill discovery
- Compatibility with AI agents and tooling

| Component | Requirement |
|-----------|-------------|
| Skill name | English, lowercase |
| Description | English only |
| Instructions | English only |
| Examples | English only |
| Comments | English only |
| Template content | English only |

## Directory Structure

```
.cursor/skills/{category}_{action}_{target}/
├── SKILL.md              # Required - main instructions
├── reference.md          # Optional - detailed documentation
├── examples.md           # Optional - usage examples
├── templates/            # Optional - template files
│   └── template.md
└── scripts/              # Optional - utility scripts
    └── helper.py
```

### Storage Locations

| Type | Path | Scope |
|------|------|-------|
| Project | `.cursor/skills/skill_name/` | Shared with repository |
| Personal | `~/.cursor/skills/skill_name/` | Available across all projects |

## SKILL.md Template

```markdown
---
name: category_action_target
description: Brief description of what this skill does. Use when [trigger scenarios].
---

# Skill Title

## Overview

[One paragraph describing the skill's purpose and scope]

## Instructions

[Step-by-step guidance for using this skill]

### Step 1: [Action]

[Details]

### Step 2: [Action]

[Details]

## Examples

### Example 1: [Scenario]

**Input:**
[Input description or content]

**Output:**
[Expected output]

### Example 2: [Scenario]

**Input:**
[Input description or content]

**Output:**
[Expected output]

## Reference

[Optional: Link to additional documentation]
- For detailed specifications, see [reference.md](reference.md)
- For more examples, see [examples.md](examples.md)
```

## Writing Effective Descriptions

The description is critical for skill discovery. The agent uses it to decide when to apply your skill.

### Description Format

```
{What the skill does}. Use when {trigger scenarios}.
```

### Description Rules

| Rule | Good | Bad |
|------|------|-----|
| **Use English** | "Generates commit messages" | "生成提交消息" |
| Third person | "Generates commit messages" | "I help generate commit messages" |
| Include triggers | "Use when committing changes" | (no trigger) |
| Be specific | "Review Python code for PEP8 compliance" | "Reviews code" |
| Include keywords | "git, commit, message, conventional" | (no keywords) |

### Description Examples

```yaml
# Git operations
description: Generate conventional commit messages from staged changes. Use when committing code or when the user asks for help with commit messages.

# Code review
description: Review code for quality, security, and maintainability. Use when reviewing pull requests, examining code changes, or when the user asks for a code review.

# Documentation
description: Generate API documentation from code comments. Use when creating or updating API docs, or when the user mentions documentation generation.

# Testing
description: Set up unit test scaffolding for new modules. Use when creating tests for new code or when the user asks about test setup.
```

## Content Authoring Principles

### 1. Concise is Key

The agent is already smart. Only add context it doesn't already have.

**Good:**
```markdown
## Generate commit message

Use conventional commits format:
- feat: new feature
- fix: bug fix
- docs: documentation
- refactor: code refactoring
```

**Bad:**
```markdown
## Generate commit message

Commit messages are important for tracking changes in a project.
They help team members understand what was changed and why.
There are many formats, but we recommend conventional commits...
```

### 2. Keep SKILL.md Under 500 Lines

For optimal performance, keep the main file concise. Use progressive disclosure for detailed content.

### 3. Progressive Disclosure

Put essential information in SKILL.md; detailed reference material in separate files.

```markdown
## Quick start
[Essential instructions]

## Additional resources
- For complete API details, see [reference.md](reference.md)
- For usage examples, see [examples.md](examples.md)
```

### 4. Use Tables for Structured Data

```markdown
| Type | Format | Example |
|------|--------|---------|
| feat | feat(scope): description | feat(auth): add login |
| fix | fix(scope): description | fix(api): handle null |
```

### 5. Provide Concrete Examples

```markdown
**Example:**
Input: Added user authentication
Output:
```
feat(auth): implement user authentication

Add login endpoint with JWT token validation
```
```

## Common Patterns

### Template Pattern

Provide output format templates when format matters:

```markdown
## Output format

Use this template:
\`\`\`
{type}({scope}): {description}

{body}
\`\`\`
```

### Workflow Pattern

Break complex operations into clear steps:

```markdown
## Workflow

1. Analyze the input
2. Determine the type
3. Generate output
4. Validate result
```

### Conditional Pattern

Guide through decision points:

```markdown
## Choose approach

**Creating new?** → Follow "Creation workflow"
**Updating existing?** → Follow "Update workflow"
```

## Checklist

Before finalizing a skill, verify:

### Naming
- [ ] Uses category prefix (e.g., `git_`, `code_`, `test_`)
- [ ] Uses underscores as separators
- [ ] All lowercase
- [ ] Under 64 characters
- [ ] Descriptive and clear

### Structure
- [ ] Has YAML frontmatter with `name` and `description`
- [ ] Description includes WHAT and WHEN
- [ ] Description is in third person
- [ ] SKILL.md is under 500 lines
- [ ] References are one level deep

### Language
- [ ] **All content is written in English**
- [ ] Name, description, and instructions use English
- [ ] Examples and comments use English
- [ ] No non-English text in skill files

### Content
- [ ] Instructions are clear and actionable
- [ ] Examples are concrete, not abstract
- [ ] Consistent terminology throughout
- [ ] No time-sensitive information
- [ ] No Windows-style paths (use forward slashes)

## Quick Reference

### Create a new skill

1. Choose category prefix from the table above
2. Create folder: `.cursor/skills/{category}_{action}_{target}/`
3. Create `SKILL.md` with frontmatter and content
4. Add optional reference files if needed
5. Verify with checklist

### Skill name examples by category

```
git_commit_messages
git_branch_naming
git_submitting_issues
git_pr_description

code_review
code_formatting
code_refactoring
code_documentation

test_unit_setup
test_integration
test_coverage

doc_api_reference
doc_readme
doc_changelog

deploy_docker
deploy_kubernetes
deploy_ci_cd

api_rest_design
api_error_handling
api_versioning
```
