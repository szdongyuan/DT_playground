---
name: git-submitting-issues
description: Create structured GitHub issues with clear scope, labels, assignment, and branch context. Use when creating issues, submitting bugs, requesting features, tracking tasks, filing refactors, or recording project work items.
---

# Submitting Issues

## Overview

Standard format for submitting issues on the current branch. Ensures consistent, actionable reports with proper categorization.

## Language Requirement

**IMPORTANT: All issues MUST be written entirely in English.**

This includes:
- Title
- Description
- All section content (stacktrace, repro steps, acceptance criteria, etc.)
- Comments and labels

Rationale:
- Ensures consistency across international teams
- Makes issues searchable and accessible to all contributors
- Aligns with common open-source practices

## Issue Types

| Type | Use For | Default Label |
|------|---------|---------------|
| BUG | Something broken | `bug` |
| FEAT | New functionality | `feature` |
| TASK | Non-code work (docs, config) | `task` |
| REFACTOR | Code improvement, no behavior change | `refactor` |

## Title Format

```
{TYPE}{DATE}{SEQ}, {short description} (key detail)
```

**Examples:**
```
BUG2026012601, cross-day saving throws FileNotFoundError (missing test_result_log)
FEAT2026012602, add batch export for test results (CSV format)
TASK2026012603, update deployment documentation (add rollback steps)
REFACTOR2026012604, extract validation logic (reduce duplication)
```

## Sequence Number Check (avoid duplicate SEQ)

Before creating a new issue, you MUST ensure `{TYPE}{DATE}{SEQ}` is unique for the day.

- **Rule**: `SEQ` is a 2-digit number (`01`, `02`, ...). Start from `01` each day per `TYPE`.
- **How**:
  - Search existing issues by the prefix `{TYPE}{DATE}` in the issue title.
  - Pick the next `SEQ` = (max existing `SEQ`) + 1.

### Quick manual check (GitHub UI)

- Search in Issues: `in:title BUG20260126` (replace `BUG` and date)
- Find the largest `SEQ` used, then use the next number.

### CLI check (GitHub CLI + PowerShell)

On Windows Codex Desktop, run the GitHub CLI environment hydration before any
`gh` command. Codex shell processes may not inherit persisted proxy or
`GH_TOKEN` values into the current process.

```powershell
$env:HTTP_PROXY  = [Environment]::GetEnvironmentVariable('HTTP_PROXY', 'User')  ?? [Environment]::GetEnvironmentVariable('HTTP_PROXY', 'Machine')
$env:HTTPS_PROXY = [Environment]::GetEnvironmentVariable('HTTPS_PROXY', 'User') ?? [Environment]::GetEnvironmentVariable('HTTPS_PROXY', 'Machine')
$env:ALL_PROXY   = [Environment]::GetEnvironmentVariable('ALL_PROXY', 'User')   ?? [Environment]::GetEnvironmentVariable('ALL_PROXY', 'Machine')
$env:GH_TOKEN    = [Environment]::GetEnvironmentVariable('GH_TOKEN', 'User')    ?? [Environment]::GetEnvironmentVariable('GH_TOKEN', 'Machine')
```

```powershell
# Example: find next SEQ for today's BUG issues
$TYPE = "BUG"
$DATE = (Get-Date -Format "yyyyMMdd")
$PREFIX = "$TYPE$DATE"

# Get all titles containing the prefix (open + closed)
$titles = gh issue list --state all --search $PREFIX --limit 200 --json title --jq '.[].title'

# Extract 2-digit SEQ from titles like "BUG2026012601, ..."
$seqs = $titles |
  ForEach-Object {
    if ($_ -match "^$PREFIX(?<seq>\d{2}),") { [int]$Matches.seq } else { $null }
  } |
  Where-Object { $_ -ne $null }

$max = if ($seqs.Count -gt 0) { ($seqs | Measure-Object -Maximum).Maximum } else { 0 }
$next = "{0:D2}" -f ($max + 1)

"Next SEQ for $PREFIX is $next"
```

## Required Sections by Type

| Section | BUG | FEAT | TASK | REFACTOR |
|---------|:---:|:----:|:----:|:--------:|
| Description | ✓ | ✓ | ✓ | ✓ |
| Stacktrace | ✓ | - | - | - |
| Repro | ✓ | - | - | - |
| Root Cause | ✓ | - | - | - |
| User Story | - | ✓ | - | - |
| Acceptance Criteria | - | ✓ | - | - |
| Scope | - | - | ✓ | ✓ |
| Suggested Fix / Approach | ✓ | ○ | ○ | ✓ |
| Related Code | ✓ | ○ | ○ | ✓ |

✓ = Required, ○ = Optional, - = Not applicable

## Body Templates

### BUG
```markdown
## Description
[What's broken]

## Stacktrace
[Error traceback]

## Repro
1. [Steps to reproduce]

## Root Cause
[Why it happens]

## Suggested Fix
[Proposed solution]

## Related Code
- `path/to/file.py:function_name`
```

### FEAT
```markdown
## Description
[What to build]

## User Story
As a [user], I want [goal] so that [benefit].

## Acceptance Criteria
- [ ] [Criterion 1]
- [ ] [Criterion 2]

## Related Code (optional)
- `path/to/file.py`
```

### TASK / REFACTOR
```markdown
## Description
[What needs to be done]

## Scope
- [Item 1]
- [Item 2]

## Approach (optional)
[How to approach]

## Related Code
- `path/to/file.py:function_name`
```

## Labels

| Label | When to Use |
|-------|-------------|
| `bug` | Something broken |
| `feature` | New functionality |
| `task` | Non-code work |
| `refactor` | Code improvement |
| `critical` | Blocks core functionality |
| `low-priority` | Nice to have |

## Assignment

- Assign to yourself if you will fix it
- Assign to code owner if known
- Leave unassigned if unsure

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Non-English content | Write everything in English only |
| Vague title | Include error/feature type in title |
| BUG without repro | Always include reproduction steps |
| FEAT without acceptance criteria | Define done state clearly |
| Wrong TYPE prefix | Match actual issue nature |
