---
name: project-python-init
description: Initialize Python projects by gathering requirements and generating project structure, configuration, and documentation. Use when creating a new Python project, initializing project files, setting up scaffolding, or defining initial architecture.
---

# Python Project Initialization

Guide users through the complete Python project initialization workflow, including requirements gathering, architecture design, configuration files and documentation generation.

## Language Requirement

**IMPORTANT: All project content MUST be written entirely in English.**

This includes:
- Code comments (inline, block, docstrings)
- Documentation files (README.md, ARCHITECTURE.md, CHANGELOG.md, docs/*)
- Configuration file comments
- License text
- Commit messages and PR descriptions

Rationale:
- Ensures project is accessible to international contributors
- Maintains consistency across all project files
- Aligns with common open-source practices

## Execution Flow

Execute the following 5 phases in sequence. Complete each phase before moving to the next.

---

## Phase 1: Basic Information Collection

Use AskQuestion tool to collect the following information:

```
Question 1: Project Name
- Enter project name (lowercase letters, numbers, underscores)

Question 2: Project Type
Options:
- web: Web Application
- cli: CLI Tool
- lib: Python Library/Package
- data: Data Analysis Project
- api: API Service
- other: Other

Question 3: Python Version
Options:
- 3.12: Python 3.12+
- 3.11: Python 3.11+
- 3.10: Python 3.10+
- 3.9: Python 3.9+

Question 4: Main Framework (dynamically shown based on project type)
Web Application/API Service:
- fastapi: FastAPI
- flask: Flask
- django: Django
- none: No framework

CLI Tool:
- click: Click
- typer: Typer
- argparse: argparse (standard library)

Data Analysis:
- pandas: Pandas
- polars: Polars
- none: No specific framework
```

---

## Phase 2: Requirements Communication

Gather project requirements through conversational exchange. Ask the user:

If the user only has a vague idea, unclear scope, or needs to evaluate multiple approaches, use `design_brainstorming` first to clarify intent, constraints, success criteria, and trade-offs. Then continue with the questions below.

1. **Core Features**: What problem does this project solve? What are the main features?
2. **Module Division**: What major modules are expected? What are their responsibilities?
3. **Extensibility**: What are potential future extensions?
4. **External Dependencies**: Which external services or APIs need to be integrated?
5. **Data Storage**: Is a database needed? What type?

Record user responses for architecture design and documentation generation.

---

## Phase 3: Architecture Design Confirmation

Based on information collected in previous phases, propose architecture and confirm with user:

### 3.1 Directory Structure Recommendation

Recommend directory structure based on project type:

**Standard Python Project Structure:**
```
{project_name}/
├── .cursor/
│   └── rules.mdc
├── docs/
│   ├── architecture.md
│   └── api.md
├── src/
│   └── {project_name}/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── .gitignore
├── ARCHITECTURE.md
├── CHANGELOG.md
├── LICENSE
├── README.md
└── pyproject.toml
```

### 3.2 Architecture Pattern Selection

Use AskQuestion to inquire about architecture pattern:

```
Question: Project Architecture Pattern
Options:
- simple: Simple Structure (suitable for small projects)
- layered: Layered Architecture (presentation/service/data)
- clean: Clean Architecture (entities/usecases/adapters)
- hexagonal: Hexagonal Architecture (ports and adapters)
```

### 3.3 Confirm Module Division

Display recommended module structure for user confirmation or adjustment.

---

## Phase 4: Configuration Options Confirmation

Use AskQuestion to collect optional configurations:

```
Question 1: Initialize Git repository?
Options:
- yes: Yes
- no: No

Question 2: Create virtual environment?
Options:
- venv: Use venv
- conda: Use conda
- none: Do not create

Question 3: Generate test folder?
Options:
- pytest: Use pytest
- unittest: Use unittest
- none: Do not generate

Question 4: Open source license type
Options:
- mit: MIT License
- apache: Apache License 2.0
- gpl3: GPL v3
- none: No license
```

---

## Phase 5: File Generation

Generate the following files based on collected information using templates:

### 5.1 Required Files

| File | Template | Description |
|------|----------|-------------|
| `.cursor/rules.mdc` | [rules.mdc.template](templates/rules.mdc.template) | Cursor rules file |
| `pyproject.toml` | [pyproject.toml.template](templates/pyproject.toml.template) | Project configuration |
| `README.md` | [readme.md.template](templates/readme.md.template) | Project documentation |
| `ARCHITECTURE.md` | [architecture.md.template](templates/architecture.md.template) | Architecture overview |
| `docs/architecture.md` | Detailed version | Detailed architecture doc |
| `.gitignore` | [gitignore.template](templates/gitignore.template) | Git ignore rules |
| `CHANGELOG.md` | [changelog.md.template](templates/changelog.md.template) | Change log |

### 5.2 Optional Files

| File | Condition |
|------|-----------|
| `LICENSE` | User selected a license type |
| `tests/` | User chose to generate test folder |
| `src/{project_name}/` | Always generated |

### 5.3 Generation Steps

1. Read corresponding template files
2. Replace placeholders in templates:
   - `{{PROJECT_NAME}}` - Project name
   - `{{PROJECT_DESCRIPTION}}` - Project description
   - `{{PYTHON_VERSION}}` - Python version
   - `{{FRAMEWORK}}` - Framework name
   - `{{AUTHOR}}` - Author name
   - `{{YEAR}}` - Current year
   - `{{MODULES}}` - Module list
   - `{{ARCHITECTURE_PATTERN}}` - Architecture pattern
3. Create directory structure
4. Write generated files (**all content in English**)
5. If Git initialization selected, run `git init`
6. If virtual environment creation selected, run corresponding command

**Note:** All generated documentation and code comments must be in English.

---

## Framework-Specific Configuration

### FastAPI Project

Additional files:
- `src/{project_name}/api/` - API routes directory
- `src/{project_name}/schemas/` - Pydantic models
- `src/{project_name}/core/config.py` - Configuration management

pyproject.toml dependencies:
```toml
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0.0",
]
```

### Flask Project

Additional files:
- `src/{project_name}/routes/` - Routes directory
- `src/{project_name}/templates/` - Jinja2 templates

pyproject.toml dependencies:
```toml
dependencies = [
    "flask>=3.0.0",
]
```

### Django Project

Prompt user to use `django-admin startproject` command, then generate supplementary files:
- `.cursor/rules.mdc`
- `docs/` documentation directory
- `ARCHITECTURE.md`

### CLI Project (Click/Typer)

Additional files:
- `src/{project_name}/cli.py` - CLI entry point
- `src/{project_name}/commands/` - Commands directory

### Data Analysis Project

Additional files:
- `notebooks/` - Jupyter notebooks directory
- `data/` - Data directory (added to .gitignore)
- `src/{project_name}/analysis/` - Analysis module

---

## Completion Confirmation

After all files are generated, show user:

1. List of generated files
2. Directory structure preview
3. Next steps:
   - Install dependencies: `pip install -e .`
   - Run tests: `pytest` (if tests were generated)
   - Start development: specific command based on project type

---

## Common Issue Handling

### Project Name Conflict

If target directory already exists, ask user:
- Overwrite existing files
- Only generate missing files
- Cancel operation

### Framework Version Compatibility

Automatically adjust framework version requirements based on Python version.

### Custom Templates

Users can place custom templates in `~/.cursor/skills/project_python_init/templates/`, which will be used with priority.
