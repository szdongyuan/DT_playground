---
name: project-python-environment
description: Enforce use of this repository's .venv Python interpreter for scripts, dependency management, tests, and PyInstaller packaging. Use whenever running Python, pip, pytest, project scripts, or packaging commands in this repository.
---

# Project Python Environment

## Rule

All repository Python work must use `./.venv/Scripts/python.exe` explicitly.
Do not rely on shell activation or `PATH`, because each tool call may start a new
PowerShell process.

Never execute bare `python`, `py`, `pip`, `pytest`, `pyinstaller`, or
`PyInstaller` commands for this repository. Do not fall back to a global Python
installation when `.venv` is missing or incomplete.

## Required preflight

Before tests, application execution, dependency changes, or packaging:

```powershell
if (-not (Test-Path -LiteralPath './.venv/Scripts/python.exe')) {
    throw 'Project interpreter not found: ./.venv/Scripts/python.exe'
}

& './.venv/Scripts/python.exe' -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix, 'Not running in a virtual environment'"
```

The printed executable must resolve inside this repository's `.venv`. If the
check fails, stop and report the environment problem. Do not continue with a
global interpreter.

## Command patterns

```powershell
# Run a project script
& './.venv/Scripts/python.exe' ./main.py

# Install or inspect dependencies
& './.venv/Scripts/python.exe' -m pip install -r ./requirements.txt
& './.venv/Scripts/python.exe' -m pip list

# Run tests
& './.venv/Scripts/python.exe' -m pytest

# Package the application
& './.venv/Scripts/python.exe' -m PyInstaller --clean ./AudioTrainingPlatform.spec
```

Use `python -m <module>` through the explicit interpreter so the command and
its imported packages always come from the same environment.

## Verification and reporting

For testing and packaging results:

1. Record the interpreter path printed by the preflight.
2. Run the requested command through that interpreter.
3. Treat results from any other interpreter as invalid and rerun them in
   `.venv`.
4. Include the interpreter path in the completion summary.

## Related skills

- For pytest execution and result analysis, also use
  `../test_running/SKILL.md`.
- For PowerShell command mechanics, also use
  `../shell_powershell_commands/SKILL.md`.
