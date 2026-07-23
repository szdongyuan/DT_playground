---
name: shell-powershell-commands
description: Execute PowerShell commands on Windows with safe path, process, file, GitHub CLI, and environment handling. Use when running terminal commands, file operations, process management, Git or gh commands, scripts, network checks, or Windows shell operations.
---

# PowerShell Commands Guide

This skill provides command references and best practices for executing PowerShell commands on Windows systems.

## Overview

PowerShell is the default shell on Windows. When executing commands via the Shell tool, use PowerShell syntax instead of Unix/Bash commands.

## Quick Reference

### File & Directory Operations

| Task | PowerShell Command | Bash Equivalent |
|------|-------------------|-----------------|
| List files | `Get-ChildItem` or `ls` or `dir` | `ls` |
| List with details | `Get-ChildItem -Force` | `ls -la` |
| Change directory | `Set-Location path` or `cd path` | `cd path` |
| Current directory | `Get-Location` or `pwd` | `pwd` |
| Create directory | `New-Item -ItemType Directory -Name "dir"` or `mkdir dir` | `mkdir dir` |
| Remove directory | `Remove-Item -Recurse -Force dir` | `rm -rf dir` |
| Copy file | `Copy-Item src dst` or `cp src dst` | `cp src dst` |
| Move file | `Move-Item src dst` or `mv src dst` | `mv src dst` |
| Delete file | `Remove-Item file` or `rm file` | `rm file` |
| Read file | `Get-Content file` or `cat file` | `cat file` |
| Write to file | `Set-Content file -Value "content"` | `echo "content" > file` |
| Append to file | `Add-Content file -Value "content"` | `echo "content" >> file` |
| Find files | `Get-ChildItem -Recurse -Filter "*.txt"` | `find . -name "*.txt"` |

### Process Management

| Task | PowerShell Command |
|------|-------------------|
| List processes | `Get-Process` |
| Find process by name | `Get-Process -Name "python"` |
| Kill process by name | `Stop-Process -Name "python" -Force` |
| Kill process by PID | `Stop-Process -Id 1234 -Force` |
| Start process | `Start-Process "program.exe"` |
| Run in background | `Start-Process "program.exe" -NoNewWindow` |

### Text Processing

| Task | PowerShell Command | Bash Equivalent |
|------|-------------------|-----------------|
| Search text in files | `Select-String -Path "*.txt" -Pattern "search"` | `grep "search" *.txt` |
| Search recursively | `Get-ChildItem -Recurse | Select-String "pattern"` | `grep -r "pattern" .` |
| Count lines | `(Get-Content file).Count` | `wc -l file` |
| First N lines | `Get-Content file -Head 10` | `head -10 file` |
| Last N lines | `Get-Content file -Tail 10` | `tail -10 file` |
| Sort output | `Get-Content file | Sort-Object` | `sort file` |
| Unique lines | `Get-Content file | Sort-Object -Unique` | `sort -u file` |

### Environment & Variables

| Task | PowerShell Command |
|------|-------------------|
| View env variable | `$env:PATH` |
| Set env variable (session) | `$env:MY_VAR = "value"` |
| List all env variables | `Get-ChildItem Env:` |
| View PATH | `$env:PATH -split ';'` |

### GitHub CLI Environment in Codex Sandbox

On this Windows Codex Desktop setup, shell processes may run as the
`codexsandboxoffline` user and may not inherit User/Machine environment
variables into the current process. Before using GitHub CLI (`gh`) or commands
that need GitHub HTTPS access, explicitly hydrate proxy and token variables from
persisted User/Machine environment values:

```powershell
$env:HTTP_PROXY  = [Environment]::GetEnvironmentVariable('HTTP_PROXY', 'User')  ?? [Environment]::GetEnvironmentVariable('HTTP_PROXY', 'Machine')
$env:HTTPS_PROXY = [Environment]::GetEnvironmentVariable('HTTPS_PROXY', 'User') ?? [Environment]::GetEnvironmentVariable('HTTPS_PROXY', 'Machine')
$env:ALL_PROXY   = [Environment]::GetEnvironmentVariable('ALL_PROXY', 'User')   ?? [Environment]::GetEnvironmentVariable('ALL_PROXY', 'Machine')
$env:GH_TOKEN    = [Environment]::GetEnvironmentVariable('GH_TOKEN', 'User')    ?? [Environment]::GetEnvironmentVariable('GH_TOKEN', 'Machine')

gh auth status
```

Do not print `GH_TOKEN`. Check only whether it exists with
`[bool]$env:GH_TOKEN` when diagnosing authentication.

### Network Operations

| Task | PowerShell Command |
|------|-------------------|
| Download file | `Invoke-WebRequest -Uri "url" -OutFile "file"` |
| HTTP GET request | `Invoke-RestMethod -Uri "url"` |
| HTTP POST request | `Invoke-RestMethod -Uri "url" -Method Post -Body $data` |
| Check port | `Test-NetConnection -ComputerName localhost -Port 8080` |
| Get IP info | `Get-NetIPAddress` |

### Package Management

| Task | PowerShell Command |
|------|-------------------|
| Python pip install | `pip install package` or `python -m pip install package` |
| Node npm install | `npm install package` |
| Chocolatey install | `choco install package` |
| Winget install | `winget install package` |

## Command Chaining

### Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `;` | Run sequentially (ignore errors) | `cmd1; cmd2` |
| `&&` | Run if previous succeeds (PS 7+) | `cmd1 && cmd2` |
| `\|\|` | Run if previous fails (PS 7+) | `cmd1 \|\| cmd2` |
| `\|` | Pipe output | `Get-Process \| Where-Object {$_.CPU -gt 10}` |

### PowerShell 5.x Chaining

For older PowerShell versions, use:

```powershell
# Instead of && (run if success)
cmd1; if ($?) { cmd2 }

# Instead of || (run if fail)
cmd1; if (-not $?) { cmd2 }
```

## Path Handling

### Important Notes

1. **Use forward slashes or escaped backslashes**
   ```powershell
   # Both work in PowerShell
   cd D:/pyProject/folder
   cd D:\pyProject\folder
   ```

2. **Quote paths with spaces**
   ```powershell
   cd "D:\My Documents\folder"
   ```

3. **Use Join-Path for building paths**
   ```powershell
   $path = Join-Path $env:USERPROFILE "Documents"
   ```

## Common Patterns

### Run Python Scripts

```powershell
# Run directly
python script.py

# With arguments
python script.py --arg1 value1

# With virtual environment
.\.venv\Scripts\Activate.ps1; python script.py
```

### Git Operations

```powershell
# Status and diff
git status
git diff

# Add and commit
git add .; git commit -m "message"

# Push to remote
git push origin branch-name
```

### Development Servers

```powershell
# Python Flask/Django
python manage.py runserver

# Node.js
npm run dev

# Note: Use block_until_ms: 0 for long-running servers
```

### Error Handling

```powershell
# Try-catch block
try {
    Get-Content "nonexistent.txt" -ErrorAction Stop
} catch {
    Write-Host "Error: $_"
}

# Check last command status
command; if ($?) { "Success" } else { "Failed" }
```

## Best Practices

### For Agent Usage

1. **Prefer dedicated tools over shell commands**
   - Use Read tool instead of `Get-Content`
   - Use LS tool instead of `Get-ChildItem`
   - Use Grep tool instead of `Select-String`
   - Use Write/StrReplace tools instead of `Set-Content`

2. **Use shell for actual system operations**
   - Git commands
   - Package installation (pip, npm)
   - Running scripts and programs
   - Process management
   - Network operations

3. **Quote all paths with spaces**
   ```powershell
   cd "D:\path with spaces\folder"
   ```

4. **Check PowerShell version for compatibility**
   ```powershell
   $PSVersionTable.PSVersion
   ```

5. **Use `-Force` for non-interactive deletion**
   ```powershell
   Remove-Item -Recurse -Force folder
   ```

### Working Directory

- Use `working_directory` parameter in Shell tool instead of `cd`
- Each Shell call starts fresh; state does not persist between calls

### Long-Running Commands

For dev servers or watchers, set `block_until_ms: 0` to run in background.

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Script execution disabled | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Command not found | Check PATH, use full path, or install package |
| Access denied | Run as Administrator or check permissions |
| Encoding issues | Use `-Encoding UTF8` with file operations |

### Check if Command Exists

```powershell
Get-Command python -ErrorAction SilentlyContinue
```

### Get Help

```powershell
Get-Help Get-ChildItem
Get-Help Get-ChildItem -Examples
```
