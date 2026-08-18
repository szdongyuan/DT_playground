# Salt License Protection

## License format

The application reads an extensionless `salt` file from the application root.
Its content must be the lowercase hexadecimal SHA-256 digest of:

```text
<lowercase colon-separated physical MAC address>dongyuan
```

Example input:

```text
a1:b2:c3:d4:e5:f6dongyuan
```

If the file is missing, unreadable, or does not match the current machine, the
application displays `软件授权失败` and exits before loading the main window.

## Generate a salt file from source

Always use the repository virtual environment:

```powershell
& '.\.venv\Scripts\python.exe' '.\tools\generate_salt.py' `
  --output-dir '.\dist\AudioTrainingPlatform'
```

## Build the internal generator

```powershell
& '.\.venv\Scripts\python.exe' -m PyInstaller `
  --clean --noconfirm '.\SaltGenerator.spec'
```

The build creates `dist\SaltGenerator.exe`. To generate a license directly in
the packaged application directory:

```powershell
'.\dist\SaltGenerator.exe' --output-dir '.\dist\AudioTrainingPlatform'
```

`SaltGenerator.exe` is an internal licensing tool. Never include it in the
customer distribution because it can generate a valid license for the machine
on which it runs.

## Minimal verification

```powershell
& '.\.venv\Scripts\python.exe' -m pytest `
  '.\tests\test_generate_salt.py' `
  '.\tests\test_license_validation.py' -q
```

The integration branch for this protection line is `develop_salt_protected`.
Create future salt-related pull requests from a personal `codex/` branch and
target that integration branch.
