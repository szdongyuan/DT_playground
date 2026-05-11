---
name: project_managing_i18n
description: Manage gettext i18n updates for this project. Use when adding, changing, removing, or reviewing user-visible UI text, labels, menus, dialogs, tooltips, validation messages, or translated strings.
---

# Project Managing I18n

## Overview

Use this skill whenever a change touches user-visible interface text. Every UI text change must finish the gettext cycle: extract, merge, translate, compile.

## Required Workflow

Run commands from the repository root.

1. Extract updated strings into `src/locale/messages.pot`:
   ```powershell
   ./tools/i18n.ps1 -Action extract -MsysRoot "C:/msys64" -Toolchain ucrt64
   ```
2. Merge the template into existing catalogs:
   ```powershell
   ./tools/i18n.ps1 -Action merge -MsysRoot "C:/msys64" -Toolchain ucrt64
   ```
3. Translate every new or changed entry in:
   - `src/locale/en_US/LC_MESSAGES/messages.po`
   - `src/locale/zh_CN/LC_MESSAGES/messages.po`
4. Compile catalogs after translations are complete:
   ```powershell
   ./tools/i18n.ps1 -Action compile -MsysRoot "C:/msys64" -Toolchain ucrt64
   ```

## Shortcut

For a full catalog refresh, this command performs extract, merge, and compile:

```powershell
./tools/i18n.ps1 -Action all -MsysRoot "C:/msys64" -Toolchain ucrt64
```

If this command introduces untranslated or fuzzy entries, translate the `.po` files and run the compile command again before finishing.

## Checks Before Finishing

- Confirm new UI strings are wrapped in the project's gettext helper, usually `tr_(...)`.
- Confirm `.po` files have no accidental empty `msgstr` values for new user-visible strings.
- Preserve format placeholders exactly, including names such as `{path}` and printf-style placeholders if present.
- Preserve keyboard accelerators and menu markers such as `(&F)` when translating.
- Include generated `.pot`, `.po`, and `.mo` changes when they are caused by the UI text change.

## Common Mistakes

- Do not stop after editing source files; catalogs must be updated in the same task.
- Do not run compile before translating and call the task complete.
- Do not change `msgid` text inside `.po` files manually; update source strings, then extract and merge.
- Do not drop placeholders, punctuation, ellipses, or accelerator markers during translation.
