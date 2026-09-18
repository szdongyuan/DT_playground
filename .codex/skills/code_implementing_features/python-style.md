# Python Import Order

Apply these rules when creating or modifying Python production code, tests, and
helper scripts. They make the existing project-initialization template's import
standards explicit for everyday development.

## Required grouping

After the module docstring and any `from __future__` imports, organize imports in
this order, with one blank line between non-empty groups:

1. Python standard library.
2. Third-party packages.
3. Local application code, including absolute `src.*` and relative imports.

For example, `dataclasses`, `hashlib`, and `json` belong to the standard library;
`numpy`, `tensorflow`, and `PySide6` are third-party packages;
`from src.ui.i18n import tr_` is a local application import and belongs last.

```python
import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

from src.ui.i18n import tr_
```

## Editing and review

- Follow the repository's configured import sorter for ordering within groups,
  when one exists. Group order is required even without automated linting.
- Check import grouping in every Python file created or modified by the task
  before declaring the work complete.
- Preserve deliberate lazy/function-local imports for optional dependencies,
  startup cost, or circular-import avoidance. Apply grouping within an import
  block without moving imports across scopes or executable statements solely
  to satisfy formatting.
- Preserve documented import side effects and initialization ordering; explain
  any necessary exception rather than silently changing runtime behavior.
