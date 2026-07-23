# Coding Standards

> **Purpose**: This document defines the coding standards and best practices for the AI Acoustic Signal Training Platform.

## Import Standards

Organize import statements in the following order, with one blank line between groups, **alphabetically sorted within each group**:

1. **Python standard library** (e.g., os, sys, json)
2. **Third-party libraries** (e.g., numpy, tensorflow, PySide6)
3. **Local modules** (e.g., src.audio, src.ui)

```python
# Python standard library
import json
import os
import sys
from pathlib import Path

# Third-party libraries
import numpy as np
import tensorflow as tf
from PySide6.QtWidgets import QWidget

# Local modules
from src.audio.features import FeatureExtractor
from src.ui.styles import Styles
from src.workflow.engine import WorkflowEngine
```

---

## Python Style

- Follow PEP 8 guidelines
- Use type annotations (Type Hints)
- Use docstrings to document functions and classes
- Use `snake_case` for variable names
- Use `PascalCase` for class names
- Use `UPPER_SNAKE_CASE` for constants
- New comments and docstrings must be written in English. Existing Chinese comments/docstrings may be migrated opportunistically when touching nearby code.

---

## PySide6 Standards

- UI component classes end with `Widget`/`Dialog`/`Window`
- Define signals using `Signal`
- Slot functions start with `_on_` or `_handle_`
- Use `QThread` for time-consuming operations to avoid blocking UI
- **Unified style management**: Use the `Styles` class from `src/ui/styles.py`
- Workflow and model editor UI should reuse shared graph editor items from `src/ui/graph_editor/` where practical.

```python
from src.ui.styles import Styles

widget.setStyleSheet(Styles.FORM_CONTROLS)
button.setStyleSheet(Styles.BUTTON_PRIMARY)
group.setStyleSheet(Styles.group_box(Styles.COLORS['blue']))
```

---

## UI Text and i18n

- Use English source msgids wrapped with `tr_(...)` from `src.ui.i18n` for user-facing strings.
- Use `ngettext()` for plural text and `pgettext()` when a short msgid needs context.
- Do not hardcode Simplified Chinese in Python UI code unless the string is intentionally not localized.
- Update gettext catalogs under `src/locale/` when adding, changing, or removing user-facing strings.
- Use the `project_managing_i18n` skill for the procedural extract, merge, translate, and compile workflow.

```python
from src.ui.i18n import tr_

button.setText(tr_("Open workflow"))
```

---

## TensorFlow Standards

- Build models using Keras Sequential or Functional API
- Use custom Callbacks for training process to interact with UI
- Use `tf.data.Dataset` when it fits the data pipeline; the current audio training path also uses Keras `Sequence` generators.
- Save trained models in the format expected by the caller (`.keras`, `.h5`, or SavedModel when explicitly needed).

---

## Audio Processing Standards

- Use librosa or soundfile uniformly for audio loading
- Sample rate handling: explicitly specify target sample rate when loading
- Keep feature extraction parameters consistent (`n_fft`, `hop_length`, etc.)
- Normalize audio data to `[-1, 1]` range

---

## Common Acoustic Signal Processing Parameters

```python
# Standard audio parameters
SAMPLE_RATE = 44100      # Sample rate
DURATION = 3.0           # Audio duration (seconds)
N_FFT = 2048             # FFT window size
HOP_LENGTH = 512         # Hop length
N_MELS = 128             # Number of Mel filters
N_MFCC = 20              # Number of MFCC coefficients
```

---

## Thread Safety

- Training tasks run in independent `QThread`
- Audio processing tasks execute in separate threads
- Use signals to pass state to UI thread
- Avoid updating UI components directly from worker threads

---

## Error Handling

- Use `try-except` to catch exceptions
- Display user-friendly error messages
- Provide detailed information when audio loading fails
- Log detailed error logs

---

## Performance Optimization

- Use GPU acceleration (if available)
- Load audio data using generators
- Cache spectrogram calculations
- Batch UI updates using timers

---

## Testing Requirements

- Write unit tests for core functionality
- The current UI test pattern uses `QT_QPA_PLATFORM=offscreen` and a manual `QApplication.instance() or QApplication([])` setup.
- Tests currently mix pytest functions and `unittest.TestCase`; follow the surrounding file style unless creating a new focused pytest test.
- Test audio processing module independently
- Use the `test_running` and `test_writing` skills for procedural testing guidance.

```python
import os

from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
app = QApplication.instance() or QApplication([])
```

---

## Dependencies and Tooling

- Runtime dependencies are declared in `requirements.txt`.
- Current core versions include TensorFlow `>=2.17.0`, PySide6 `>=6.6.0`, NumPy `>=1.26,<2.0`, librosa, soundfile, scipy, resampy, sounddevice, matplotlib, and pyqtgraph.
- Do not document optional or unused packages as required dependencies unless they are added to `requirements.txt`.
- No project-level pytest configuration file is currently present; add one only as a deliberate tooling change.

---

*Last Updated: 2026-05-24*
