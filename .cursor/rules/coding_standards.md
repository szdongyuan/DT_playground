# Coding Standards

> **Purpose**: This document defines the coding standards and best practices for the AI Acoustic Signal Training Platform.

## Import Standards

Organize import statements in the following order, with one blank line between groups, **alphabetically sorted within each group**:

1. **Python standard library** (e.g., os, sys, json)
2. **Third-party libraries** (e.g., numpy, tensorflow, PyQt6)
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
from PyQt6.QtWidgets import QWidget

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

---

## PyQt6 Standards

- UI component classes end with `Widget`/`Dialog`/`Window`
- Define signals using `pyqtSignal`
- Slot functions start with `_on_` or `_handle_`
- Use `QThread` for time-consuming operations to avoid blocking UI
- **Unified style management**: Use the `Styles` class from `src/ui/styles.py`

```python
from src.ui.styles import Styles

widget.setStyleSheet(Styles.FORM_CONTROLS)
button.setStyleSheet(Styles.BUTTON_PRIMARY)
group.setStyleSheet(Styles.group_box(Styles.COLORS['blue']))
```

---

## TensorFlow Standards

- Build models using Keras Sequential or Functional API
- Use custom Callbacks for training process to interact with UI
- Use `tf.data.Dataset` for data pipeline processing
- Save models using SavedModel format

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
- Use `pytest-qt` for UI testing
- Test audio processing module independently

---

*Last Updated: 2026-01-24*
