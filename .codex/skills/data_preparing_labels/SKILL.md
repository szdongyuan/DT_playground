---
name: data-preparing-labels
description: Prepare supervised classification labels for audio datasets while preserving filename alignment and project JSON contracts. Use when creating labels.json files, filename-to-label mappings, class mappings, dataset labels, or label validation data.
---

# Data Preparing Labels

## Overview

Use this skill when preparing label data for supervised classification datasets in this project. The label file should preserve filename alignment and stay compatible with the current `SaveAudioNode` / `LabelFileNode` JSON contract.

## Required JSON Shape

For training-ready datasets, prefer integer class IDs in the top-level `labels` object:

```json
{
  "version": "1.0",
  "description": "Audio labels saved by SaveAudioNode",
  "labels": {
    "audio_0000.wav": 0,
    "audio_0001.wav": 1
  },
  "label_count": 2
}
```

Rules:

- Use `labels.json` as the default filename.
- Use audio filenames as keys. Keys must exactly match the files produced or loaded by the workflow.
- Use one label value per audio file.
- Keep `label_count` equal to the number of entries in `labels`.
- Treat the top-level `labels` object as the canonical source of per-file labels.

## String Class Names

String labels are allowed, but integer class IDs are preferred for training. When using string labels, include a `label_names` mapping from class name to integer ID:

```json
{
  "version": "1.0",
  "description": "Audio labels for supervised classification",
  "labels": {
    "dog_0001.wav": "dog",
    "cat_0001.wav": "cat"
  },
  "label_count": 2,
  "label_names": {
    "dog": 0,
    "cat": 1
  }
}
```

## Preparation Workflow

1. List the audio files that will be used for supervised classification.
2. Assign each file exactly one class label.
3. Prefer stable integer IDs starting at `0`; keep class meanings documented through `label_names` when labels are not self-evident.
4. Write `labels.json` with the top-level `labels` object.
5. Verify every filename in `labels` exists in the dataset folder and every dataset file has a label unless intentionally excluded.

## Avoid

- Do not use a bare JSON list for prepared datasets, even though the reader supports it; lists lose filename alignment.
- Do not mix integer IDs and string names in the same `labels` object.
- Do not rely on row order when filenames are available.
- Do not set `label_count` to the number of classes; it is the number of labeled files.
