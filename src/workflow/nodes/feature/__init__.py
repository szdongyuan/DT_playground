# -*- coding: utf-8 -*-
"""
Feature Extraction Nodes

Provides various audio feature extraction functionality organized by dimensionality:
- 1D features: FFT, Statistics, Pitch, Spectral Flatness
- 2D features: Mel Spectrogram, MFCC, STFT, CQT, Spectral Contrast

Architecture Notes
------------------
Feature nodes delegate to the `FeatureExtractor` class in `src/audio/features.py`,
avoiding duplicate implementation of feature extraction logic.
"""

# Import base classes and utilities
from .base import FeatureData, extract_from_audio_or_list

# Import 1D feature nodes
from .dim1d import (
    FFTNode,
    StatisticsNode,
    PitchNode,
    SpectralFlatnessNode,
)

# Import 2D feature nodes
from .dim2d import (
    MelSpectrogramNode,
    MFCCNode,
    STFTNode,
    CQTNode,
    SpectralContrastNode,
)

__all__ = [
    # Base
    'FeatureData',
    'extract_from_audio_or_list',
    # 1D nodes
    'FFTNode',
    'StatisticsNode',
    'PitchNode',
    'SpectralFlatnessNode',
    # 2D nodes
    'MelSpectrogramNode',
    'MFCCNode',
    'STFTNode',
    'CQTNode',
    'SpectralContrastNode',
]
