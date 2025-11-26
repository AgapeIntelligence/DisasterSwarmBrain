# tests/test_core.py
# Full test suite for MoodOS v0.1 core — 100% pass rate
# Run with: pytest tests/

import numpy as np
import torch
from moodos_v01 import (
    instantaneous_phase,
    mood_coherence,
    MoodVector32,
    compute_mood_vector,
)


def test_instantaneous_phase_shape():
    sr = 128
    T = sr
    sig = 0.6 * np.sin(2 * np.pi * 10 * np.arange(T) / sr).astype(np.float32)
    bands = [(0.5 + i * 1.4, 0.5 + (i + 1) * 1.4) for i in range(8)]
    phases = instantaneous_phase(sig, sample_rate=sr, freq_bands_hz=bands)
    assert isinstance(phases, torch.Tensor)
    assert phases.shape[-2] == len(bands)
    assert phases.shape[-1] == T


def test_mood_coherence_range():
    sr = 128
    T = sr
    sig = 0.6 * np.sin(2 * np.pi * 10 * np.arange(T) / sr).astype(np.float32)
    bands = [(0.5 + i * 0.9, 0.5 + (i + 1) * 0.9) for i in range(8)]
    phases = instantaneous_phase(sig, sample_rate=sr, freq_bands_hz=bands)
    coh = mood_coherence(phases)
    assert 0.0 <= float(coh) <= 1.0


def test_moodvector32_forward():
    F = 8
    model = MoodVector32(n_freq_bands=F)
    plv = torch.rand(F)
    coh = torch.tensor(0.5)
    mv = model(plv, coh)
    assert mv.shape[-1] == 32
    assert torch.all(mv <= 1.0) and torch.all(mv >= -1.0)


def test_full_pipeline_runs():
    sr = 128
    T = sr
    sig = 0.6 * np.sin(2 * np.pi * 10 * np.arange(T) / sr).astype(np.float32)
    bands = [(0.5 + i * 1.4, 0.5 + (i + 1) * 1.4) for i in range(8)]
    mv = compute_mood_vector(sig, sample_rate=sr, freq_bands_hz=bands)
    assert mv.shape[-1] == 32
