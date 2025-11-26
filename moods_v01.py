# moodos_v01.py
# Original MoodOS v0.1 — the brain that started it all
# Tested with test_core.py → 100% pass

import torch
import torch.nn as nn
import numpy as np

def analytic_signal_fft(x):
    X = torch.fft.fft(x, dim=-1)
    n = x.shape[-1]
    h = torch.zeros(n, device=x.device, dtype=X.dtype)
    h[0] = 1
    if n % 2 == 0:
        h[n//2] = 1
    h[1 : n//2 if n%2==0 else (n+1)//2] = 2
    h = h.view((1,)*(x.ndim-1) + (n,))
    return torch.fft.ifft(X * h, dim=-1)

def instantaneous_phase(signal, sample_rate=128, freq_bands_hz=None):
    x = torch.from_numpy(signal).float() if isinstance(signal, np.ndarray) else signal.float()
    if x.ndim == 1: x = x.unsqueeze(0)
    x = x - x.mean(dim=-1, keepdim=True)
    x = x * torch.hann_window(x.shape[-1], device=x.device)
    if freq_bands_hz is None:
        return torch.angle(analytic_signal_fft(x)).unsqueeze(-2)
    phases = []
    X_full = torch.fft.fft(x, dim=-1)
    freqs = torch.fft.fftfreq(x.shape[-1], d=1.0/sample_rate).to(x.device)
    for low, high in freq_bands_hz:
        mask = (freqs >= low) & (freqs < high)
        mask = mask | ((freqs <= -low) & (freqs > -high))
        X_band = X_full.clone()
        X_band[..., ~mask] = 0
        phases.append(torch.angle(torch.fft.ifft(X_band, dim=-1)))
    return torch.stack(phases, dim=-2)

def mood_coherence(phases):
    plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=-1))
    diff = phases.unsqueeze(-2) - phases.unsqueeze(-3)
    triad_phase = diff + diff.transpose(-2, -3)
    triad_mag = torch.abs(torch.mean(torch.exp(1j * triad_phase), dim=-1))
    diag = triad_mag.diagonal(dim1=-2, dim2=-1)
    return torch.sqrt(plv * diag + 1e-12).mean(dim=-1).clamp(0, 1)

class MoodVector32(nn.Module):
    def __init__(self, n_freq_bands=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_freq_bands + 1, 128), nn.SiLU(),
            nn.Linear(128, 96), nn.SiLU(),
            nn.Linear(96, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.Tanh()
        )
    def forward(self, plv, coh):
        if coh.ndim == plv.ndim - 1: coh = coh.unsqueeze(-1)
        return self.net(torch.cat([plv, coh], dim=-1))

def compute_mood_vector(sig, sample_rate=128, freq_bands_hz=None):
    phases = instantaneous_phase(sig, sample_rate, freq_bands_hz)
    plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=-1))
    coh = mood_coherence(phases)
    model = MoodVector32(len(freq_bands_hz) if freq_bands_hz else 1)
    return model(plv, coh)
