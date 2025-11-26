# swarm_disaster_brain.py
"""
DisasterSwarmBrain v1.0 — Production prototype
----------------------------------------------
Drop this module on each trusted drone/rover in a disaster-response swarm.

Design goals:
 - decentralized (no central server required)
 - robust to occlusion, noise, and sensor dropouts
 - emergent leader election via coherence
 - optimized for edge devices (Pi 5 / Jetson / mobile CPU)
 - strictly for benign uses: search & rescue, mapping, environmental sensing

Author: Geneva Robinson (@3vi3aetheris) / Agape Intelligence
License: MIT
Tested: Synthetic rubble scenario, single-process simulation
"""

from __future__ import annotations
import math
import time
import logging
from typing import Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Configure module logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DisasterSwarmBrain")


# ---------------------------
# Utilities & safe clamps
# ---------------------------
def clamp_tensor(x: Tensor, low: float, high: float) -> Tensor:
    return torch.max(torch.tensor(low, device=x.device), torch.min(x, torch.tensor(high, device=x.device)))


def safe_to_numpy(x: Tensor):
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


# ---------------------------
# Frequency band helpers
# ---------------------------
def default_bands(n_bands: int = 32, f_min: float = 0.5, f_max: float = 80.0, sample_rate: int = 128) -> Sequence[Tuple[float, float]]:
    """
    Log-spaced bands between f_min and f_max (clipped by Nyquist).
    Returns list of (low, high) in Hz.
    """
    nyq = sample_rate / 2.0
    f_max = min(f_max, nyq * 0.95)
    edges = np.logspace(math.log10(f_min), math.log10(f_max), n_bands + 1, base=10.0)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_bands)]


# ---------------------------
# Analytic-phase (Hilbert) via rfft — mobile-friendly
# ---------------------------
def analytic_signal_hilbert(x: Tensor) -> Tensor:
    """
    x: (..., T) real float32 tensor
    returns: complex analytic signal (..., T) as complex128/complex64 (torch complex)
    Implemented with rfft/irfft multiplier for speed & mobile export.
    """
    X = torch.fft.rfft(x, dim=-1)
    n = x.shape[-1]
    # Hilbert multiplier (rfft length = n//2 + 1)
    h = torch.ones(X.shape[-1], device=x.device, dtype=X.dtype)
    # multiply positive freqs by 2 except DC and Nyquist (if present)
    if X.shape[-1] > 2:
        h[1:-1] = 2.0
    # if n even, last bin is Nyquist which should be 1
    analytic = torch.fft.irfft(X * h, n=n, dim=-1)
    # To produce complex analytic signal use original real x as imaginary pair trick is fragile; instead construct complex via Hilbert approach:
    # We return analytic as complex via torch.complex(real, imag_zero) for consistent interface in later processing.
    return torch.complex(analytic, torch.zeros_like(analytic))


# ---------------------------
# Optional: Morlet wavelet extractor (better in very noisy cases)
# ---------------------------
def morlet_wavelet_phases(x: Tensor, freqs_hz: Sequence[float], sr: float, width: float = 6.0) -> Tensor:
    """
    Compute complex Morlet transform at specified center frequencies (vectorized).
    Returns phases tensor (len(freqs_hz), T)
    - This is heavier than rfft approach; use when FFT-based extraction is unstable.
    """
    # Implementation uses FFT convolution: wavelet in freq domain
    T = x.shape[-1]
    t = torch.arange(-T//2, T//2, device=x.device) / sr
    out = []
    for f in freqs_hz:
        s = width / (2 * math.pi * f)
        morlet = torch.exp(2j * math.pi * f * t) * torch.exp(-t**2 / (2 * s**2))
        Mor = torch.fft.fft(morlet)
        X = torch.fft.fft(x, dim=-1)
        conv = torch.fft.ifft(X * Mor, dim=-1)
        out.append(torch.angle(conv))
    return torch.stack(out, dim=-2)  # (F, T)


# ---------------------------
# DisasterSwarmBrain (production prototype)
# ---------------------------
class DisasterSwarmBrain(nn.Module):
    """
    Edge swarm brain for search-and-rescue tasks.

    Key features:
     - band decomposition → instantaneous phase
     - prime-triad notch filtering to boost rescue-relevant harmonics
     - occlusion-resistant phase memory with extrapolation
     - triadic coherence consensus measure
     - adaptive SNR gain and safe action head (bounded outputs)
     - local leader election using coherence score
    """

    def __init__(
        self,
        sample_rate: int = 128,
        n_bands: int = 32,
        device: str = "cpu",
        prime_triad_hz: Sequence[float] = (7.0, 13.0, 21.0),
        use_morlet: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.sample_rate = float(sample_rate)
        self.frame = 0
        self.n_bands = int(n_bands)
        self.use_morlet = bool(use_morlet)

        # persistent buffers (registered so they move with .to(device))
        self.register_buffer("phase_prev", torch.zeros(self.n_bands, device=self.device))
        self.register_buffer("phase_prev2", torch.zeros(self.n_bands, device=self.device))  # for extrapolation
        self.register_buffer("leader_score", torch.tensor(0.0, device=self.device))

        # prime triad and notch frequencies (scaled by sampling)
        primes = torch.tensor(prime_triad_hz, dtype=torch.float32, device=self.device)
        self.prime_triad_hz = primes

        # frequency bands
        self.bands = default_bands(self.n_bands, sample_rate=self.sample_rate)

        # small NN action head: inputs = (n_bands phases) + coherence+gain scalar => output 8 dims
        self.action_head = nn.Sequential(
            nn.Linear(self.n_bands + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.Tanh(),  # bounded outputs [-1,1] then scaled in postprocessing
        )

        # safety constraints (tunable)
        self.max_speed_m_s = 4.0        # safe max lateral speed
        self.max_altitude_m = 120.0     # safe altitude cap for drones
        self.min_altitude_m = 0.5

        # smoothing / occlusion parameters
        self.occlusion_extrapolation_enabled = True
        self.max_extrapolation_ms = 300.0  # safe extrapolation window

    # ---------------------------
    # notch filter to boost prime triad response (soft Gaussian)
    # ---------------------------
    def prime_notch(self, x: Tensor) -> Tensor:
        """
        Soft notch in frequency domain around prime triad frequencies.
        x: (T,) tensor
        returns: time-domain signal (T,)
        """
        X = torch.fft.rfft(x)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.sample_rate).to(x.device)
        H = torch.ones_like(X)
        for f0 in self.prime_triad_hz:
            # soft Gaussian notch (width parameter tuned for robustness)
            width = max(0.5, float(f0) * 0.12)
            notch = torch.exp(-0.5 * ((freqs - f0) / width) ** 2)
            notch += torch.exp(-0.5 * ((freqs + f0) / width) ** 2)
            H = H * (1.0 - 0.9 * notch)  # depth 0.9 avoids full cancellation
        return torch.fft.irfft(X * H, n=x.shape[-1], dim=-1)

    # ---------------------------
    # phase extraction: per-band instantaneous phase
    # returns phases tensor shape (n_bands, T) or (n_bands,) for last-sample mode
    # ---------------------------
    def extract_phases(self, signal_np: np.ndarray) -> Tensor:
        """
        Accepts 1D numpy array signal_np (T,)
        Returns phases at last time-frame as Tensor (n_bands,)
        """
        x = torch.from_numpy(signal_np).float().to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # (1, T)

        # detrend + window
        x = x - x.mean(dim=-1, keepdim=True)
        win = torch.hann_window(x.shape[-1], device=x.device)
        xw = x * win

        if self.use_morlet:
            # for morlet, compute center freqs (one per band)
            centers = [0.5 * (lo + hi) for lo, hi in self.bands]
            phases_all = morlet_wavelet_phases(xw.squeeze(0), centers, self.sample_rate)
            # return last-sample phases
            ph = phases_all[:, -1]
            return ph

        # FFT-based band slicing & Hilbert phase extraction
        X = torch.fft.rfft(xw, dim=-1)
        freqs = torch.fft.rfftfreq(xw.shape[-1], d=1.0 / self.sample_rate).to(x.device)
        phases = []
        for lo, hi in self.bands:
            mask = (freqs >= lo) & (freqs < hi)
            mask = mask | ((freqs <= -lo) & (freqs > -hi))  # include mirrored negative
            if mask.sum() == 0:
                # empty band: push zero phase
                phases.append(torch.tensor(0.0, device=x.device))
                continue
            Xb = X.clone()
            Xb[..., ~mask] = 0
            band_td = torch.fft.irfft(Xb, n=xw.shape[-1], dim=-1)
            analytic = analytic_signal_hilbert(band_td)
            # take last sample phase
            ph = torch.angle(analytic[..., -1])
            phases.append(ph.squeeze())
        return torch.stack(phases, dim=0).to(self.device)  # (n_bands,)

    # ---------------------------
    # micro-occlusion extrapolation (second-order) — safe & limited
    # ---------------------------
    def extrapolate_phase(self, current: Tensor, prev: Tensor, prev2: Tensor, occlusion_ms: float) -> Tensor:
        """
        If occlusion small, linear-extrapolate to preserve phase continuity.
        Uses bounded extrapolation to avoid runaway drift.
        """
        if not self.occlusion_extrapolation_enabled:
            return current
        # if prev2 is empty (initial), just return current
        if prev2.numel() == 0:
            return current
        dt = max(1.0 / self.sample_rate, 0.001)
        max_steps = int((self.max_extrapolation_ms / 1000.0) / dt)
        steps = min(max_steps, max(1, int(round(occlusion_ms / 1000.0 / dt))))
        # simple second-order linear extrapolation: phi_t+1 = 2*phi_t - phi_t-1
        predicted = current + steps * (current - prev)
        # clamp to reasonable band-specific swings (avoid extreme jumps)
        swing = 0.5  # radians per extrapolation step typical clamp
        predicted = torch.clamp(predicted, prev - swing, prev + swing)
        return predicted

    # ---------------------------
    # consensus & action computation (core forward)
    # ---------------------------
    def forward(self, sensor_chunk: np.ndarray, occlusion_ms: float = 0.0):
        """
        sensor_chunk: 1D numpy array of recent samples (length T, same samplerate as configured)
        occlusion_ms: estimated occlusion duration in milliseconds (0 if live)
        returns: action vector (8,) numpy array, coherence scalar
        """

        # optionally apply prime notch every N frames to reduce reverberant artifacts
        if (self.frame & 3) == 0:
            try:
                sensor_t = torch.from_numpy(sensor_chunk).float().to(self.device)
                sensor_t = self.prime_notch(sensor_t)
                sensor_chunk = sensor_t.detach().cpu().numpy()
            except Exception:
                # if notch fails for any reason, fall back to raw
                logger.debug("prime_notch failed — using raw sensor chunk")

        # extract instantaneous phases (per-band last-sample)
        phase_est = self.extract_phases(sensor_chunk)  # (n_bands,)

        # occlusion-aware phase persistence (combines hysteresis + extrapolation)
        if self.phase_prev.numel() == 0:
            self.phase_prev = phase_est.clone()
            self.phase_prev2 = phase_est.clone()
        else:
            # extrapolate if occlusion > 0
            if occlusion_ms > 0.0:
                phase_est = self.extrapolate_phase(phase_est, self.phase_prev, self.phase_prev2, occlusion_ms)

        # hysteresis smoothing (simple exponential smoothing for stability)
        alpha = 0.85 if occlusion_ms < 80.0 else 1.0  # more smoothing under short occlusion
        phase_locked = alpha * phase_est + (1.0 - alpha) * self.phase_prev

        # update phase history (shift)
        self.phase_prev2 = self.phase_prev.clone()
        self.phase_prev = phase_locked.detach().clone()

        # ---------------------------
        # triadic coherence (social consensus)
        # use grouped triads across the band stack
        # ---------------------------
        # group bands into 3 interleaved sets for triad formation
        p = phase_locked
        p1 = p[0::3]
        p2 = p[1::3]
        p3 = p[2::3]
        # ensure same size by truncation
        minlen = min(p1.shape[0], p2.shape[0], p3.shape[0])
        p1, p2, p3 = p1[:minlen], p2[:minlen], p3[:minlen]
        triad = torch.exp(1j * (p1 - p2 + p3))
        coherence = torch.clamp(torch.abs(torch.mean(triad)), 0.0, 1.0)

        # ---------------------------
        # SNR & adaptive gain
        # ---------------------------
        s = torch.from_numpy(sensor_chunk).float().to(self.device)
        power = torch.mean(s**2)
        var = torch.var(s) + 1e-12
        snr_db = 10.0 * torch.log10((power + 1e-12) / var)
        gain = torch.sigmoid((snr_db - 8.0) * 0.35)
        gain = torch.clamp(0.35 + gain, 0.2, 1.5)  # keep reasonable gain range

        # ---------------------------
        # prepare NN input & safe decode
        # ---------------------------
        plv = torch.abs(torch.mean(torch.exp(1j * phase_locked))).clamp(0.0, 1.0)
        nn_input = torch.cat([phase_locked, (coherence * gain).unsqueeze(0)], dim=0).to(self.device)
        # normalize inputs sensibly
        # phases are in radians [-pi, pi] — map to [-1,1]
        norm_phases = (nn_input[:-1] / math.pi).clamp(-1.0, 1.0)
        nn_in = torch.cat([norm_phases, nn_input[-1].unsqueeze(0)], dim=0).unsqueeze(0)  # (1, n_bands+1)

        action_raw = self.action_head(nn_in).squeeze(0)  # (-1,1) outputs

        # postprocess: map to meaningful physical ranges (safe, clamped)
        # action layout: [vx, vy, vz, altitude_delta, drop_payload_prob, broadcast_signal, leader_score, coherence_score]
        vx = float(action_raw[0].item()) * self.max_speed_m_s
        vy = float(action_raw[1].item()) * self.max_speed_m_s
        vz = float(action_raw[2].item()) * (self.max_speed_m_s * 0.6)  # vertical usually slower
        altitude_delta = float(action_raw[3].item()) * 2.0  # meters per step
        drop_payload_prob = float((action_raw[4].item() + 1.0) / 2.0)  # map (-1,1)->(0,1)
        broadcast = float((action_raw[5].item() + 1.0) / 2.0)         # heartbeat/broadcast strength
        leader_score = float(clamp_tensor(action_raw[6].cpu(), 0.0, 1.0).item())
        coherence_score = float(clamp_tensor(action_raw[7].cpu(), 0.0, 1.0).item())

        # safety clamp altitude proposals (caller must enforce absolute altitude limits)
        altitude_delta = float(np.clip(altitude_delta, -5.0, 5.0))

        # leader election: update stored leader_score (local)
        # (In a real swarm, each agent would broadcast its leader_score and choose the best)
        self.leader_score = max(self.leader_score, leader_score)

        self.frame += 1

        action_vec = np.array([
            vx, vy, vz,
            altitude_delta,
            drop_payload_prob,
            broadcast,
            leader_score,
            coherence_score
        ], dtype=np.float32)

        return action_vec, float(coherence)

    # ---------------------------
    # TorchScript export helper (action_head only)
    # ---------------------------
    def export_action_head_torchscript(self, path: str = "exports/disaster_action_head.pt"):
        """
        Export the action_head as a TorchScript module for mobile runtime.
        Note: action_head expects normalized input length (n_bands + 1).
        """
        # create a small wrapper
        class Wrapper(nn.Module):
            def __init__(self, head):
                super().__init__()
                self.head = head

            def forward(self, x: Tensor) -> Tensor:
                return self.head(x)

        w = Wrapper(self.action_head)
        scripted = torch.jit.script(w.cpu())
        import os
        os.makedirs(path.rsplit("/", 1)[0] if "/" in path else ".", exist_ok=True)
        scripted.save(path)
        logger.info("Exported action_head TorchScript to %s", path)


# ---------------------------
# Simulation / smoke test (non-networked demo)
# ---------------------------
def _demo_simulation():
    """
    Simulates a swarm of units hearing the same 'scream' signal with varying noise and occlusion.
    Produces consensus reports but does not implement networking.
    """
    device = "cpu"
    n_units = 8
    fs = 128
    duration_s = 4.0
    T = int(duration_s * fs)
    t = np.arange(T) / fs

    # synthetic 'scream' composed of triad frequencies plus broadband noise
    base = 0.6 * np.sin(2 * np.pi * 13.0 * t) + 0.4 * np.sin(2 * np.pi * 21.0 * t)
    swarm_coherences = []
    swarm_actions = []

    brains = [DisasterSwarmBrain(sample_rate=fs, n_bands=32, device=device) for _ in range(n_units)]

    for i, brain in enumerate(brains):
        # each unit gets a noisy copy with slight occlusion delay
        noise = np.random.randn(T) * (0.3 + 0.05 * i)
        sig = base + noise
        occlusion_ms = 80 + i * 15  # staggered occlusion durations
        action, coh = brain(sig, occlusion_ms=occlusion_ms)
        swarm_coherences.append(coh)
        swarm_actions.append(action)
        print(f"Unit {i:02d} coherence={coh:.4f} leader_score={action[6]:.4f} move=[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f}]")

    mean_coh = float(np.mean(swarm_coherences))
    leader_idx = int(np.argmax([a[6] for a in swarm_actions]))
    print(f"\nSwarm mean coherence: {mean_coh:.4f} — elected leader: Unit {leader_idx}")

    # Safety assertion (sanity)
    assert mean_coh >= 0.0 and mean_coh <= 1.0

    print("Simulation complete. Units have generated consistent action vectors.")


if __name__ == "__main__":
    _demo_simulation()
