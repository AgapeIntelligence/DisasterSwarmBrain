# swarm_disaster_brain.py
# Production v1.0.0 — now auto-exports TorchScript model on run

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# ... [all your existing DisasterSwarmBrain class code unchanged up to forward()] ...

# ← Keep your entire DisasterSwarmBrain class exactly as before
#    (including forward(), prime_notch(), extract_phases(), etc.)
#    Just paste your current class here — only the bottom changes

class DisasterSwarmBrain(nn.Module):
    # ... your full existing class (unchanged) ...
    # (I'm omitting the body here for brevity — keep yours exactly as-is)
    # Just make sure forward() returns (action_vec: np.ndarray, coherence: float)
    pass  # ← replace with your full class

# ——————————————————————————————————————
# AUTO-EXPORT TORCHSCRIPT ON FIRST RUN
# ——————————————————————————————————————
def export_brain_to_torchscript():
    """Export the action head as TorchScript for mobile/embedded deployment"""
    device = "cpu"
    brain = DisasterSwarmBrain(sample_rate=128, n_bands=32, device=device)
    
    # Create dummy input: (n_bands + 1) normalized
    dummy_phases = torch.randn(32) / np.pi  # normalized phases
    dummy_coh_gain = torch.tensor([0.5])
    dummy_input = torch.cat([dummy_phases, dummy_coh_gain]).unsqueeze(0)  # (1, 33)

    # Script only the action head (pure NN — no state)
    scripted_head = torch.jit.trace(brain.action_head, dummy_input)

    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "disaster_swarm_action_head.pt")
    
    scripted_head.save(export_path)
    print(f"TorchScript model exported → {export_path}")
    print(f"   Size: {os.path.getsize(export_path) / 1024:.1f} KB")
    print("   Ready for Android, iOS, Raspberry Pi, Jetson, or MCU via TorchScript runtime")

# ——————————————————————————————————————
# Run export + demo on import
# ——————————————————————————————————————
if __name__ == "__main__":
    export_brain_to_torchscript()  # ← This runs automatically
    
    print("\nRunning demo simulation...")
    brain = DisasterSwarmBrain()
    
    # Your existing _demo_simulation() or quick test
    T = 512
    t = np.arange(T) / 128.0
    scream = 0.7 * np.sin(2*np.pi*13*t) + 0.5 * np.sin(2*np.pi*21*t)
    
    print("Unit  Coherence  LeaderScore  Move[X,Y,Z]")
    for i in range(8):
        noise = scream + np.random.randn(T) * (0.5 + i*0.08)
        action, coh = brain(noise, occlusion_ms=100 + i*25)
        print(f"{i:02d}    {coh:.4f}      {action[6]:.4f}      [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]")
    
    print("\nSwarm brain ready. TorchScript model exported. Deploy anywhere.")
