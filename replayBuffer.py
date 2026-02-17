import torch
import numpy as np
from collections import deque
class ReplayBuffer:
    """
    Replay Buffer, speichert 
    (s) state
    (a) action
    (r) reward
    (s2) next state
    (d) done
    Speichert historische Zustände, actions und r4ewards zum Training der Q Networks

    """
    def __init__(self, device, size=100_000):
        self.buffer = deque(maxlen=size)
        self.size = size
        self.device = device

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch):
        idx = np.random.randint(0, len(self.buffer), size=batch)
        s, a, r, s2, d = zip(*[self.buffer[i] for i in idx])
        device = self.device
        # Erst in ein NumPy-Array konvertieren
        s  = np.array(s, dtype=np.float32)
        a  = np.array(a, dtype=np.float32)
        r  = np.array(r, dtype=np.float32).reshape(-1, 1)
        s2 = np.array(s2, dtype=np.float32)
        d  = np.array(d, dtype=np.float32).reshape(-1, 1)

        # Dann in Torch-Tensor
        return (
            torch.from_numpy(s).to(device),
            torch.from_numpy(a).to(device),
            torch.from_numpy(r).to(device),
            torch.from_numpy(s2).to(device),
            torch.from_numpy(d).to(device),
        )