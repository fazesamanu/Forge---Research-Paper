"""Tik-Tok baseline (Rahman et al., PETS 2020) fusing directional and temporal features."""

import torch
import torch.nn.functional as F
from .df import DF


class TikTok(DF):
    def __init__(self, num_classes: int = 100, seq_len: int = 8000):
        super().__init__(num_classes=num_classes, seq_len=seq_len, in_channels=3)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        cur = x.size(-1)
        if cur > self.seq_len:
            x = x[..., : self.seq_len]
        elif cur < self.seq_len:
            x = F.pad(x, (0, self.seq_len - cur))
        x = x.float()
        d = torch.sign(x)
        ts = torch.abs(x)
        # Normalize timestamps per sample to stabilize CNN gradients
        max_ts = ts.amax(dim=-1, keepdim=True).clamp_min(1.0)
        ts_norm = ts / max_ts
        dir_time = d * ts_norm
        return torch.stack([d, ts_norm, dir_time], dim=1)  # (B, 3, seq_len)

