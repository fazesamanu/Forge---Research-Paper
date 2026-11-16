"""Robust Fingerprinting (RF) baseline (Shen et al., USENIX Security 2023) with 2D TAM."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RF(nn.Module):
    def __init__(self, num_classes: int = 100, seq_len: int = 8000, num_slots: int = 1800, max_time: float = 80.0):
        super().__init__()
        self.seq_len = seq_len
        self.num_slots = num_slots
        self.max_time = max_time

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 6), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 6), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=(3, 6), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 3)),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=(1, 6), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 6), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(0.1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    @torch.no_grad()
    def build_tam(self, x: torch.Tensor) -> torch.Tensor:
        """Build 2D Traffic Aggregation Matrix of shape (B, 1, 2, num_slots)."""
        b, l = x.shape
        x = x.float()
        d = torch.sign(x)
        ts = torch.abs(x)

        # Fallback if trace only contains {-1, 0, +1}
        nz = d != 0
        if nz.any() and ts[nz].max() <= 1.0 and torch.all(ts[nz] == 1.0):
            ts = (torch.arange(l, device=x.device, dtype=x.dtype).unsqueeze(0).expand_as(x) / float(l)) * self.max_time

        slot_idx = ((ts / self.max_time) * self.num_slots).long().clamp(0, self.num_slots - 1)
        tam = torch.zeros((b, 2, self.num_slots), device=x.device, dtype=torch.float32)

        up_mask = (d > 0).float()
        down_mask = (d < 0).float()
        tam[:, 0, :].scatter_add_(1, slot_idx, up_mask)
        tam[:, 1, :].scatter_add_(1, slot_idx, down_mask)
        return tam.unsqueeze(1)  # (B, 1, 2, num_slots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        cur = x.size(-1)
        if cur > self.seq_len:
            x = x[..., : self.seq_len]
        elif cur < self.seq_len:
            x = F.pad(x, (0, self.seq_len - cur))
        tam = self.build_tam(x)
        feat = self.pool(self.features(tam))
        return self.classifier(feat)

