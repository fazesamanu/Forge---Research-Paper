"""AWF baseline (Rimmer et al., NDSS 2018) adapted for multi-label WF (Appendix C)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AWF(nn.Module):
    def __init__(self, num_classes: int = 100, seq_len: int = 8000):
        super().__init__()
        self.seq_len = seq_len
        self.features = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        cur = x.size(-1)
        if cur > self.seq_len:
            x = x[..., : self.seq_len]
        elif cur < self.seq_len:
            x = F.pad(x, (0, self.seq_len - cur))
        d = torch.sign(x).float().unsqueeze(1)
        feat = self.pool(self.features(d))
        return self.classifier(feat)

