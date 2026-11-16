"""Deep Fingerprinting (DF) baseline (Sirinam et al., CCS 2018) adapted for multi-label WF."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_cls=nn.ELU, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=8, padding=4),
            nn.BatchNorm1d(out_ch),
            act_cls(),
            nn.Conv1d(out_ch, out_ch, kernel_size=8, padding=3),
            nn.BatchNorm1d(out_ch),
            act_cls(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DF(nn.Module):
    def __init__(self, num_classes: int = 100, seq_len: int = 8000, in_channels: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.backbone = nn.Sequential(
            DFBlock(in_channels, 32, act_cls=nn.ELU, dropout=0.1),
            DFBlock(32, 64, act_cls=nn.ReLU, dropout=0.1),
            DFBlock(64, 128, act_cls=nn.ReLU, dropout=0.1),
            DFBlock(128, 256, act_cls=nn.ReLU, dropout=0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        cur = x.size(-1)
        if cur > self.seq_len:
            x = x[..., : self.seq_len]
        elif cur < self.seq_len:
            x = F.pad(x, (0, self.seq_len - cur))
        return torch.sign(x).float().unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = self._prepare_input(x)
        feat = self.pool(self.backbone(inp))
        return self.classifier(feat)

