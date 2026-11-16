"""ARES baseline (Deng et al., IEEE S&P 2023): Robust Multi-tab Website Fingerprinting."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARESBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ARES(nn.Module):
    """Multi-tab Transformer ensemble inspired by ARES (local profiling + Trans-WF attention)."""

    def __init__(
        self,
        num_classes: int = 100,
        seq_len: int = 8000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        # Multi-level local profiling CNN
        self.local_profiler = nn.Sequential(
            ARESBlock(1, 32, dropout=0.1),
            ARESBlock(32, 64, dropout=0.1),
            ARESBlock(64, 128, dropout=0.1),
            ARESBlock(128, embed_dim, dropout=0.1),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        # Multi-head classification projection
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes),
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
        feat = self.local_profiler(d).transpose(1, 2)  # (B, N, D)
        n = feat.size(1)
        pos = self.pos_embed[:, :n, :] if n <= self.pos_embed.size(1) else F.interpolate(
            self.pos_embed.transpose(1, 2), size=n, mode="linear", align_corners=False
        ).transpose(1, 2)
        h = self.norm(self.transformer(feat + pos))
        # Combine mean + max pooling for multi-label detection
        pooled = 0.5 * (h.mean(dim=1) + h.amax(dim=1))
        return self.classifier(pooled)

