"""BAPM baseline (Guan et al., ACSAC 2021): Block Attention Profiling Model for multi-tab WF."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BAPM(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        seq_len: int = 8000,
        num_blocks: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        self.block_len = seq_len // num_blocks

        self.block_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.block_attention = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        target_len = self.num_blocks * self.block_len
        cur = x.size(-1)
        if cur > target_len:
            x = x[..., :target_len]
        elif cur < target_len:
            x = F.pad(x, (0, target_len - cur))

        b = x.size(0)
        d = torch.sign(x).float()
        blocks = d.view(b * self.num_blocks, 1, self.block_len)
        tokens = self.block_encoder(blocks).view(b, self.num_blocks, -1)
        tokens = tokens + self.pos_embed
        out = self.block_attention(tokens)
        return self.classifier(out.mean(dim=1))

