"""TMWF baseline (Jin et al., CCS 2023): Transformer-based Multi-tab Website Fingerprinting."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TMWF(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        seq_len: int = 8000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_queries: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(128, embed_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
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
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Learnable tab queries decoded via cross-attention
        self.tab_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        cur = x.size(-1)
        if cur > self.seq_len:
            x = x[..., : self.seq_len]
        elif cur < self.seq_len:
            x = F.pad(x, (0, self.seq_len - cur))

        d = torch.sign(x).float().unsqueeze(1)
        feat = self.cnn_stem(d).transpose(1, 2)  # (B, N, D)
        n = feat.size(1)
        pos = self.pos_embed[:, :n, :] if n <= self.pos_embed.size(1) else F.interpolate(
            self.pos_embed.transpose(1, 2), size=n, mode="linear", align_corners=False
        ).transpose(1, 2)
        memory = self.encoder(feat + pos)

        queries = self.tab_queries.expand(x.size(0), -1, -1)
        dec = self.decoder(queries, memory)  # (B, Q, D)
        query_logits = self.head(dec)        # (B, Q, num_classes)
        # Aggregate per-tab query predictions into multi-label class logits
        return query_logits.amax(dim=1)

