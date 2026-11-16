"""Module 3: Website Identifier with Dual-Domain Consistency Attention (DDCA, Section 4.3)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head scaled dot-product self-attention (Eq. 3)."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv_proj(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, N, H, D_h)
        q = q.transpose(1, 2)        # (B, H, N, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)  # (B, H, N, D_h)
        out = out.transpose(1, 2).contiguous().reshape(b, n, d)
        return self.out_proj(out)


class DDCABlock(nn.Module):
    """Dual-Domain Consistency Attention (DDCA) Transformer Block.

    Combines:
      1) Time-Domain Dependency Analyzer (self-attention over temporal tokens)
      2) Frequency-Domain Structure Validator (self-attention over FFT magnitude spectrum |X_f|)
      3) Learnable gated residual fusion (initial alpha = 0.1) + FFN
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.time_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.freq_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        # Time-domain branch
        y_time = self.time_attn(h)

        # Frequency-domain branch: orthonormal FFT along sequence dimension N in float32
        h_f32 = h.float()
        x_f = torch.fft.fft(h_f32, dim=1, norm="ortho")
        mag_f = torch.sqrt(x_f.real.pow(2) + x_f.imag.pow(2) + 1e-8).to(h.dtype)
        y_freq = self.freq_attn(mag_f)

        # Residual fusion gated by learnable alpha
        x = x + self.drop1(y_time + self.alpha * y_freq)
        x = x + self.ffn(self.norm2(x))
        return x


class WebsiteIdentifier(nn.Module):
    """Transformer-based Website Identifier using stacked DDCA blocks (Section 4.3, Table 7)."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 100,
        num_blocks: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                DDCABlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    alpha_init=alpha_init,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, s_hat: torch.Tensor) -> torch.Tensor:
        """Args:
            s_hat: Separated feature streams of shape (B, embed_dim, N).
        Returns:
            logits: Classification logits of shape (B, num_classes).
        """
        x = s_hat.transpose(1, 2)  # (B, N, embed_dim)
        n = x.size(1)
        if n > self.pos_embed.size(1):
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=n, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embed[:, :n, :]
        x = self.pos_drop(x + pos)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)

