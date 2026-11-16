"""Complete FORGE model and ablation variants (Table 6, Section 5.7)."""

import torch
import torch.nn as nn

from .config import ForgeConfig
from .channel_construction import TrafficChannelConstructor
from .bss_solver import BSSSolver
from .identifier import WebsiteIdentifier


class FORGE(nn.Module):
    """FORGE: Multi-tab Website Fingerprinting via Blind Source Separation.

    Supports ablation flags matching Table 6:
        Case1: use_tde=False, use_bss=True,  use_identifier=True
        Case2: use_tde=True,  use_bss=False, use_identifier=True
        Case3: use_tde=True,  use_bss=True,  use_identifier=False
        Case4: use_tde=True,  use_bss=True,  use_identifier=True (default full FORGE)
    """

    def __init__(
        self,
        config: ForgeConfig = None,
        use_tde: bool = True,
        use_bss: bool = True,
        use_identifier: bool = True,
    ):
        super().__init__()
        if config is None:
            config = ForgeConfig()
        self.config = config
        self.use_tde = use_tde
        self.use_bss = use_bss
        self.use_identifier = use_identifier

        # Module 1: Time-Delay Embedding
        if use_tde:
            self.tde = TrafficChannelConstructor(
                interval=config.interval, seq_len=config.seq_len
            )
            in_channels = config.num_channels
        else:
            self.tde = None
            in_channels = 1

        # Module 2: BSS Solver
        if use_bss:
            self.bss = BSSSolver(
                in_channels=in_channels,
                filters=config.filters,
                kernel_sizes=config.kernel_sizes,
                pool_configs=config.pool_configs,
                dropout=config.bss_dropout,
            )
            feat_dim = self.bss.out_channels
        else:
            # Downsample & linearly project channels to embed_dim without iterative BSS separation
            self.bss = None
            total_stride = 1
            for _, s in config.pool_configs:
                total_stride *= s
            self.no_bss_proj = nn.Sequential(
                nn.AvgPool1d(kernel_size=total_stride, stride=total_stride),
                nn.Conv1d(in_channels, config.embed_dim, kernel_size=1),
            )
            feat_dim = config.embed_dim

        # Module 3: Website Identifier (DDCA vs linear ablation Case3)
        if use_identifier:
            self.identifier = WebsiteIdentifier(
                embed_dim=feat_dim,
                num_classes=config.num_classes,
                num_blocks=config.num_transformer_blocks,
                num_heads=config.num_heads,
                mlp_dim=config.mlp_dim,
                max_len=config.max_len,
                dropout=config.id_dropout,
                alpha_init=config.ddca_alpha_init,
            )
        else:
            self.identifier = None
            self.linear_head = nn.Linear(feat_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Raw traffic trace tensor of shape (B, L) or (B, 1, L).
        Returns:
            logits: Classification logits of shape (B, num_classes).
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        if self.use_tde:
            v = self.tde(x)  # (B, 8, seq_len)
        else:
            x_aligned = TrafficChannelConstructor.align_length(
                x.float(), self.config.seq_len
            )
            v = x_aligned.unsqueeze(1)  # (B, 1, seq_len)

        if self.use_bss:
            s_hat = self.bss(v)  # (B, 256, N)
        else:
            s_hat = self.no_bss_proj(v)  # (B, 256, N)

        if self.use_identifier:
            logits = self.identifier(s_hat)
        else:
            logits = self.linear_head(s_hat.mean(dim=-1))

        return logits

