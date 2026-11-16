"""Module 2: Blind Source Separation (BSS) Solver via unrolled Iterative Correction Modules (Section 4.2)."""

from typing import List, Tuple
import torch
import torch.nn as nn


class IterativeCorrectionModule(nn.Module):
    """Single unrolled Iterative Correction Module (ICM):
        Step 1: Residual Feature Extraction (Conv1d -> BatchNorm1d -> ReLU)
        Step 2: Correction Signal Synthesis (Conv1d -> BatchNorm1d) -> Delta H
        Step 3: Iterative Refinement (Residual Update -> ReLU -> MaxPool1d -> Dropout)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        pool_kernel: int = 8,
        pool_stride: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        # Step 1: Learnable Analysis Filters + Non-linear Activation
        self.analysis_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.analysis_bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Step 2: Correction Signal Synthesis
        self.synthesis_conv = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.synthesis_bn = nn.BatchNorm1d(out_channels)

        # Residual shortcut projection when channel count changes
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        # Step 3: Multi-scale Feature Propagation
        pool_pad = (pool_kernel - pool_stride) // 2
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_pad)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Step 1: Residual Feature Extraction
        feat = self.relu(self.analysis_bn(self.analysis_conv(h)))
        # Step 2: Correction Signal Synthesis
        delta_h = self.synthesis_bn(self.synthesis_conv(feat))
        # Step 3: Iterative Refinement
        updated = self.relu(self.shortcut(h) + delta_h)
        out = self.dropout(self.pool(updated))
        return out


class BSSSolver(nn.Module):
    """Cascaded L=4 Iterative Correction Modules separating V in R^(B x C_in x T) into S_hat in R^(B x 256 x T')."""

    def __init__(
        self,
        in_channels: int = 8,
        filters: List[int] = None,
        kernel_sizes: List[int] = None,
        pool_configs: List[Tuple[int, int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [7, 7, 7, 7]
        if pool_configs is None:
            pool_configs = [(8, 4), (8, 4), (8, 4), (8, 4)]

        assert len(filters) == len(kernel_sizes) == len(pool_configs)

        stages = []
        c_in = in_channels
        for c_out, k_size, (p_k, p_s) in zip(filters, kernel_sizes, pool_configs):
            stages.append(
                IterativeCorrectionModule(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=k_size,
                    pool_kernel=p_k,
                    pool_stride=p_s,
                    dropout=dropout,
                )
            )
            c_in = c_out

        self.icms = nn.ModuleList(stages)
        self.out_channels = c_in

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        h = v
        for icm in self.icms:
            h = icm(h)
        return h

