"""Hyperparameter and simulation configurations matching Table 7 and Appendices A, E, F."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class ForgeConfig:
    # Sequence Division (Traffic Channel Construction)
    num_channels: int = 8
    interval: int = 20          # tau
    seq_len: int = 8000         # l

    # Local Profiling (BSS Solver - L cascaded ICMs)
    num_icm_blocks: int = 4
    filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 7, 7, 7])
    pool_configs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(8, 4), (8, 4), (8, 4), (8, 4)]
    )
    bss_dropout: float = 0.1
    bss_out_dim: int = 256

    # Website Identification (DDCA Transformer)
    num_transformer_blocks: int = 2
    num_heads: int = 8
    head_dim: int = 32
    embed_dim: int = 256
    mlp_dim: int = 1024
    id_dropout: float = 0.1
    ddca_alpha_init: float = 0.1
    max_len: int = 512          # Safe upper bound for downsampled sequence length (8000 -> ~31, 10000 -> ~39)
    num_classes: int = 100

    # Training defaults
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    seed: int = 1234


REGULATOR_CONFIGS: Dict[str, Dict[str, float]] = {
    "light": {
        "orig_rate": 260.0,
        "depreciation_rate": 0.86,
        "max_padding_budget": 2080,
        "burst_threshold": 3.75,
        "upload_ratio": 3.95,
        "delay_cap": 1.77,
    },
    "heavy": {
        "orig_rate": 277.0,
        "depreciation_rate": 0.94,
        "max_padding_budget": 3550,
        "burst_threshold": 3.55,
        "upload_ratio": 3.95,
        "delay_cap": 1.77,
    },
}

NOISE_DEFAULT_PARAMS: Dict[str, object] = {
    "seed": 1234,
    "p_gb": 0.003,
    "p_bg": 0.25,
    "p_drop": 0.2,
    "sigma_ms": 2.0,
    "rho": 0.6,
    "p_tail": 0.0,
    "tail_scale_ms": (120.0, 500.0),
    "p_swap": 1e-4,
    "window": 5,
    "snr_levels": [0.1, 0.5, 1, 2, 5, 10],
}

