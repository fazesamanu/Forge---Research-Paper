"""Shared utilities for RQ evaluation scripts."""

import json
from pathlib import Path
from typing import Dict, Optional
import torch

from forge.config import ForgeConfig
from forge.model import FORGE
from forge.baselines import build_baseline

ALL_MODELS = ["AWF", "DF", "Tik-Tok", "RF", "BAPM", "TMWF", "ARES", "FORGE"]


def resolve_dataset_path(data_dir: Path, name: str) -> Path:
    """Resolve dataset file path case-insensitively (e.g., Closed_2tab.npz vs closed_2tab.npz)."""
    direct = data_dir / name
    if direct.exists():
        return direct
    target_lower = name.lower()
    if data_dir.exists():
        for p in data_dir.iterdir():
            if p.name.lower() == target_lower:
                return p
    raise FileNotFoundError(f"Could not find dataset {name} in {data_dir}")


def build_model_by_name(
    model_name: str,
    num_classes: int,
    seq_len: int = 8000,
    interval: int = 20,
) -> torch.nn.Module:
    if model_name == "FORGE":
        cfg = ForgeConfig(num_classes=num_classes, seq_len=seq_len, interval=interval)
        return FORGE(cfg)
    return build_baseline(model_name, num_classes=num_classes, seq_len=seq_len)


def load_json_if_exists(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

