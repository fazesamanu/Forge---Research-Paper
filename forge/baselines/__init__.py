"""Baseline WF models adapted for multi-label classification (Section 5.1.2 & Appendix C)."""

from .awf import AWF
from .df import DF
from .tiktok import TikTok
from .rf import RF
from .bapm import BAPM
from .tmwf import TMWF
from .ares import ARES

BASELINE_REGISTRY = {
    "AWF": AWF,
    "DF": DF,
    "Tik-Tok": TikTok,
    "RF": RF,
    "BAPM": BAPM,
    "TMWF": TMWF,
    "ARES": ARES,
}


def build_baseline(name: str, num_classes: int, seq_len: int = 8000):
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[name](num_classes=num_classes, seq_len=seq_len)


__all__ = ["AWF", "DF", "TikTok", "RF", "BAPM", "TMWF", "ARES", "BASELINE_REGISTRY", "build_baseline"]

