"""Configurable network noise-injection pipeline (Appendix F, Table 14).

Applies operators sequentially in order:
    Burst Loss (Gilbert-Elliott) -> Temporal Jitter (AR(1) + optional tail) -> Local Reordering
repairing monotonic non-decreasing timestamps after each step.
"""

from typing import Tuple
import numpy as np


def _apply_burst_loss(
    t: np.ndarray,
    d: np.ndarray,
    alpha_loss: float,
    p_gb: float,
    p_bg: float,
    p_drop: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-state Gilbert-Elliott Markov burst packet loss model."""
    n = len(t)
    if n == 0 or alpha_loss <= 0:
        return t, d

    # Scale activation density with alpha_loss while clamping probabilities <= 1.0
    eff_p_gb = min(1.0, p_gb * alpha_loss)
    eff_p_drop = min(0.95, p_drop * min(alpha_loss, 4.0) if alpha_loss > 1.0 else p_drop)

    keep = np.ones(n, dtype=bool)
    in_bad = False
    u_trans = rng.random(n)
    u_drop = rng.random(n)

    for k in range(n):
        if not in_bad:
            if u_trans[k] < eff_p_gb:
                in_bad = True
        else:
            if u_trans[k] < p_bg:
                in_bad = False
        if in_bad and u_drop[k] < eff_p_drop:
            keep[k] = False

    t_kept = t[keep]
    d_kept = d[keep]
    if len(t_kept) > 0:
        t_kept = np.maximum.accumulate(t_kept)
    return t_kept, d_kept


def _apply_temporal_jitter(
    t: np.ndarray,
    alpha_jit: float,
    sigma_ms: float,
    rho: float,
    p_tail: float,
    tail_scale_ms: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """AR(1) temporal jitter + optional heavy tail, with monotonic repair."""
    n = len(t)
    if n == 0 or alpha_jit <= 0:
        return t

    # Convert ms to seconds (WFlib timestamps are in seconds)
    sigma_s = (sigma_ms / 1000.0) * alpha_jit
    eps = rng.normal(0.0, sigma_s, size=n)
    j = np.empty(n, dtype=np.float64)
    cur = 0.0
    for k in range(n):
        cur = rho * cur + eps[k]
        j[k] = cur

    if p_tail > 0:
        tail_mask = rng.random(n) < (p_tail * min(alpha_jit, 1.0))
        if np.any(tail_mask):
            s_min, s_max = tail_scale_ms[0] / 1000.0, tail_scale_ms[1] / 1000.0
            j[tail_mask] += rng.uniform(s_min, s_max, size=int(np.sum(tail_mask)))

    t_new = np.maximum(0.0, t + j)
    return np.maximum.accumulate(t_new)


def _apply_local_reordering(
    t: np.ndarray,
    d: np.ndarray,
    alpha_reorder: float,
    p_swap: float,
    window: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Local reordering within window size w with probability alpha_reorder * p_swap."""
    n = len(t)
    if n <= 1 or alpha_reorder <= 0:
        return t, d

    eff_p_swap = min(1.0, p_swap * alpha_reorder * 100.0 if alpha_reorder <= 1.0 else p_swap * 100.0)
    # Apply exact p_swap * alpha_reorder per Appendix F
    eff_p = min(0.5, p_swap * alpha_reorder)
    t_out = t.copy()
    d_out = d.copy()

    probs = rng.random(n - 1)
    for k in range(n - 1):
        if probs[k] < eff_p:
            span = min(n - 1, k + rng.integers(1, max(2, window)))
            d_out[k], d_out[span] = d_out[span], d_out[k]
            t_out[k], t_out[span] = t_out[span], t_out[k]

    t_out = np.maximum.accumulate(t_out)
    return t_out, d_out


def inject_network_noise(
    X: np.ndarray,
    snr_level: float,
    seed: int = 1234,
    p_gb: float = 0.003,
    p_bg: float = 0.25,
    p_drop: float = 0.2,
    sigma_ms: float = 2.0,
    rho: float = 0.6,
    p_tail: float = 0.0,
    tail_scale_ms: Tuple[float, float] = (120.0, 500.0),
    p_swap: float = 1e-4,
    window: int = 5,
    return_mask: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Inject network noise into flows X (N, L) at nominal SNR level s (Table 14).

    s in {0.1, 0.5, 1, 2, 5, 10}:
        alpha_loss = s, alpha_jit = s, alpha_reorder = 0.1 * s.
    """
    rng = np.random.default_rng(seed)
    n, target_len = X.shape
    X_noisy = np.zeros((n, target_len), dtype=np.float32)
    masks = np.zeros((n, target_len), dtype=np.float32) if return_mask else None

    alpha_loss = float(snr_level)
    alpha_jit = float(snr_level)
    alpha_reorder = 0.1 * float(snr_level)

    for i in range(n):
        flow = X[i]
        nz = np.flatnonzero(flow != 0)
        if len(nz) == 0:
            continue
        valid = flow[nz]
        t = np.abs(valid).astype(np.float64)
        d = np.sign(valid).astype(np.float32)

        # Ensure initial monotonicity on valid packets
        t = np.maximum.accumulate(t)

        # 1. Burst loss
        t, d = _apply_burst_loss(t, d, alpha_loss, p_gb, p_bg, p_drop, rng)
        # 2. Temporal jitter
        t = _apply_temporal_jitter(t, alpha_jit, sigma_ms, rho, p_tail, tail_scale_ms, rng)
        # 3. Local reordering
        t, d = _apply_local_reordering(t, d, alpha_reorder, p_swap, window, rng)

        keep = min(len(t), target_len)
        if keep > 0:
            X_noisy[i, :keep] = (t[:keep].astype(np.float32)) * d[:keep]
            if masks is not None:
                masks[i, :keep] = 1.0

    if return_mask:
        return X_noisy, masks
    return (X_noisy,)

