"""Defense simulators for Random dummy injection and RegulaTor (Appendix E, Table 13)."""

from bisect import bisect_right
from typing import Tuple
import numpy as np
from ..config import REGULATOR_CONFIGS


def simulate_random_defense(
    X: np.ndarray,
    y: np.ndarray,
    num_instances: int = 50000,
    max_rate: float = 0.20,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray]:
    """Appendix E: Inject dummy packets into D_CW-2tab at a random rate below 20%, yielding 50,000 instances."""
    rng = np.random.default_rng(seed)
    n_avail, seq_len = X.shape
    n_out = min(num_instances, n_avail)
    sel = rng.choice(n_avail, size=n_out, replace=False)

    X_sub = X[sel].astype(np.float32, copy=True)
    y_sub = y[sel].copy()
    X_def = np.zeros_like(X_sub, dtype=np.float32)

    for i in range(n_out):
        trace = X_sub[i]
        nz = np.flatnonzero(trace != 0)
        if len(nz) == 0:
            continue
        valid = trace[nz]
        rate = float(rng.uniform(0.01, max_rate))
        n_dummy = max(1, int(round(len(valid) * rate)))

        ts = np.abs(valid)
        t_min, t_max = float(ts.min()), float(ts.max())
        if t_max <= t_min:
            t_max = t_min + 1.0

        dummy_ts = rng.uniform(t_min, t_max, size=n_dummy)
        dummy_dir = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_dummy)
        dummy_vals = dummy_ts.astype(np.float32) * dummy_dir

        combined = np.concatenate([valid, dummy_vals])
        order = np.argsort(np.abs(combined), kind="mergesort")
        sorted_trace = combined[order]

        keep = min(len(sorted_trace), seq_len)
        X_def[i, :keep] = sorted_trace[:keep]

    return X_def, y_sub


def _regulator_single_trace(
    trace: np.ndarray,
    orig_rate: float,
    depreciation_rate: float,
    max_padding_budget: int,
    burst_threshold: float,
    upload_ratio: float,
    delay_cap: float,
    rng: np.random.Generator,
    seq_len: int,
) -> np.ndarray:
    """Simulate RegulaTor surge-shaped regularization on one trace (Holland & Hopper, 2020)."""
    nz = np.flatnonzero(trace != 0)
    if len(nz) == 0:
        return np.zeros(seq_len, dtype=np.float32)

    valid = trace[nz]
    dirs = np.sign(valid)
    ts = np.abs(valid).astype(np.float64)
    # Handle direction-only traces cleanly
    if ts.max() <= 1.0 and np.all(ts == 1.0):
        ts = np.arange(len(valid), dtype=np.float64) * 0.005

    ts = ts - ts[0]
    down_ts = ts[dirs < 0]
    up_ts = ts[dirs > 0]

    # Build download surge schedule (cells sent at exponentially decaying rate)
    budget = int(rng.integers(0, int(max_padding_budget) + 1)) if max_padding_budget > 0 else 0
    out_ts = []
    out_dir = []

    if len(down_ts) > 0:
        down_list = down_ts.tolist()
        t_cur = down_list[0]
        surge_start = t_cur
        rate = float(orig_rate)
        idx = 0
        n_down = len(down_list)
        pad_used = 0
        n_sent_in_surge = 0

        # Schedule download packets + dummy cells along surge intervals
        while idx < n_down and len(out_ts) < seq_len:
            elapsed = max(0.0, t_cur - surge_start)
            cur_rate = max(1.0, rate * (depreciation_rate ** elapsed))
            dt = 1.0 / cur_rate
            next_t = t_cur + dt

            # Count queued packets in O(log N)
            queued = bisect_right(down_list, t_cur + 1e-9, lo=idx) - idx

            if queued > burst_threshold * cur_rate * 0.1 and n_sent_in_surge > 10:
                surge_start = t_cur
                n_sent_in_surge = 0

            if down_list[idx] <= next_t:
                send_t = t_cur if t_cur > down_list[idx] else down_list[idx]
                out_ts.append(send_t)
                out_dir.append(-1.0)
                idx += 1
                n_sent_in_surge += 1
            elif pad_used < budget:
                out_ts.append(next_t)
                out_dir.append(-1.0)
                pad_used += 1
                n_sent_in_surge += 1
            else:
                next_t = down_list[idx]
                out_ts.append(next_t)
                out_dir.append(-1.0)
                idx += 1
                n_sent_in_surge += 1

            t_cur = next_t

    # Upload synchronization: fixed upload ratio to download rate with capped delay
    if len(up_ts) > 0:
        down_arr = np.sort(np.array(out_ts, dtype=np.float64)) if out_ts else np.array([0.0])
        # Sample upload slots at ratio 1 / upload_ratio of download times
        step = max(1, int(round(upload_ratio)))
        upload_slots = down_arr[::step]
        slot_ptr = 0
        for ut in up_ts:
            if len(out_ts) >= seq_len:
                break
            while slot_ptr < len(upload_slots) and upload_slots[slot_ptr] < ut:
                slot_ptr += 1
            if slot_ptr < len(upload_slots) and (upload_slots[slot_ptr] - ut) <= delay_cap:
                send_t = float(upload_slots[slot_ptr])
                slot_ptr += 1
            else:
                send_t = float(ut + min(delay_cap, 0.05))
            out_ts.append(send_t)
            out_dir.append(1.0)

    if not out_ts:
        return np.zeros(seq_len, dtype=np.float32)

    out_ts_arr = np.array(out_ts, dtype=np.float64)
    out_dir_arr = np.array(out_dir, dtype=np.float32)
    order = np.argsort(out_ts_arr, kind="mergesort")
    signed = (out_ts_arr[order] + 1e-6).astype(np.float32) * out_dir_arr[order]

    res = np.zeros(seq_len, dtype=np.float32)
    keep = min(len(signed), seq_len)
    res[:keep] = signed[:keep]
    return res


def simulate_regulator(
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "light",
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate RegulaTor-Light or RegulaTor-Heavy defended dataset from D_CW-2tab (Table 13)."""
    mode_key = mode.lower()
    if mode_key not in REGULATOR_CONFIGS:
        raise ValueError(f"Unknown RegulaTor mode '{mode}'. Choose 'light' or 'heavy'.")
    cfg = REGULATOR_CONFIGS[mode_key]
    rng = np.random.default_rng(seed)

    n, seq_len = X.shape
    X_out = np.zeros((n, seq_len), dtype=np.float32)
    for i in range(n):
        X_out[i] = _regulator_single_trace(
            X[i],
            orig_rate=cfg["orig_rate"],
            depreciation_rate=cfg["depreciation_rate"],
            max_padding_budget=int(cfg["max_padding_budget"]),
            burst_threshold=cfg["burst_threshold"],
            upload_ratio=cfg["upload_ratio"],
            delay_cap=cfg["delay_cap"],
            rng=rng,
            seq_len=seq_len,
        )
    return X_out, y.copy()

