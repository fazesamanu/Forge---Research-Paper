"""Dataset loading, chunked .npz streaming, batch tensor loader, and dynamic corpus construction."""

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.lib.format as npy_fmt
import torch


def _align_numpy_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Truncate or right-pad 2D array X (N, L) to (N, seq_len) in float32."""
    n, cur_len = X.shape
    if cur_len == seq_len and X.dtype == np.float32:
        return X
    out = np.zeros((n, seq_len), dtype=np.float32)
    copy_len = min(cur_len, seq_len)
    out[:, :copy_len] = X[:, :copy_len]
    return out


def encode_multihot_labels(
    y: np.ndarray, num_classes: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Convert raw WFlib labels (1D indices, 2D tab-index matrix, or 2D multi-hot) to float32 multi-hot."""
    y = np.asarray(y)
    if y.ndim == 2:
        uniq = np.unique(y[: min(len(y), 200)])
        if y.shape[1] >= 20 and np.all(np.isin(uniq, [0, 1])):
            n_cls = num_classes if num_classes is not None else y.shape[1]
            if y.shape[1] == n_cls:
                return y.astype(np.float32), n_cls
            out = np.zeros((y.shape[0], n_cls), dtype=np.float32)
            c_min = min(y.shape[1], n_cls)
            out[:, :c_min] = y[:, :c_min]
            return out, n_cls
        max_lbl = int(np.max(y))
        n_cls = num_classes if num_classes is not None else (max_lbl + 1)
        out = np.zeros((y.shape[0], n_cls), dtype=np.float32)
        for i in range(y.shape[0]):
            for lbl in y[i]:
                idx = int(lbl)
                if 0 <= idx < n_cls:
                    out[i, idx] = 1.0
                elif idx < 0:
                    out[i, n_cls - 1] = 1.0
        return out, n_cls

    if y.ndim == 1 and y.dtype != object:
        max_lbl = int(np.max(y))
        has_neg = bool(np.any(y < 0))
        n_cls = num_classes if num_classes is not None else (max_lbl + 1 + (1 if has_neg else 0))
        out = np.zeros((y.shape[0], n_cls), dtype=np.float32)
        idx = np.where(y >= 0, y, n_cls - 1)
        valid = (idx >= 0) & (idx < n_cls)
        out[np.arange(y.shape[0])[valid], idx[valid]] = 1.0
        return out, n_cls

    inferred_max = 0
    for row in y:
        if len(row) > 0:
            inferred_max = max(inferred_max, int(np.max(row)))
    n_cls = num_classes if num_classes is not None else (inferred_max + 1)
    out = np.zeros((len(y), n_cls), dtype=np.float32)
    for i, row in enumerate(y):
        for lbl in row:
            idx = int(lbl)
            if 0 <= idx < n_cls:
                out[i, idx] = 1.0
            elif idx < 0:
                out[i, n_cls - 1] = 1.0
    return out, n_cls


def _stream_2d_array_from_npz(
    npz_path: Path,
    key_candidates: Tuple[str, ...],
    seq_len: int,
    chunk_rows: int = 4096,
) -> Optional[np.ndarray]:
    """Stream a 2D numeric .npy entry from an .npz zip archive into (N, seq_len) float32."""
    with zipfile.ZipFile(npz_path, "r") as zf:
        names = zf.namelist()
        target = None
        for cand in key_candidates:
            if f"{cand}.npy" in names:
                target = f"{cand}.npy"
                break
        if target is None:
            return None

        with zf.open(target, "r") as raw_fp:
            fp = io.BufferedReader(raw_fp, buffer_size=4 * 1024 * 1024)
            ver = npy_fmt.read_magic(fp)
            if ver == (1, 0):
                shape, fortran, dtype = npy_fmt.read_array_header_1_0(fp)
            else:
                shape, fortran, dtype = npy_fmt.read_array_header_2_0(fp)

            if fortran or len(shape) != 2 or dtype.kind not in ("f", "i", "u"):
                return None

            n_rows, cur_len = shape
            copy_len = min(cur_len, seq_len)
            out = np.zeros((n_rows, seq_len), dtype=np.float32)
            row_bytes = cur_len * dtype.itemsize

            row_idx = 0
            while row_idx < n_rows:
                rows_to_read = min(chunk_rows, n_rows - row_idx)
                buf = fp.read(rows_to_read * row_bytes)
                if not buf:
                    break
                actual_rows = len(buf) // row_bytes
                chunk = np.frombuffer(buf[: actual_rows * row_bytes], dtype=dtype).reshape(
                    actual_rows, cur_len
                )
                out[row_idx : row_idx + actual_rows, :copy_len] = chunk[:, :copy_len]
                row_idx += actual_rows
            return out


def load_npz_data(
    npz_path: Union[str, Path],
    seq_len: int = 8000,
    num_classes: Optional[int] = None,
    max_samples: Optional[int] = None,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load WFlib .npz dataset -> (X_float32, y_multihot_float32, num_classes) with low peak RAM."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    X_aligned = _stream_2d_array_from_npz(npz_path, ("X", "x"), seq_len=seq_len)

    with np.load(npz_path, allow_pickle=True) as data:
        keys = list(data.keys())
        y_key = "y" if "y" in keys else ("Y" if "Y" in keys else keys[1])
        y = data[y_key]
        if X_aligned is None:
            x_key = "X" if "X" in keys else ("x" if "x" in keys else keys[0])
            X_aligned = _align_numpy_sequences(data[x_key], seq_len=seq_len)

    if max_samples is not None and len(X_aligned) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_aligned), size=max_samples, replace=False)
        idx.sort()
        X_aligned = X_aligned[idx]
        y = y[idx]

    y_mh, n_cls = encode_multihot_labels(y, num_classes=num_classes)
    return X_aligned, y_mh, n_cls


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1234,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split (X, y) into (train, val, test) with deterministic shuffling."""
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]
    return (X[idx_train], y[idx_train]), (X[idx_val], y[idx_val]), (X[idx_test], y[idx_test])


class BatchTensorLoader:
    """Fast vectorized batch slicer over CPU tensors (eliminates per-sample collate overhead)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        shuffle: bool = False,
        seed: int = 1234,
    ):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.n = self.X.size(0)
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.n, generator=self.gen)
            for start in range(0, self.n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                yield self.X[idx], self.y[idx]
        else:
            for start in range(0, self.n, self.batch_size):
                end = min(start + self.batch_size, self.n)
                yield self.X[start:end], self.y[start:end]


WFDataset = BatchTensorLoader


def get_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1234,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Dict[str, BatchTensorLoader]:
    """Create fast train/val/test BatchTensorLoaders from (X, y)."""
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_dataset(
        X, y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    return {
        "train": BatchTensorLoader(X_tr, y_tr, batch_size=batch_size, shuffle=True, seed=seed),
        "val": BatchTensorLoader(X_va, y_va, batch_size=batch_size, shuffle=False, seed=seed),
        "test": BatchTensorLoader(X_te, y_te, batch_size=batch_size, shuffle=False, seed=seed),
    }


def build_dynamic_multitab_dataset(
    data_dir: Union[str, Path],
    seq_len: int = 8000,
    n_closed: int = 100000,
    n_open: int = 20000,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Construct D_mix-num-tabs per Appendix B.4:
    100,000 closed-world + 20,000 open-world with tab-count distribution 40%/30%/20%/10% for 2/3/4/5 tabs.
    """
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)
    ratios = {2: 0.40, 3: 0.30, 4: 0.20, 5: 0.10}

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    target_classes = 101

    for tab, r in ratios.items():
        c_count = int(round(n_closed * r))
        c_path = _resolve_case_insensitive(data_dir, f"Closed_{tab}tab.npz")
        if c_path is not None and c_count > 0:
            Xc, yc, _ = load_npz_data(c_path, seq_len=seq_len, num_classes=target_classes, seed=seed + tab)
            idx = rng.choice(len(Xc), size=min(c_count, len(Xc)), replace=False)
            xs.append(Xc[idx])
            ys.append(yc[idx])
            del Xc, yc

        o_count = int(round(n_open * r))
        o_path = _resolve_case_insensitive(data_dir, f"Open_{tab}tab.npz")
        if o_path is not None and o_count > 0:
            Xo, yo, _ = load_npz_data(o_path, seq_len=seq_len, num_classes=target_classes, seed=seed + 10 + tab)
            idx = rng.choice(len(Xo), size=min(o_count, len(Xo)), replace=False)
            xs.append(Xo[idx])
            ys.append(yo[idx])
            del Xo, yo

    X_mix = np.concatenate(xs, axis=0)
    y_mix = np.concatenate(ys, axis=0)
    del xs, ys
    perm = rng.permutation(len(X_mix))
    return X_mix[perm], y_mix[perm], target_classes


def build_dynamic_defense_dataset(
    data_dir: Union[str, Path],
    seq_len: int = 8000,
    per_defense: int = 15000,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Construct D_mix-defenses per Appendix B.4:
    15,000 WTF-PAD + 15,000 Front = 30,000 instances sharing the same monitoring set.
    """
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)

    w_path = _resolve_case_insensitive(data_dir, "WTF-PAD.npz")
    f_path = _resolve_case_insensitive(data_dir, "Front.npz")
    if w_path is None or f_path is None:
        raise FileNotFoundError("WTF-PAD.npz and Front.npz are required for D_mix-defenses")

    Xw, yw, nw = load_npz_data(w_path, seq_len=seq_len, seed=seed)
    iw = rng.choice(len(Xw), size=min(per_defense, len(Xw)), replace=False)
    Xw_sub, yw_sub = Xw[iw], yw[iw]
    del Xw, yw

    Xf, yf, _ = load_npz_data(f_path, seq_len=seq_len, num_classes=nw, seed=seed + 1)
    i_f = rng.choice(len(Xf), size=min(per_defense, len(Xf)), replace=False)
    Xf_sub, yf_sub = Xf[i_f], yf[i_f]
    del Xf, yf

    X_mix = np.concatenate([Xw_sub, Xf_sub], axis=0)
    y_mix = np.concatenate([yw_sub, yf_sub], axis=0)
    perm = rng.permutation(len(X_mix))
    return X_mix[perm], y_mix[perm], nw


def _resolve_case_insensitive(data_dir: Path, name: str) -> Optional[Path]:
    direct = data_dir / name
    if direct.exists():
        return direct
    target = name.lower()
    if data_dir.exists():
        for p in data_dir.iterdir():
            if p.name.lower() == target:
                return p
    return None
