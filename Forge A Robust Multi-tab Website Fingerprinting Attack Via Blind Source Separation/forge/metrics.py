"""Evaluation metrics for multi-label and thresholded WF classification (Section 5.1.3 & Appendix C)."""

from typing import Dict, Iterable, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """P@k (Eq. 5): fraction of true labels in top-k predicted probabilities."""
    k_eff = min(k, y_prob.shape[1])
    top_k_idx = np.argpartition(-y_prob, kth=k_eff - 1, axis=1)[:, :k_eff]
    hits = np.take_along_axis(y_true, top_k_idx, axis=1).sum(axis=1)
    return float(np.mean(hits / float(k)))


def map_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """MAP@k (Eq. 6): mean of P@i for i = 1..k."""
    _, num_classes = y_prob.shape
    k_eff = min(k, num_classes)
    order = np.argsort(-y_prob, axis=1)[:, :k_eff]
    ranked_rel = np.take_along_axis(y_true, order, axis=1)
    cum_hits = np.cumsum(ranked_rel, axis=1)
    ranks = np.arange(1, k_eff + 1, dtype=np.float64)[None, :]
    p_at_i = cum_hits / ranks
    return float(np.mean(np.mean(p_at_i, axis=1)))


def ir_ap_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Standard Information-Retrieval AP@k: (1 / min(k, |y|)) * sum_{i=1}^k P@i * rel_i."""
    _, num_classes = y_prob.shape
    k_eff = min(k, num_classes)
    order = np.argsort(-y_prob, axis=1)[:, :k_eff]
    ranked_rel = np.take_along_axis(y_true, order, axis=1)
    cum_hits = np.cumsum(ranked_rel, axis=1)
    ranks = np.arange(1, k_eff + 1, dtype=np.float64)[None, :]
    p_at_i = cum_hits / ranks
    denom = np.minimum(k_eff, np.maximum(y_true.sum(axis=1), 1.0))
    ap = (p_at_i * ranked_rel).sum(axis=1) / denom
    return float(np.mean(ap))


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro-averaged ROC-AUC across all classes with both positive and negative examples."""
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if np.any(col == 1) and np.any(col == 0):
            aucs.append(roc_auc_score(col, y_prob[:, c]))
    if not aucs:
        return 0.5
    return float(np.mean(aucs))


def threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Precision, Recall, and F1-score from global TP, FP, FN (Section 5.1.3 & Eq. 10)."""
    y_pred = (y_prob >= threshold).astype(np.float64)
    y_t = (y_true > 0.5).astype(np.float64)
    tp = float(np.sum((y_pred == 1) & (y_t == 1)))
    fp = float(np.sum((y_pred == 1) & (y_t == 0)))
    fn = float(np.sum((y_pred == 0) & (y_t == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Scan thresholds in [0.10, 0.85] on validation probabilities to maximize F1 (Appendix C, Eq. 10)."""
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    best_t = 0.5
    best_m = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for t in np.linspace(0.10, 0.85, 16):
        m = threshold_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_m["f1"]:
            best_m = m
            best_t = float(t)
    return best_t, best_m


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_list: Iterable[int] = (2, 3, 4, 5),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute complete suite of multi-label and single-label metrics."""
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    res: Dict[str, float] = {}
    res["auc"] = macro_auc(y_true, y_prob)
    res.update(threshold_metrics(y_true, y_prob, threshold=threshold))
    res["threshold"] = float(threshold)
    for k in k_list:
        res[f"p@{k}"] = precision_at_k(y_true, y_prob, k)
        res[f"ap@{k}"] = ir_ap_at_k(y_true, y_prob, k)
        res[f"map@{k}"] = map_at_k(y_true, y_prob, k)
    return res
