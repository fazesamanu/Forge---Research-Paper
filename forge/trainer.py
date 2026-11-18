"""GPU/bfloat16-AMP training and evaluation engine for FORGE and WF baselines."""

import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_all_metrics, find_best_threshold


def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over loader and return (y_true, y_prob)."""
    model.eval()
    all_y = []
    all_p = []
    amp_enabled = use_amp and (device.type == "cuda")

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits = model(xb)
            probs = torch.sigmoid(logits)
        all_y.append(yb.numpy())
        all_p.append(probs.float().cpu().numpy())

    return np.concatenate(all_y, axis=0), np.concatenate(all_p, axis=0)


def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    k_list: Iterable[int] = (2, 3, 4, 5),
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Dict[str, float]:
    y_true, y_prob = predict_probs(model, loader, device, use_amp=use_amp)
    return compute_all_metrics(y_true, y_prob, k_list=k_list, threshold=threshold)


def train_and_evaluate(
    model: nn.Module,
    loaders: Dict,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: Optional[torch.Tensor] = None,
    k_list: Iterable[int] = (2, 3, 4, 5),
    select_metric: str = "map@2",
    checkpoint_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    patience: int = 8,
    use_amp: bool = True,
    tag: str = "",
) -> Dict[str, float]:
    """Train model with bfloat16 AMP + BCEWithLogitsLoss, tune threshold on val, evaluate on test."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    amp_enabled = use_amp and (device.type == "cuda")

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_score = -1.0
    best_state = None
    no_improve = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for xb, yb in loaders["train"]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(xb)
                loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, n_batches)

        val_metrics = evaluate_model(
            model, loaders["val"], device, k_list=k_list, use_amp=use_amp
        )
        score = val_metrics.get(select_metric, val_metrics["auc"])

        improved = score > best_score
        if improved:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            no_improve += 1

        print(
            f"[{tag}] Epoch {epoch:02d}/{epochs:02d} | loss={avg_loss:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | val_{select_metric}={score:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} {'*' if improved else ''}",
            flush=True,
        )

        if no_improve >= patience:
            print(f"[{tag}] Early stopping at epoch {epoch} (best {select_metric}={best_score:.4f})", flush=True)
            break

    train_time = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    # Tune decision threshold tau on validation predictions per Appendix C (Eq. 10)
    y_val_true, y_val_prob = predict_probs(model, loaders["val"], device, use_amp=use_amp)
    best_thr, _ = find_best_threshold(y_val_true, y_val_prob)

    test_metrics = evaluate_model(
        model, loaders["test"], device, k_list=k_list, threshold=best_thr, use_amp=use_amp
    )
    test_metrics["train_time_sec"] = round(train_time, 2)
    test_metrics["best_val_" + select_metric] = best_score

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    print(
        f"[{tag}] TEST | thr={best_thr:.2f} | AUC={test_metrics['auc']:.4f} | "
        f"Pre={test_metrics['precision']:.4f} | Rec={test_metrics['recall']:.4f} | "
        f"F1={test_metrics['f1']:.4f} | {select_metric}={test_metrics.get(select_metric, 0.0):.4f} | "
        f"time={train_time:.1f}s",
        flush=True,
    )
    return test_metrics
