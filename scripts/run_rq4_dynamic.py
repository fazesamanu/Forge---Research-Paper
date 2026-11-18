"""RQ4: Dynamic Settings Evaluation (Table 5, Figure 8)."""

import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score

from forge.config import ForgeConfig
from forge.model import FORGE
from forge.data.dataset import (
    BatchTensorLoader,
    build_dynamic_defense_dataset,
    build_dynamic_multitab_dataset,
    get_dataloaders,
    load_npz_data,
    split_dataset,
)
from forge.trainer import predict_probs, set_seed, train_and_evaluate
from scripts.common import (
    ALL_MODELS,
    build_model_by_name,
    load_json_if_exists,
    resolve_dataset_path,
    save_json,
)


def eval_cross_tab_roc(
    data_dir: Path,
    ckpt_dir: Path,
    results_dir: Path,
    seq_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    max_samples: int = None,
):
    """Reproduce Figure 8: cross-tab ROC curves when training and testing tab counts differ."""
    pairs = [
        (3, 3),
        (5, 3),
        (5, 5),
        (3, 5),
        (4, 3),
        (4, 4),
        (3, 4),
    ]
    roc_data = {}

    # Load test splits for tabs 3, 4, 5
    test_loaders = {}
    for t in (3, 4, 5):
        p = resolve_dataset_path(data_dir, f"Closed_{t}tab.npz")
        X, y, n_cls = load_npz_data(p, seq_len=seq_len, max_samples=max_samples, seed=seed)
        _, _, (X_te, y_te) = split_dataset(X, y, seed=seed)
        test_loaders[t] = (BatchTensorLoader(X_te, y_te, batch_size=batch_size, shuffle=False), n_cls)
        del X, y

    for train_t in (3, 4, 5):
        ckpt_rq1 = Path("checkpoints/rq1") / f"{train_t}tab_FORGE.pt"
        ckpt_local = ckpt_dir / f"cross_{train_t}tab_FORGE.pt"
        ckpt_use = ckpt_rq1 if ckpt_rq1.exists() else ckpt_local

        n_cls = test_loaders[train_t][1]
        model = FORGE(ForgeConfig(num_classes=n_cls, seq_len=seq_len))

        if ckpt_use.exists():
            model.load_state_dict(torch.load(ckpt_use, map_location="cpu", weights_only=True))
            model = model.to(device)
        else:
            p = resolve_dataset_path(data_dir, f"Closed_{train_t}tab.npz")
            X, y, _ = load_npz_data(p, seq_len=seq_len, max_samples=max_samples, seed=seed)
            loaders = get_dataloaders(X, y, batch_size=batch_size, seed=seed)
            set_seed(seed)
            train_and_evaluate(
                model,
                loaders,
                epochs=epochs,
                lr=lr,
                k_list=(train_t,),
                select_metric=f"map@{train_t}",
                checkpoint_path=ckpt_local,
                device=device,
                tag=f"RQ4-cross-train{train_t}",
            )
            del X, y, loaders

        for tr_t, te_t in pairs:
            if tr_t != train_t:
                continue
            loader, _ = test_loaders[te_t]
            y_true, y_prob = predict_probs(model, loader, device)
            fpr, tpr, _ = roc_curve(y_true.ravel(), y_prob.ravel())
            auc_val = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))
            # Subsample ROC curve points for compact JSON storage
            step = max(1, len(fpr) // 400)
            key = f"train{tr_t}test{te_t}"
            roc_data[key] = {
                "auc": auc_val,
                "fpr": fpr[::step].tolist(),
                "tpr": tpr[::step].tolist(),
            }
            print(f"[RQ4-CrossTab] {key}: micro-AUC={auc_val:.4f}")

    save_json(results_dir / "rq4_crosstab_roc.json", roc_data)
    return roc_data


def main():
    parser = argparse.ArgumentParser(description="RQ4: Dynamic mix & cross-tab experiments.")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/rq4")
    parser.add_argument("--models", type=str, nargs="+", default=ALL_MODELS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scale", type=float, default=1.0, help="Dataset fraction for smoke tests")
    parser.add_argument("--skip_crosstab", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.ckpt_dir)

    summary = load_json_if_exists(results_dir / "rq4_summary.json") or {}

    # 1. D_mix-num-tabs
    summary.setdefault("D_mix-num-tabs", {})
    if not (args.resume and all((results_dir / f"rq4_mix_tabs_{m}.json").exists() for m in args.models)):
        X_mt, y_mt, n_cls_mt = build_dynamic_multitab_dataset(
            data_dir,
            seq_len=args.seq_len,
            n_closed=int(100000 * args.scale),
            n_open=int(20000 * args.scale),
            seed=args.seed,
        )
        loaders_mt = get_dataloaders(X_mt, y_mt, batch_size=args.batch_size, seed=args.seed)
        for model_name in args.models:
            res_path = results_dir / f"rq4_mix_tabs_{model_name}.json"
            if args.resume and res_path.exists():
                summary["D_mix-num-tabs"][model_name] = load_json_if_exists(res_path)
                continue
            set_seed(args.seed)
            model = build_model_by_name(model_name, num_classes=n_cls_mt, seq_len=args.seq_len)
            metrics = train_and_evaluate(
                model,
                loaders_mt,
                epochs=args.epochs,
                lr=args.lr,
                k_list=(2, 5),
                select_metric="auc",
                checkpoint_path=ckpt_dir / f"mix_tabs_{model_name}.pt",
                results_path=res_path,
                device=device,
                tag=f"RQ4-MixTabs-{model_name}",
            )
            summary["D_mix-num-tabs"][model_name] = metrics
            save_json(results_dir / "rq4_summary.json", summary)
        del X_mt, y_mt, loaders_mt
    else:
        for m in args.models:
            summary["D_mix-num-tabs"][m] = load_json_if_exists(results_dir / f"rq4_mix_tabs_{m}.json")

    # 2. D_mix-defenses
    summary.setdefault("D_mix-defenses", {})
    if not (args.resume and all((results_dir / f"rq4_mix_def_{m}.json").exists() for m in args.models)):
        X_md, y_md, n_cls_md = build_dynamic_defense_dataset(
            data_dir,
            seq_len=args.seq_len,
            per_defense=int(15000 * args.scale),
            seed=args.seed,
        )
        loaders_md = get_dataloaders(X_md, y_md, batch_size=args.batch_size, seed=args.seed)
        for model_name in args.models:
            res_path = results_dir / f"rq4_mix_def_{model_name}.json"
            if args.resume and res_path.exists():
                summary["D_mix-defenses"][model_name] = load_json_if_exists(res_path)
                continue
            set_seed(args.seed)
            model = build_model_by_name(model_name, num_classes=n_cls_md, seq_len=args.seq_len)
            metrics = train_and_evaluate(
                model,
                loaders_md,
                epochs=args.epochs,
                lr=args.lr,
                k_list=(2,),
                select_metric="auc",
                checkpoint_path=ckpt_dir / f"mix_def_{model_name}.pt",
                results_path=res_path,
                device=device,
                tag=f"RQ4-MixDef-{model_name}",
            )
            summary["D_mix-defenses"][model_name] = metrics
            save_json(results_dir / "rq4_summary.json", summary)
        del X_md, y_md, loaders_md
    else:
        for m in args.models:
            summary["D_mix-defenses"][m] = load_json_if_exists(results_dir / f"rq4_mix_def_{m}.json")

    # 3. Cross-tab mismatched evaluation (Figure 8)
    if not args.skip_crosstab:
        max_samp = int(58000 * args.scale) if args.scale < 1.0 else None
        eval_cross_tab_roc(
            data_dir=data_dir,
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            device=device,
            max_samples=max_samp,
        )


if __name__ == "__main__":
    main()

