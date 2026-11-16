"""RQ2: Open-world Multi-tab Evaluation (Table 3, Figure 6)."""

import argparse
from pathlib import Path
import torch

from forge.data.dataset import load_npz_data, get_dataloaders
from forge.trainer import set_seed, train_and_evaluate
from scripts.common import (
    ALL_MODELS,
    build_model_by_name,
    load_json_if_exists,
    resolve_dataset_path,
    save_json,
)


def main():
    parser = argparse.ArgumentParser(description="RQ2: Open-world multi-tab experiments.")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/rq2")
    parser.add_argument("--tabs", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--models", type=str, nargs="+", default=ALL_MODELS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.ckpt_dir)

    summary = load_json_if_exists(results_dir / "rq2_summary.json") or {}

    for tab in args.tabs:
        ds_name = f"Open_{tab}tab.npz"
        ds_key = f"D_OW-{tab}tab"
        summary.setdefault(ds_key, {})

        if args.resume and all(
            (results_dir / f"rq2_{tab}tab_{m}.json").exists() for m in args.models
        ):
            for m in args.models:
                summary[ds_key][m] = load_json_if_exists(results_dir / f"rq2_{tab}tab_{m}.json")
            continue

        ds_path = resolve_dataset_path(data_dir, ds_name)
        print(f"\n=== Loading {ds_path} ===")
        X, y, num_classes = load_npz_data(
            ds_path, seq_len=args.seq_len, max_samples=args.max_samples, seed=args.seed
        )
        loaders = get_dataloaders(X, y, batch_size=args.batch_size, seed=args.seed)

        for model_name in args.models:
            res_path = results_dir / f"rq2_{tab}tab_{model_name}.json"
            if args.resume and res_path.exists():
                print(f"[RESUME] Skipping {ds_key} / {model_name}")
                summary[ds_key][model_name] = load_json_if_exists(res_path)
                continue

            set_seed(args.seed)
            model = build_model_by_name(model_name, num_classes=num_classes, seq_len=args.seq_len)
            metrics = train_and_evaluate(
                model=model,
                loaders=loaders,
                epochs=args.epochs,
                lr=args.lr,
                k_list=(2, 3, 4, 5),
                select_metric=f"map@{tab}",
                checkpoint_path=ckpt_dir / f"{tab}tab_{model_name}.pt",
                results_path=res_path,
                device=device,
                tag=f"RQ2-{tab}tab-{model_name}",
            )
            summary[ds_key][model_name] = metrics
            save_json(results_dir / "rq2_summary.json", summary)

        del X, y, loaders


if __name__ == "__main__":
    main()

