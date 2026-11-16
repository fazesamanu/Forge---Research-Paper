"""RQ3: Evaluation Against WF Defenses (Table 4, Figure 7, Table 11)."""

import argparse
from pathlib import Path
import numpy as np
import torch

from forge.data.dataset import load_npz_data, get_dataloaders
from forge.simulators.defenses import simulate_random_defense, simulate_regulator
from forge.trainer import set_seed, train_and_evaluate
from scripts.common import (
    ALL_MODELS,
    build_model_by_name,
    load_json_if_exists,
    resolve_dataset_path,
    save_json,
)


def prepare_simulated_defense(
    name: str, data_dir: Path, gen_dir: Path, seq_len: int, seed: int, max_samples: int = None
):
    gen_dir.mkdir(parents=True, exist_ok=True)
    cache_path = gen_dir / f"{name}.npz"
    if cache_path.exists():
        return load_npz_data(cache_path, seq_len=seq_len, max_samples=max_samples, seed=seed)

    cw2_path = resolve_dataset_path(data_dir, "Closed_2tab.npz")
    X_base, y_base, n_cls = load_npz_data(cw2_path, seq_len=seq_len, max_samples=max_samples, seed=seed)

    if name == "Random":
        n_inst = min(50000, len(X_base))
        X_def, y_def = simulate_random_defense(X_base, y_base, num_instances=n_inst, seed=seed)
    elif name == "RegularTor-Light":
        X_def, y_def = simulate_regulator(X_base, y_base, mode="light", seed=seed)
    elif name == "RegularTor-Heavy":
        X_def, y_def = simulate_regulator(X_base, y_base, mode="heavy", seed=seed)
    else:
        raise ValueError(f"Unknown simulated defense: {name}")

    np.savez_compressed(cache_path, X=X_def, y=y_def)
    return X_def, y_def, n_cls


def main():
    parser = argparse.ArgumentParser(description="RQ3: Evaluation against WF defenses.")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/rq3")
    parser.add_argument(
        "--defenses",
        type=str,
        nargs="+",
        default=[
            "Random",
            "Front",
            "WTF-PAD",
            "RegularTor-Light",
            "RegularTor-Heavy",
            "Walkie-Talkie",
            "TrafficSliver",
        ],
    )
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
    gen_dir = data_dir / "generated"
    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.ckpt_dir)

    summary = load_json_if_exists(results_dir / "rq3_summary.json") or {}

    for defense in args.defenses:
        summary.setdefault(defense, {})
        if args.resume and all(
            (results_dir / f"rq3_{defense}_{m}.json").exists() for m in args.models
        ):
            for m in args.models:
                summary[defense][m] = load_json_if_exists(results_dir / f"rq3_{defense}_{m}.json")
            continue

        print(f"\n=== Preparing Defense Dataset: {defense} ===")
        if defense in ("Random", "RegularTor-Light", "RegularTor-Heavy"):
            X, y, num_classes = prepare_simulated_defense(
                defense, data_dir, gen_dir, args.seq_len, args.seed, max_samples=args.max_samples
            )
        else:
            ds_path = resolve_dataset_path(data_dir, f"{defense}.npz")
            X, y, num_classes = load_npz_data(
                ds_path, seq_len=args.seq_len, max_samples=args.max_samples, seed=args.seed
            )

        is_single_tab = defense in ("Walkie-Talkie", "TrafficSliver")
        sel_metric = "auc" if is_single_tab else "map@2"
        loaders = get_dataloaders(X, y, batch_size=args.batch_size, seed=args.seed)

        for model_name in args.models:
            res_path = results_dir / f"rq3_{defense}_{model_name}.json"
            if args.resume and res_path.exists():
                print(f"[RESUME] Skipping {defense} / {model_name}")
                summary[defense][model_name] = load_json_if_exists(res_path)
                continue

            set_seed(args.seed)
            model = build_model_by_name(model_name, num_classes=num_classes, seq_len=args.seq_len)
            metrics = train_and_evaluate(
                model=model,
                loaders=loaders,
                epochs=args.epochs,
                lr=args.lr,
                k_list=(1, 2) if is_single_tab else (2,),
                select_metric=sel_metric,
                checkpoint_path=ckpt_dir / f"{defense}_{model_name}.pt",
                results_path=res_path,
                device=device,
                tag=f"RQ3-{defense}-{model_name}",
            )
            summary[defense][model_name] = metrics
            save_json(results_dir / "rq3_summary.json", summary)

        del X, y, loaders


if __name__ == "__main__":
    main()

