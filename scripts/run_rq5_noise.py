"""RQ5: Evaluation Against Network Noise (Figure 9, Table 12, Appendix F)."""

import argparse
from pathlib import Path
import torch

from forge.data.dataset import BatchTensorLoader, get_dataloaders, load_npz_data, split_dataset
from forge.simulators.noise import inject_network_noise
from forge.trainer import evaluate_model, set_seed, train_and_evaluate
from scripts.common import (
    ALL_MODELS,
    build_model_by_name,
    load_json_if_exists,
    resolve_dataset_path,
    save_json,
)


def main():
    parser = argparse.ArgumentParser(description="RQ5: Network noise robustness evaluation.")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/rq1")
    parser.add_argument("--models", type=str, nargs="+", default=ALL_MODELS)
    parser.add_argument("--snr_levels", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
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

    cw2_path = resolve_dataset_path(data_dir, "Closed_2tab.npz")
    X, y, num_classes = load_npz_data(
        cw2_path, seq_len=args.seq_len, max_samples=args.max_samples, seed=args.seed
    )
    (_, _), (_, _), (X_test, y_test) = split_dataset(X, y, seed=args.seed)

    # Ensure 2-tab checkpoints exist (reuse RQ1 checkpoints when available)
    models_loaded = {}
    loaders_clean = None
    for model_name in args.models:
        ckpt_path = ckpt_dir / f"2tab_{model_name}.pt"
        model = build_model_by_name(model_name, num_classes=num_classes, seq_len=args.seq_len)
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
            model = model.to(device)
        else:
            if loaders_clean is None:
                loaders_clean = get_dataloaders(X, y, batch_size=args.batch_size, seed=args.seed)
            set_seed(args.seed)
            train_and_evaluate(
                model,
                loaders_clean,
                epochs=args.epochs,
                lr=args.lr,
                k_list=(2,),
                select_metric="map@2",
                checkpoint_path=ckpt_path,
                device=device,
                tag=f"RQ5-base-{model_name}",
            )
        models_loaded[model_name] = model

    del X, y, loaders_clean

    summary = load_json_if_exists(results_dir / "rq5_summary.json") or {}

    for snr in args.snr_levels:
        snr_key = str(int(snr)) if snr == int(snr) else str(snr)
        summary.setdefault(snr_key, {})

        if args.resume and all(m in summary[snr_key] for m in args.models):
            print(f"[RESUME] Skipping SNR={snr_key}")
            continue

        if snr == 0.0:
            X_eval = X_test
        else:
            print(f"\n=== Injecting network noise at SNR level s={snr_key} ===")
            (X_eval,) = inject_network_noise(X_test, snr_level=snr, seed=args.seed)

        eval_loader = BatchTensorLoader(
            X_eval, y_test, batch_size=args.batch_size, shuffle=False
        )

        for model_name, model in models_loaded.items():
            metrics = evaluate_model(model, eval_loader, device, k_list=(2,))
            summary[snr_key][model_name] = metrics
            print(
                f"[RQ5 s={snr_key}] {model_name:8s} | AUC={metrics['auc']:.4f} | "
                f"P@2={metrics['p@2']:.4f} | MAP@2={metrics['map@2']:.4f}"
            )
        save_json(results_dir / "rq5_summary.json", summary)


if __name__ == "__main__":
    main()

