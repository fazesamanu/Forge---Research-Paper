"""RQ6: Ablation Study (Table 6) and Parameter Sensitivity Analysis (Figure 10)."""

import argparse
from pathlib import Path
import torch

from forge.config import ForgeConfig
from forge.model import FORGE
from forge.data.dataset import load_npz_data, get_dataloaders
from forge.trainer import set_seed, train_and_evaluate
from scripts.common import load_json_if_exists, resolve_dataset_path, save_json


def main():
    parser = argparse.ArgumentParser(description="RQ6: Ablation study and sensitivity analysis.")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/rq6")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_sensitivity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.ckpt_dir)
    cw2_path = resolve_dataset_path(data_dir, "Closed_2tab.npz")

    # 1. Ablation Study (Table 6) on D_CW-2tab with l=8000, tau=20
    ablation_summary = load_json_if_exists(results_dir / "rq6_ablation.json") or {}
    rq1_2tab_forge = load_json_if_exists(results_dir / "rq1_2tab_FORGE.json")
    if rq1_2tab_forge is not None and "Case4_Full_FORGE" not in ablation_summary:
        ablation_summary["Case4_Full_FORGE"] = rq1_2tab_forge
        save_json(results_dir / "rq6_ablation.json", ablation_summary)
    ablation_cases = {
        "Case1_no_TDE": (False, True, True),
        "Case2_no_BSS": (True, False, True),
        "Case3_no_Identifier": (True, True, False),
        "Case4_Full_FORGE": (True, True, True),
    }

    if not (args.resume and all(k in ablation_summary for k in ablation_cases)):
        X8k, y8k, n_cls = load_npz_data(
            cw2_path, seq_len=8000, max_samples=args.max_samples, seed=args.seed
        )
        loaders8k = get_dataloaders(X8k, y8k, batch_size=args.batch_size, seed=args.seed)

        for case_name, (u_tde, u_bss, u_id) in ablation_cases.items():
            if args.resume and case_name in ablation_summary:
                continue
            set_seed(args.seed)
            cfg = ForgeConfig(num_classes=n_cls, seq_len=8000, interval=20)
            model = FORGE(cfg, use_tde=u_tde, use_bss=u_bss, use_identifier=u_id)
            metrics = train_and_evaluate(
                model,
                loaders8k,
                epochs=args.epochs,
                lr=args.lr,
                k_list=(2,),
                select_metric="map@2",
                checkpoint_path=ckpt_dir / f"{case_name}.pt",
                results_path=results_dir / f"rq6_{case_name}.json",
                device=device,
                tag=f"RQ6-{case_name}",
            )
            ablation_summary[case_name] = metrics
            save_json(results_dir / "rq6_ablation.json", ablation_summary)
    else:
        n_cls = 100

    if args.skip_sensitivity:
        return

    # 2. Parameter Sensitivity Analysis (Figure 10)
    sens_summary = load_json_if_exists(results_dir / "rq6_sensitivity.json") or {
        "tau": {},
        "seq_len": {},
    }

    # Sweep tau in [10, 20, 30, 40, 50] with l=8000
    taus = [10, 20, 30, 40, 50]
    if not (args.resume and all(str(t) in sens_summary["tau"] for t in taus)):
        X8k, y8k, n_cls = load_npz_data(
            cw2_path, seq_len=8000, max_samples=args.max_samples, seed=args.seed
        )
        loaders8k = get_dataloaders(X8k, y8k, batch_size=args.batch_size, seed=args.seed)
        for tau in taus:
            if args.resume and str(tau) in sens_summary["tau"]:
                continue
            if tau == 20 and "Case4_Full_FORGE" in ablation_summary:
                sens_summary["tau"][str(tau)] = ablation_summary["Case4_Full_FORGE"]
                save_json(results_dir / "rq6_sensitivity.json", sens_summary)
                continue
            set_seed(args.seed)
            cfg = ForgeConfig(num_classes=n_cls, seq_len=8000, interval=tau)
            model = FORGE(cfg)
            metrics = train_and_evaluate(
                model,
                loaders8k,
                epochs=args.epochs,
                lr=args.lr,
                k_list=(2,),
                select_metric="map@2",
                checkpoint_path=ckpt_dir / f"sens_tau_{tau}.pt",
                device=device,
                tag=f"RQ6-tau={tau}",
            )
            sens_summary["tau"][str(tau)] = metrics
            save_json(results_dir / "rq6_sensitivity.json", sens_summary)
        del X8k, y8k, loaders8k

    # Sweep seq_len l in [2000, 4000, 6000, 8000, 10000] with tau=20
    lengths = [2000, 4000, 6000, 8000, 10000]
    for sl in lengths:
        if args.resume and str(sl) in sens_summary["seq_len"]:
            continue
        if sl == 8000 and "Case4_Full_FORGE" in ablation_summary:
            sens_summary["seq_len"][str(sl)] = ablation_summary["Case4_Full_FORGE"]
            save_json(results_dir / "rq6_sensitivity.json", sens_summary)
            continue
        X, y, n_cls = load_npz_data(
            cw2_path, seq_len=sl, max_samples=args.max_samples, seed=args.seed
        )
        loaders = get_dataloaders(X, y, batch_size=args.batch_size, seed=args.seed)
        set_seed(args.seed)
        cfg = ForgeConfig(num_classes=n_cls, seq_len=sl, interval=20)
        model = FORGE(cfg)
        metrics = train_and_evaluate(
            model,
            loaders,
            epochs=args.epochs,
            lr=args.lr,
            k_list=(2,),
            select_metric="map@2",
            checkpoint_path=ckpt_dir / f"sens_len_{sl}.pt",
            device=device,
            tag=f"RQ6-len={sl}",
        )
        sens_summary["seq_len"][str(sl)] = metrics
        save_json(results_dir / "rq6_sensitivity.json", sens_summary)
        del X, y, loaders


if __name__ == "__main__":
    main()

