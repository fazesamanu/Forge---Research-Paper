"""Master orchestrator to run all RQ1-RQ6 experiments and generate tables/figures."""

import argparse
import subprocess
import sys


def run_step(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run full FORGE reproduction suite (RQ1-RQ6).")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    py = sys.executable
    common = ["--data_dir", args.data_dir, "--results_dir", args.results_dir]
    if args.resume:
        common.append("--resume")

    train_args = common + ["--epochs", str(args.epochs), "--batch_size", str(args.batch_size)]

    run_step([py, "-u", "-m", "scripts.run_rq1_closed_world"] + train_args)
    run_step([py, "-u", "-m", "scripts.run_rq6_ablation_sensitivity"] + train_args)
    run_step([py, "-u", "-m", "scripts.run_rq2_open_world"] + train_args)
    run_step([py, "-u", "-m", "scripts.run_rq3_defenses"] + train_args)
    run_step([py, "-u", "-m", "scripts.run_rq4_dynamic"] + train_args)
    run_step([py, "-u", "-m", "scripts.run_rq5_noise"] + train_args)
    run_step([py, "-u", "-m", "scripts.plot_results", "--results_dir", args.results_dir])


if __name__ == "__main__":
    main()

