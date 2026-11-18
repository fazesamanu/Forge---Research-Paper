"""Generate Markdown summary tables (Tables 2-6, 9-12) and figures (Figures 4-10) from JSON results."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

# Avoid Windows miniforge OpenMP duplicate-runtime abort (libomp.dll vs libiomp5md.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.common import load_json_if_exists


def _fmt(val) -> str:
    if val is None:
        return "-"
    return f"{float(val):.4f}"


def generate_markdown_report(results_dir: Path) -> None:
    lines = ["# FORGE Reproduction Results Report\n"]

    # Table 2, 9, 10: RQ1 Closed World
    rq1 = load_json_if_exists(results_dir / "rq1_summary.json")
    if rq1:
        lines.append("## Table 2: Closed-world Multi-tab Performance (Pre / Recall / F1)\n")
        header = "| Attack | " + " | ".join(
            f"{t}-tab Pre | {t}-tab Rec | {t}-tab F1" for t in (2, 3, 4, 5)
        ) + " |"
        sep = "|---|" + "|".join(["---"] * 12) + "|"
        lines.extend([header, sep])
        models = ["AWF", "DF", "Tik-Tok", "RF", "BAPM", "TMWF", "ARES", "FORGE"]
        for m in models:
            row = [m]
            for t in (2, 3, 4, 5):
                d = rq1.get(f"D_CW-{t}tab", {}).get(m, {})
                row.extend([_fmt(d.get("precision")), _fmt(d.get("recall")), _fmt(d.get("f1"))])
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("## Closed-world AUC & MAP@k (Figures 4 & Tables 9-10)\n")
        header = "| Attack | " + " | ".join(f"{t}-tab AUC | {t}-tab MAP@{t}" for t in (2, 3, 4, 5)) + " |"
        sep = "|---|" + "|".join(["---"] * 8) + "|"
        lines.extend([header, sep])
        for m in models:
            row = [m]
            for t in (2, 3, 4, 5):
                d = rq1.get(f"D_CW-{t}tab", {}).get(m, {})
                row.extend([_fmt(d.get("auc")), _fmt(d.get(f"map@{t}"))])
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Table 3: RQ2 Open World
    rq2 = load_json_if_exists(results_dir / "rq2_summary.json")
    if rq2:
        lines.append("## Table 3: Open-world Multi-tab Performance\n")
        lines.append("| Dataset | Attack | Pre | AP@k (IR) | P@k | MAP@k |")
        lines.append("|---|---|---|---|---|---|")
        for t in (2, 3, 4, 5):
            ds_key = f"D_OW-{t}tab"
            for m in ["AWF", "DF", "Tik-Tok", "RF", "BAPM", "TMWF", "ARES", "FORGE"]:
                d = rq2.get(ds_key, {}).get(m, {})
                if not d:
                    continue
                lines.append(
                    f"| {ds_key} | {m} | {_fmt(d.get('precision'))} | "
                    f"{_fmt(d.get(f'ap@{t}'))} | {_fmt(d.get(f'p@{t}'))} | {_fmt(d.get(f'map@{t}'))} |"
                )
        lines.append("")

    # Table 4 & 11: RQ3 Defenses
    rq3 = load_json_if_exists(results_dir / "rq3_summary.json")
    if rq3:
        lines.append("## Table 4: Multi-tab Defense Performance (2-tab)\n")
        defs = ["Random", "Front", "WTF-PAD", "RegularTor-Light", "RegularTor-Heavy"]
        lines.append("| Attack | " + " | ".join(f"{df} Pre | {df} AUC | {df} MAP@2" for df in defs) + " |")
        lines.append("|---|" + "|".join(["---"] * (len(defs) * 3)) + "|")
        for m in ["AWF", "DF", "Tik-Tok", "RF", "BAPM", "TMWF", "ARES", "FORGE"]:
            row = [m]
            for df in defs:
                d = rq3.get(df, {}).get(m, {})
                row.extend([_fmt(d.get("precision")), _fmt(d.get("auc")), _fmt(d.get("map@2"))])
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("## Figure 7 & Table 11: Single-tab Defenses (Walkie-Talkie & TrafficSliver)\n")
        lines.append("| Attack | Walkie-Talkie AUC | Walkie-Talkie Pre | TrafficSliver AUC | TrafficSliver Pre |")
        lines.append("|---|---|---|---|---|")
        for m in ["AWF", "DF", "Tik-Tok", "RF", "BAPM", "TMWF", "ARES", "FORGE"]:
            wt = rq3.get("Walkie-Talkie", {}).get(m, {})
            ts = rq3.get("TrafficSliver", {}).get(m, {})
            lines.append(
                f"| {m} | {_fmt(wt.get('auc'))} | {_fmt(wt.get('precision'))} | "
                f"{_fmt(ts.get('auc'))} | {_fmt(ts.get('precision'))} |"
            )
        lines.append("")

    # Table 5: RQ4 Dynamic
    rq4 = load_json_if_exists(results_dir / "rq4_summary.json")
    if rq4:
        lines.append("## Table 5: Dynamic Mix Settings\n")
        lines.append("| Attack | Mix-Tabs Pre | Mix-Tabs AUC | Mix-Tabs AP@5 | Mix-Tabs MAP@5 | Mix-Def Pre | Mix-Def AUC | Mix-Def AP@2 | Mix-Def MAP@2 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for m in ["AWF", "DF", "Tik-Tok", "RF", "BAPM", "TMWF", "ARES", "FORGE"]:
            mt = rq4.get("D_mix-num-tabs", {}).get(m, {})
            md = rq4.get("D_mix-defenses", {}).get(m, {})
            lines.append(
                f"| {m} | {_fmt(mt.get('precision'))} | {_fmt(mt.get('auc'))} | {_fmt(mt.get('ap@5'))} | {_fmt(mt.get('map@5'))} | "
                f"{_fmt(md.get('precision'))} | {_fmt(md.get('auc'))} | {_fmt(md.get('ap@2'))} | {_fmt(md.get('map@2'))} |"
            )
        lines.append("")

    # Table 12: RQ5 Noise
    rq5 = load_json_if_exists(results_dir / "rq5_summary.json")
    if rq5:
        lines.append("## RQ5: AUC Across Network Noise SNR Levels\n")
        snrs = ["0", "0.1", "0.5", "1", "2", "5", "10"]
        lines.append("| Attack | " + " | ".join(f"SNR={s}" for s in snrs) + " |")
        lines.append("|---|" + "|".join(["---"] * len(snrs)) + "|")
        for m in ["AWF", "DF", "Tik-Tok", "RF", "BAPM", "TMWF", "ARES", "FORGE"]:
            row = [m] + [_fmt(rq5.get(s, {}).get(m, {}).get("auc")) for s in snrs]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Table 6: RQ6 Ablation
    rq6_ab = load_json_if_exists(results_dir / "rq6_ablation.json")
    if rq6_ab:
        lines.append("## Table 6: Ablation Analysis of FORGE (D_CW-2tab)\n")
        lines.append("| Test Case | P@2 | MAP@2 |")
        lines.append("|---|---|---|")
        for k in ["Case1_no_TDE", "Case2_no_BSS", "Case3_no_Identifier", "Case4_Full_FORGE"]:
            d = rq6_ab.get(k, {})
            lines.append(f"| {k} | {_fmt(d.get('p@2'))} | {_fmt(d.get('map@2'))} |")
        lines.append("")

    out_path = results_dir / "tables_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report -> {out_path}")


def generate_figures(results_dir: Path) -> None:
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_models = ["AWF", "Tik-Tok", "RF", "ARES", "TMWF", "FORGE"]

    # Fig 4 & Fig 5: RQ1
    rq1 = load_json_if_exists(results_dir / "rq1_summary.json")
    if rq1:
        tabs = [2, 3, 4, 5]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for m in plot_models:
            aucs = [rq1.get(f"D_CW-{t}tab", {}).get(m, {}).get("auc", np.nan) for t in tabs]
            maps = [rq1.get(f"D_CW-{t}tab", {}).get(m, {}).get(f"map@{t}", np.nan) for t in tabs]
            axes[0].plot(tabs, aucs, marker="o", label=m)
            axes[1].plot(tabs, maps, marker="o", label=m)
        axes[0].set_title("Closed-world AUC vs Tab Count")
        axes[0].set_xlabel("Number of tabs")
        axes[0].set_ylabel("AUC")
        axes[1].set_title("Closed-world MAP@k vs Tab Count")
        axes[1].set_xlabel("Number of tabs")
        axes[1].set_ylabel("MAP@k")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig4_closed_world_auc_map.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        ks = [2, 3, 4, 5]
        for idx, t in enumerate(tabs):
            for m in plot_models:
                vals = [rq1.get(f"D_CW-{t}tab", {}).get(m, {}).get(f"p@{k}", np.nan) for k in ks]
                axes[idx].plot(ks, vals, marker="o", label=m)
            axes[idx].set_title(f"Tab = {t}")
            axes[idx].set_xlabel("k")
            axes[idx].set_ylabel("P@k")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig5_closed_world_patk.png", dpi=200)
        plt.close(fig)

    # Fig 6: RQ2 Open-world AUC
    rq2 = load_json_if_exists(results_dir / "rq2_summary.json")
    if rq2:
        tabs = [2, 3, 4, 5]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for m in plot_models:
            aucs = [rq2.get(f"D_OW-{t}tab", {}).get(m, {}).get("auc", np.nan) for t in tabs]
            ax.plot(tabs, aucs, marker="o", label=m)
        ax.set_title("Open-world AUC vs Tab Count")
        ax.set_xlabel("Number of tabs")
        ax.set_ylabel("AUC")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig6_open_world_auc.png", dpi=200)
        plt.close(fig)

    # Fig 7: Single-tab defenses
    rq3 = load_json_if_exists(results_dir / "rq3_summary.json")
    if rq3 and ("Walkie-Talkie" in rq3 or "TrafficSliver" in rq3):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for idx, (df_name, metric) in enumerate([("Walkie-Talkie", "auc"), ("TrafficSliver", "precision")]):
            vals = [rq3.get(df_name, {}).get(m, {}).get(metric, 0.0) for m in plot_models]
            axes[idx].bar(plot_models, vals)
            axes[idx].set_title(f"Performance under {df_name}")
            axes[idx].set_ylabel(metric.upper())
            axes[idx].tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig7_single_tab_defenses.png", dpi=200)
        plt.close(fig)

    # Fig 8: Cross-tab ROC
    roc_data = load_json_if_exists(results_dir / "rq4_crosstab_roc.json")
    if roc_data:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        left_keys = ["train3test3", "train5test3", "train5test5", "train3test5"]
        right_keys = ["train3test3", "train4test3", "train4test4", "train3test4"]
        for ax, keys in zip(axes, [left_keys, right_keys]):
            for k in keys:
                if k in roc_data:
                    d = roc_data[k]
                    ax.semilogx(
                        np.clip(d["fpr"], 1e-4, 1.0),
                        d["tpr"],
                        label=f"{k} (AUC={d['auc']:.3f})",
                    )
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig8_crosstab_roc.png", dpi=200)
        plt.close(fig)

    # Fig 9: Noise AUC
    rq5 = load_json_if_exists(results_dir / "rq5_summary.json")
    if rq5:
        snrs = ["0", "0.1", "0.5", "1", "2", "5", "10"]
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in ["Tik-Tok", "RF", "ARES", "TMWF", "FORGE"]:
            vals = [rq5.get(s, {}).get(m, {}).get("auc", np.nan) for s in snrs]
            ax.plot(snrs, vals, marker="o", label=m)
        ax.set_xlabel("Signal-to-Noise Ratio (s)")
        ax.set_ylabel("AUC")
        ax.set_title("AUC Under Network Noise")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig9_noise_auc.png", dpi=200)
        plt.close(fig)

    # Fig 10: Sensitivity
    sens = load_json_if_exists(results_dir / "rq6_sensitivity.json")
    if sens:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if sens.get("seq_len"):
            ls = sorted(sens["seq_len"].keys(), key=int)
            maps = [sens["seq_len"][k].get("map@2", np.nan) for k in ls]
            axes[0].plot([int(k) for k in ls], maps, marker="o")
            axes[0].set_xlabel("Sequence length l")
            axes[0].set_ylabel("MAP@2")
            axes[0].set_ylim(0, 1.05)
        if sens.get("tau"):
            ts = sorted(sens["tau"].keys(), key=int)
            maps = [sens["tau"][k].get("map@2", np.nan) for k in ts]
            axes[1].plot([int(k) for k in ts], maps, marker="o")
            axes[1].set_xlabel("Interval tau")
            axes[1].set_ylabel("MAP@2")
            axes[1].set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig10_sensitivity.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    rdir = Path(args.results_dir)
    generate_markdown_report(rdir)
    generate_figures(rdir)


if __name__ == "__main__":
    main()

