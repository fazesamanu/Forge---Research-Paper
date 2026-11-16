# FORGE: A Robust Multi-tab Website Fingerprinting Attack Via Blind Source Separation (WWW '26)

Complete PyTorch implementation and reproduction suite for **FORGE** (Yitan Huang et al., WWW '26), including all 7 adapted baselines, defense simulators, network noise injection pipeline, and evaluation scripts for RQ1–RQ6.

## Architecture

1. **Module 1 (`forge/channel_construction.py`)**: Time-Delay Embedding ($\Phi: \mathcal{X} \to \mathbb{R}^{8 \times l}$) constructing 8 virtual observation channels (`v0..v7`: volumetric dynamics, temporal structure, and burst interaction patterns) vectorized on GPU.
2. **Module 2 (`forge/bss_solver.py`)**: Blind Source Separation (BSS) Solver using $L=4$ cascaded Iterative Correction Modules (ICMs) with residual analysis/synthesis Conv1d+BatchNorm1d filters (`[32, 64, 128, 256]`, `k=7`, `MaxPool1d(8, 4)`).
3. **Module 3 (`forge/identifier.py`)**: Transformer-based Website Identifier with Dual-Domain Consistency Attention (DDCA), combining time-domain self-attention and frequency-domain FFT magnitude spectrum self-attention fused via learnable gate $\alpha=0.1$.

## Project Structure

- `forge/`: Core model (`model.py`, `config.py`, `channel_construction.py`, `bss_solver.py`, `identifier.py`), metrics (`metrics.py`), training loop (`trainer.py`).
- `forge/baselines/`: Multi-label BCE adaptations of `AWF`, `DF`, `Tik-Tok`, `RF`, `BAPM`, `TMWF`, `ARES`.
- `forge/simulators/`: `Random` & `RegulaTor` (`Light`/`Heavy`) defense simulators (`defenses.py`), and Gilbert-Elliott + AR(1) jitter + local reordering network noise pipeline (`noise.py`).
- `forge/data/`: Zenodo downloader (`download_datasets.py`) and dataset loaders/mixers (`dataset.py`).
- `scripts/`: RQ1–RQ6 reproduction scripts (`run_rq1_closed_world.py` .. `run_rq6_ablation_sensitivity.py`), `plot_results.py`, and `run_all.py`.
- `tests/`: Unit & integration tests (`test_forge_modules.py`).

## Quick Start

```bash
# 1. Run unit tests
python -m tests.test_forge_modules

# 2. Download official WFlib datasets from Zenodo (Record 13731720)
python -m forge.data.download_datasets --records 13731720 --table8_only

# 3. Run full reproduction suite (RQ1-RQ6) and generate tables/figures
python -m scripts.run_all --resume
```

