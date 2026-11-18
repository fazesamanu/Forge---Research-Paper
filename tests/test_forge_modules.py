"""Unit and integration tests for FORGE modules, baselines, simulators, and metrics."""

import numpy as np
import torch

from forge.config import ForgeConfig
from forge.model import FORGE
from forge.baselines import build_baseline, BASELINE_REGISTRY
from forge.metrics import compute_all_metrics
from forge.simulators.defenses import simulate_random_defense, simulate_regulator
from forge.simulators.noise import inject_network_noise


def test_tde_exactness():
    from forge.channel_construction import TrafficChannelConstructor

    tde = TrafficChannelConstructor(interval=4, seq_len=8)
    # 1 sample, 8 packets with timestamps and directions
    x = torch.tensor([[0.1, -0.2, 0.4, 0.7, -0.9, -1.2, 1.5, 0.0]])
    v = tde(x)
    assert v.shape == (1, 8, 8)
    # At t=3, window is [0.1, -0.2, 0.4, 0.7]:
    # up packets: 0.1, 0.4, 0.7 -> v0=3, v2 = 0.7 - 0.1 = 0.6
    # down packets: -0.2 -> v1=1, v3 = 0.0
    # up bursts: [0.1], [0.4, 0.7] -> v4=2, v6 = 3/2 = 1.5
    # down bursts: [-0.2] -> v5=1, v7 = 1/1 = 1.0
    assert torch.isclose(v[0, 0, 3], torch.tensor(3.0))
    assert torch.isclose(v[0, 1, 3], torch.tensor(1.0))
    assert torch.isclose(v[0, 2, 3], torch.tensor(0.6), atol=1e-5)
    assert torch.isclose(v[0, 3, 3], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(v[0, 4, 3], torch.tensor(2.0))
    assert torch.isclose(v[0, 5, 3], torch.tensor(1.0))
    assert torch.isclose(v[0, 6, 3], torch.tensor(1.5))
    assert torch.isclose(v[0, 7, 3], torch.tensor(1.0))


def test_forge_and_ablations_forward_backward():
    cfg = ForgeConfig(seq_len=512, num_classes=10)
    x = torch.randn(4, 512)
    y = torch.zeros(4, 10)
    y[:, :2] = 1.0

    for flags in [
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ]:
        model = FORGE(cfg, use_tde=flags[0], use_bss=flags[1], use_identifier=flags[2])
        logits = model(x)
        assert logits.shape == (4, 10)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()


def test_all_baselines_forward_backward():
    x = torch.randn(4, 1024)
    y = torch.zeros(4, 10)
    y[:, :2] = 1.0
    for name in BASELINE_REGISTRY:
        m = build_baseline(name, num_classes=10, seq_len=1024)
        out = m(x)
        assert out.shape == (4, 10), f"{name} returned {out.shape}"
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        loss.backward()


def test_simulators_and_metrics():
    rng = np.random.default_rng(42)
    ts = np.cumsum(rng.uniform(0.001, 0.05, size=(10, 200)), axis=1).astype(np.float32)
    dirs = rng.choice([-1.0, 1.0], size=(10, 200)).astype(np.float32)
    X = ts * dirs
    y = np.zeros((10, 5), dtype=np.float32)
    y[:, 0] = 1.0
    y[:, 2] = 1.0

    X_rand, y_rand = simulate_random_defense(X, y, num_instances=8)
    assert X_rand.shape == (8, 200) and y_rand.shape == (8, 5)

    X_reg, y_reg = simulate_regulator(X, y, mode="light")
    assert X_reg.shape == (10, 200) and y_reg.shape == (10, 5)

    X_noisy, mask = inject_network_noise(X, snr_level=2.0, return_mask=True)
    assert X_noisy.shape == (10, 200) and mask.shape == (10, 200)

    # Analytical metrics test
    y_true = np.array([[1, 1, 0, 0], [0, 1, 0, 1]], dtype=np.float32)
    y_prob = np.array([[0.9, 0.8, 0.2, 0.1], [0.1, 0.7, 0.6, 0.3]], dtype=np.float32)
    m = compute_all_metrics(y_true, y_prob, k_list=(2,))
    # Sample 0: top-2 are {0, 1} -> P@1=1, P@2=1, MAP@2=1
    # Sample 1: top-2 are {1, 2} -> P@1=1, P@2=0.5, MAP@2=0.75
    assert abs(m["p@2"] - 0.75) < 1e-6
    assert abs(m["map@2"] - 0.875) < 1e-6


if __name__ == "__main__":
    test_tde_exactness()
    test_forge_and_ablations_forward_backward()
    test_all_baselines_forward_backward()
    test_simulators_and_metrics()
    print("ALL UNIT TESTS PASSED!")

