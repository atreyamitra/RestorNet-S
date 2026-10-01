import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from restornet import build_model, degrade
from restornet.dataset import SyntheticChipDataset, seed_worker
from restornet.metrics import psnr_ssim
from restornet.model import load_checkpoint

WEIGHTS = os.path.join(ROOT, "weights", "restornet_s_final.pth")


def test_forward_shape_x2_cpu():
    m = build_model().eval()
    with torch.no_grad():
        y = m(torch.rand(2, 1, 40, 56))
    assert y.shape == (2, 1, 80, 112)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_forward_rejects_bad_input():
    m = build_model()
    with pytest.raises(ValueError):
        m(torch.rand(1, 3, 16, 16))
    with pytest.raises(ValueError):
        m(torch.rand(16, 16))


def test_degradation_is_seed_deterministic_and_downsamples():
    gt = torch.rand(1, 64, 64)
    a = degrade(gt, rng=np.random.default_rng(0))
    b = degrade(gt, rng=np.random.default_rng(0))
    c = degrade(gt, rng=np.random.default_rng(1))
    assert a.shape == (1, 32, 32)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert float(a.min()) >= 0 and float(a.max()) <= 1


def test_synthetic_dataset_reproducible_for_fixed_seed():
    d1 = SyntheticChipDataset(length=3, patch_size=64, seed=7)
    d2 = SyntheticChipDataset(length=3, patch_size=64, seed=7)
    for i in range(3):
        (l1, g1), (l2, g2) = d1[i], d2[i]
        assert torch.equal(l1, l2) and torch.equal(g1, g2)


def test_seed_worker_gives_distinct_samples_across_workers():
    from torch.utils.data import DataLoader
    ds = SyntheticChipDataset(length=8, patch_size=32, seed=1)
    dl = DataLoader(ds, batch_size=2, num_workers=2, worker_init_fn=seed_worker)
    sums = [round(float(g.sum()), 3) for _, gb in dl for g in gb]
    assert len(set(sums)) == len(sums)  # no duplicated samples


def test_metrics_sanity():
    x = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    p, s = psnr_ssim(x, x)
    assert np.isinf(p) and s == pytest.approx(1.0)
    noisy = np.clip(x + 0.1, 0, 1)
    assert psnr_ssim(noisy, x)[0] < 40
    with pytest.raises(ValueError):
        psnr_ssim(x, x[:16])


def test_checkpoint_loads_and_infers():
    model, scale = load_checkpoint(WEIGHTS, "cpu")
    assert scale == 2
    with torch.no_grad():
        assert model(torch.rand(1, 1, 24, 24)).shape == (1, 1, 48, 48)


def test_benchmark_smoke_tiny():
    from benchmark import run_benchmark
    s, panels = run_benchmark(WEIGHTS, n=3, seed=999, patch_size=64, grid=2)
    assert s["n_images"] == 3 and len(panels) == 2
    # deterministic given seed
    s2, _ = run_benchmark(WEIGHTS, n=3, seed=999, patch_size=64, grid=2)
    assert s["model"] == s2["model"] and s["bicubic"] == s2["bicubic"]


def test_pinned_benchmark_reproduces_committed_numbers():
    """Full 50-image CPU benchmark (~5 s); guards the README table."""
    import json
    from benchmark import run_benchmark
    with open(os.path.join(ROOT, "eval_results", "benchmark_metrics.json")) as f:
        ref = json.load(f)
    s, _ = run_benchmark(WEIGHTS, n=50, seed=999)
    assert s["model"]["psnr_mean"] == pytest.approx(ref["model"]["psnr_mean"], abs=0.05)
    assert s["bicubic"]["psnr_mean"] == pytest.approx(ref["bicubic"]["psnr_mean"], abs=0.05)
