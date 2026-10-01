#!/usr/bin/env python3
"""
One-command, deterministic evaluation of RestorNet-S vs. a bicubic baseline on
SYNTHETICALLY degraded synthetic images.

    python scripts/benchmark.py --weights weights/restornet_s_final.pth \
        --n 50 --seed 999 --out-dir eval_results

Pipeline per image (identical for both methods):
    ground truth (synthetic, 160x160, 8-bit) -> degrade() -> 80x80 -> quantize to 8-bit
      baseline : cv2.resize(.., INTER_CUBIC) to 160x160       (no denoising)
      model    : RestorNet-S forward pass -> 160x160
    both outputs quantized to 8-bit, then PSNR/SSIM vs. the 8-bit ground truth
    (restornet.metrics.psnr_ssim, data_range=1.0), averaged per image.

Writes <out-dir>/benchmark_metrics.json, benchmark_table.md and
benchmark_grid.png (first --grid images by index: degraded | bicubic | model | GT;
deterministic, not cherry-picked).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restornet.dataset import SyntheticChipDataset
from restornet.metrics import psnr_ssim
from restornet.model import load_checkpoint


def to_u8(t: torch.Tensor) -> np.ndarray:
    return np.clip(t.squeeze().cpu().numpy() * 255, 0, 255).astype(np.uint8)


def run_benchmark(weights: str, n: int, seed: int, patch_size: int = 160,
                  device: str = "cpu", grid: int = 8):
    model, scale = load_checkpoint(weights, device)
    ds = SyntheticChipDataset(length=n, patch_size=patch_size, scale=scale, seed=seed)
    rows, panels = [], []
    with torch.no_grad():
        for i in range(n):
            lr, gt = ds[i]
            gt8, lr8 = to_u8(gt), to_u8(lr)
            bic8 = cv2.resize(lr8, (gt8.shape[1], gt8.shape[0]), interpolation=cv2.INTER_CUBIC)
            x = torch.from_numpy(lr8.astype(np.float32) / 255.0)[None, None].to(device)
            out8 = to_u8(model(x))
            f = lambda a: a.astype(np.float32) / 255.0
            bp, bs = psnr_ssim(f(bic8), f(gt8))
            mp, ms = psnr_ssim(f(out8), f(gt8))
            rows.append({"idx": i, "bicubic_psnr": bp, "bicubic_ssim": bs,
                         "model_psnr": mp, "model_ssim": ms})
            if i < grid:
                up = cv2.resize(lr8, (gt8.shape[1], gt8.shape[0]), interpolation=cv2.INTER_NEAREST)
                panels.append(np.hstack([up, bic8, out8, gt8]))
    m = lambda k: float(np.mean([r[k] for r in rows]))
    sd = lambda k: float(np.std([r[k] for r in rows]))
    wins = int(sum(r["model_psnr"] > r["bicubic_psnr"] for r in rows))
    with open(weights, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    summary = {
        "checkpoint": weights, "checkpoint_sha256": sha, "n_images": n, "seed": seed,
        "patch_size_gt": patch_size, "scale": scale, "data": "synthetic images, synthetic degradation",
        "bicubic": {"psnr_mean": m("bicubic_psnr"), "psnr_std": sd("bicubic_psnr"),
                    "ssim_mean": m("bicubic_ssim"), "ssim_std": sd("bicubic_ssim")},
        "model": {"psnr_mean": m("model_psnr"), "psnr_std": sd("model_psnr"),
                  "ssim_mean": m("model_ssim"), "ssim_std": sd("model_ssim")},
        "model_beats_bicubic_psnr_on": f"{wins}/{n}",
        "per_image": rows,
    }
    return summary, panels


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", default="weights/restornet_s_final.pth")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--patch-size", type=int, default=160)
    p.add_argument("--grid", type=int, default=8, help="images in the qualitative grid (first N)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="eval_results")
    a = p.parse_args()
    if not os.path.isfile(a.weights):
        sys.exit(f"error: checkpoint not found: {a.weights}")
    torch.manual_seed(a.seed)
    s, panels = run_benchmark(a.weights, a.n, a.seed, a.patch_size, a.device, a.grid)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(s, f, indent=2)
    b, m = s["bicubic"], s["model"]
    table = (
        f"Synthetic held-out set: {a.n} images, seed {a.seed}, {a.patch_size}px GT, x{s['scale']}\n\n"
        "| Method | Evaluation set | PSNR (dB) | SSIM |\n|---|---|---|---|\n"
        f"| Bicubic (no denoising) | {a.n} synthetic, seed {a.seed} | {b['psnr_mean']:.2f} ± {b['psnr_std']:.2f} | {b['ssim_mean']:.4f} ± {b['ssim_std']:.4f} |\n"
        f"| RestorNet-S | {a.n} synthetic, seed {a.seed} | {m['psnr_mean']:.2f} ± {m['psnr_std']:.2f} | {m['ssim_mean']:.4f} ± {m['ssim_std']:.4f} |\n\n"
        f"(mean ± std over images; model PSNR > bicubic PSNR on {s['model_beats_bicubic_psnr_on']} images)\n"
    )
    with open(os.path.join(a.out_dir, "benchmark_table.md"), "w") as f:
        f.write(table)
    if panels:
        g = np.vstack(panels)
        hdr = np.full((22, g.shape[1]), 255, np.uint8)
        w = g.shape[1] // 4
        for k, lab in enumerate(["Degraded (nearest-up)", "Bicubic", "RestorNet-S", "Ground truth"]):
            cv2.putText(hdr, lab, (k * w + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(a.out_dir, "benchmark_grid.png"), np.vstack([hdr, g]))
    print(table)


if __name__ == "__main__":
    main()
