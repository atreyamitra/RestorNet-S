#!/usr/bin/env python3
"""
Regenerate sample_outputs/: a handful of degraded/restored/ground-truth
triplets plus a side-by-side comparison grid, using the synthetic dataset
(synthetic data only).

    python scripts/make_samples.py --weights weights/restornet_s_final.pth
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restornet.dataset import SyntheticChipDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/restornet_s_final.pth")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--patch-size", type=int, default=160)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", default="sample_outputs")
    return p.parse_args()


def main():
    args = parse_args()
    deg_dir = os.path.join(args.out_dir, "degraded")
    rest_dir = os.path.join(args.out_dir, "restored")
    gt_dir = os.path.join(args.out_dir, "ground_truth")
    for d in (deg_dir, rest_dir, gt_dir):
        os.makedirs(d, exist_ok=True)

    ds = SyntheticChipDataset(
        length=args.n, patch_size=args.patch_size, scale=args.scale, seed=args.seed
    )
    for i in range(args.n):
        lr, gt = ds[i]
        cv2.imwrite(os.path.join(gt_dir, f"sample_{i:02d}.png"),
                    np.clip(gt.squeeze(0).numpy() * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(deg_dir, f"sample_{i:02d}.png"),
                    np.clip(lr.squeeze(0).numpy() * 255, 0, 255).astype(np.uint8))

    subprocess.run(
        [sys.executable, os.path.join("scripts", "inference.py"),
         "--input", deg_dir, "--output", rest_dir, "--weights", args.weights],
        check=True,
    )

    # Build the comparison grid
    rows = []
    for i in range(args.n):
        gt = cv2.imread(os.path.join(gt_dir, f"sample_{i:02d}.png"), cv2.IMREAD_GRAYSCALE)
        deg = cv2.imread(os.path.join(deg_dir, f"sample_{i:02d}.png"), cv2.IMREAD_GRAYSCALE)
        rest = cv2.imread(os.path.join(rest_dir, f"sample_{i:02d}_restored.png"), cv2.IMREAD_GRAYSCALE)
        deg_up = cv2.resize(deg, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        row = np.hstack([deg_up, rest, gt])
        row = cv2.copyMakeBorder(row, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=80)
        rows.append(row)
    grid = np.vstack(rows)
    col_w = gt.shape[1] + 4

    header = np.full((26, grid.shape[1]), 255, dtype=np.uint8)
    for idx, label in enumerate(["Degraded", "Restored", "Ground Truth"]):
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x = idx * col_w + (col_w - tw) // 2
        cv2.putText(header, label, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 0, 1, cv2.LINE_AA)

    full = np.vstack([header, grid])
    grid_path = os.path.join(args.out_dir, "comparison_grid.png")
    cv2.imwrite(grid_path, full)
    print(f"Wrote {args.n} triplets to {args.out_dir}/ and comparison grid to {grid_path}")


if __name__ == "__main__":
    main()
