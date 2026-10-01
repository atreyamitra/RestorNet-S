#!/usr/bin/env python3
"""
Evaluate restored images against ground truth: prints SSIM / PSNR / LPIPS.

Usage
-----
python scripts/evaluate.py \
    --restored path/to/restored_images \
    --ground-truth path/to/ground_truth_images

Matches files by filename stem (ignoring the "_restored" suffix added by
inference.py), computes per-image and averaged SSIM, PSNR, and (if the
`lpips` package is installed) LPIPS, and prints a summary table.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restornet.dataset import list_images, load_grayscale
from restornet.metrics import psnr_ssim

try:
    import lpips
    import torch
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RestorNet-S outputs")
    p.add_argument("--restored", required=True, help="Folder of restored images")
    p.add_argument("--ground-truth", required=True, help="Folder of ground-truth images")
    p.add_argument("--no-lpips", action="store_true", help="Skip LPIPS even if installed")
    return p.parse_args()


def _stem(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("_restored", "").replace("_gt", "").replace("_ground_truth", "")


def match_pairs(restored_dir: str, gt_dir: str):
    restored = {_stem(p): p for p in list_images(restored_dir)}
    gt = {_stem(p): p for p in list_images(gt_dir)}
    common = sorted(set(restored) & set(gt))
    missing_gt = sorted(set(restored) - set(gt))
    missing_pred = sorted(set(gt) - set(restored))
    if missing_gt:
        print(f"[warn] {len(missing_gt)} restored image(s) have no ground-truth match: {missing_gt[:5]}...")
    if missing_pred:
        print(f"[warn] {len(missing_pred)} ground-truth image(s) have no restored match: {missing_pred[:5]}...")
    return [(restored[k], gt[k]) for k in common]


def main():
    args = parse_args()
    pairs = match_pairs(args.restored, args.ground_truth)
    if not pairs:
        print("No matching restored/ground-truth pairs found.")
        return

    use_lpips = _LPIPS_AVAILABLE and not args.no_lpips
    lp_model = lpips.LPIPS(net="alex") if use_lpips else None
    if _LPIPS_AVAILABLE and args.no_lpips:
        print("LPIPS available but skipped (--no-lpips)")
    elif not _LPIPS_AVAILABLE:
        print("[info] `lpips` package not installed -- skipping LPIPS (pip install lpips)")

    rows = []
    for restored_path, gt_path in pairs:
        pred = load_grayscale(restored_path)
        target = load_grayscale(gt_path)

        if pred.shape != target.shape:
            # Resize prediction to GT resolution if they mismatch (should
            # normally match exactly given a correct scale factor).
            import cv2
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)

        psnr, ssim = psnr_ssim(pred, target)

        lp = None
        if use_lpips:
            t1 = torch.from_numpy(pred).float()[None, None] * 2 - 1
            t2 = torch.from_numpy(target).float()[None, None] * 2 - 1
            t1, t2 = t1.repeat(1, 3, 1, 1), t2.repeat(1, 3, 1, 1)
            lp = lp_model(t1, t2).item()

        rows.append((_stem(restored_path), psnr, ssim, lp))

    name_w = max(len(r[0]) for r in rows) + 2
    header = f"{'image':<{name_w}}{'PSNR (dB)':>12}{'SSIM':>10}{'LPIPS':>10}"
    print("\n" + header)
    print("-" * len(header))
    for name, psnr, ssim, lp in rows:
        lp_str = f"{lp:.4f}" if lp is not None else "n/a"
        print(f"{name:<{name_w}}{psnr:>12.3f}{ssim:>10.4f}{lp_str:>10}")

    psnrs = [r[1] for r in rows]
    ssims = [r[2] for r in rows]
    lps = [r[3] for r in rows if r[3] is not None]

    print("-" * len(header))
    avg_lp_str = f"{np.mean(lps):.4f}" if lps else "n/a"
    print(f"{'AVERAGE':<{name_w}}{np.mean(psnrs):>12.3f}{np.mean(ssims):>10.4f}{avg_lp_str:>10}")
    print(f"\n{len(rows)} image pair(s) evaluated.")


if __name__ == "__main__":
    main()
