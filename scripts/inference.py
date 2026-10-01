#!/usr/bin/env python3
"""
Run RestorNet-S inference on a folder of degraded grayscale images.

Usage
-----
python scripts/inference.py \
    --input path/to/degraded_images \
    --output path/to/restored_images \
    --weights weights/restornet_s_final.pth

Input : a folder containing degraded grayscale images (.png/.jpg/.tif/...).
Output: the same folder structure, each image restored (denoised +
        super-resolved by the checkpoint's --scale factor) and written as
        PNG to --output, using the same base filename.

No manual edits are required -- all paths are passed via CLI flags, and
the script creates --output if it doesn't already exist.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restornet.model import load_checkpoint
from restornet.dataset import list_images, load_grayscale

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def parse_args():
    p = argparse.ArgumentParser(description="RestorNet-S inference")
    p.add_argument("--input", required=True, help="Folder of degraded images")
    p.add_argument("--output", required=True, help="Folder to write restored images to")
    p.add_argument("--weights", default="weights/restornet_s_final.pth")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tile", type=int, default=0,
                    help="If >0, run inference in tiles of this size (for large images / low memory)")
    p.add_argument("--tile-overlap", type=int, default=16)
    return p.parse_args()


def save_image(path: str, arr: np.ndarray):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if cv2 is not None:
        cv2.imwrite(path, arr)
    else:
        from PIL import Image
        Image.fromarray(arr).save(path)


@torch.no_grad()
def run_tiled(model, img_t: torch.Tensor, scale: int, tile: int, overlap: int) -> torch.Tensor:
    _, _, h, w = img_t.shape
    out = torch.zeros(1, 1, h * scale, w * scale, device=img_t.device)
    weight = torch.zeros_like(out)

    stride = tile - overlap
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom = min(top + tile, h)
            right = min(left + tile, w)
            top0, left0 = max(0, bottom - tile), max(0, right - tile)

            patch = img_t[:, :, top0:bottom, left0:right]
            pred = model(patch)

            ot, ol = top0 * scale, left0 * scale
            ob, orr = bottom * scale, right * scale
            out[:, :, ot:ob, ol:orr] += pred
            weight[:, :, ot:ob, ol:orr] += 1.0

            if right >= w:
                break
        if bottom >= h:
            break

    return out / weight.clamp(min=1.0)


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if not os.path.isfile(args.weights):
        sys.exit(f"error: checkpoint not found: {args.weights}")
    if not os.path.isdir(args.input):
        sys.exit(f"error: input folder not found: {args.input}")
    model, scale = load_checkpoint(args.weights, args.device)
    paths = list_images(args.input)
    if not paths:
        print(f"No images found in {args.input}")
        return

    print(f"Loaded RestorNet-S (scale=x{scale}) on {args.device}")
    print(f"Found {len(paths)} image(s) in {args.input}")

    t0 = time.time()
    for path in paths:
        img = load_grayscale(path)
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(args.device)

        if args.tile > 0 and (img.shape[0] > args.tile or img.shape[1] > args.tile):
            pred = run_tiled(model, img_t, scale, args.tile, args.tile_overlap)
        else:
            pred = model(img_t)

        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
        out_name = os.path.splitext(os.path.basename(path))[0] + "_restored.png"
        out_path = os.path.join(args.output, out_name)
        save_image(out_path, pred_np)
        print(f"  {os.path.basename(path)} -> {out_name}  ({img.shape} -> {pred_np.shape})")

    dt = time.time() - t0
    print(f"\nDone: {len(paths)} image(s) in {dt:.2f}s ({dt / len(paths) * 1000:.1f} ms/image)")


if __name__ == "__main__":
    main()
