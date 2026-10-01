#!/usr/bin/env python3
"""
Train RestorNet-S.

Reproduces the training run used to produce the checkpoint shipped in
weights/. Works out of the box on the built-in synthetic dataset (no real
data required); point --data-dir at a folder of real images (untested)
(and pass --dataset folder) for real images (untested).

Examples
--------
# Quick run on synthetic data (the shipped checkpoint came from an earlier version
# of this script, before the DataLoader-worker seeding fix -- see README)
python scripts/train.py --dataset synthetic --epochs 30 --steps-per-epoch 40 \
    --batch-size 8 --patch-size 96 --scale 2 --out weights/restornet_s_final.pth

# Real data
python scripts/train.py --dataset folder --data-dir /path/to/train_images \
    --val-dir /path/to/val_images --epochs 100 --batch-size 16 \
    --scale 2 --out weights/restornet_s_final.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restornet.model import build_model
from restornet.losses import HybridLoss
from restornet.dataset import FolderDataset, SyntheticChipDataset, seed_worker


def parse_args():
    p = argparse.ArgumentParser(description="Train RestorNet-S")
    p.add_argument("--dataset", choices=["synthetic", "folder"], default="synthetic")
    p.add_argument("--data-dir", type=str, default=None, help="Training images folder (dataset=folder)")
    p.add_argument("--val-dir", type=str, default=None, help="Optional validation images folder")
    p.add_argument("--scale", type=int, default=2, choices=[2, 4])
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=40, help="Only used for synthetic dataset")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--speckle-sigma", type=float, default=0.15)
    p.add_argument("--gaussian-sigma", type=float, default=0.02)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--n-rdb", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="weights/restornet_s_final.pth")
    p.add_argument("--log", type=str, default="weights/training_log.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint to resume model weights from")
    return p.parse_args()


def build_dataset(args, train: bool):
    if args.dataset == "synthetic":
        length = args.steps_per_epoch * args.batch_size
        return SyntheticChipDataset(
            length=length,
            patch_size=args.patch_size,
            scale=args.scale,
            speckle_sigma=args.speckle_sigma,
            gaussian_sigma=args.gaussian_sigma,
            seed=args.seed if train else args.seed + 1,
        )
    folder = args.data_dir if train else (args.val_dir or args.data_dir)
    return FolderDataset(
        folder,
        patch_size=args.patch_size,
        scale=args.scale,
        speckle_sigma=args.speckle_sigma,
        gaussian_sigma=args.gaussian_sigma,
        train=train,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train_ds = build_dataset(args, train=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        worker_init_fn=seed_worker,  # otherwise every worker/epoch repeats the same samples
    )

    val_loader = None
    if args.dataset == "folder" and args.val_dir:
        val_ds = build_dataset(args, train=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    elif args.dataset == "synthetic":
        # Fixed synthetic validation set (different seed from training), built once.
        val_ds = build_dataset(args, train=False)
        val_ds.length = 32
        val_loader = DataLoader([val_ds[i] for i in range(32)], batch_size=8, shuffle=False)

    model = build_model(
        scale=args.scale, base_channels=args.base_channels, n_rdb=args.n_rdb
    ).to(device)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed weights from {args.resume} (epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', float('nan')):.4f})")
    print(f"RestorNet-S params: {model.count_parameters():,}  device={device}")

    criterion = HybridLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_loss = float("inf")
    start_epoch = 0
    if args.resume and os.path.exists(args.log):
        try:
            with open(args.log) as f:
                history = json.load(f)
            if history:
                start_epoch = history[-1]["epoch"]
                best_loss = min(h["total"] for h in history)
        except (json.JSONDecodeError, KeyError):
            history = []
    t0 = time.time()

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        running = {"total": 0.0, "l1": 0.0, "ms_ssim": 0.0, "edge": 0.0}
        n_batches = 0

        for lr_img, gt_img in train_loader:
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)
            optimizer.zero_grad()
            pred = model(lr_img)
            loss, parts = criterion(pred, gt_img)
            loss.backward()
            optimizer.step()

            running["total"] += loss.item()
            for k, v in parts.items():
                running[k] += v
            n_batches += 1

        scheduler.step()
        for k in running:
            running[k] /= max(n_batches, 1)

        msg = (
            f"epoch {epoch:3d}/{start_epoch + args.epochs} | "
            f"loss {running['total']:.4f} "
            f"(l1 {running['l1']:.4f} ssim {running['ms_ssim']:.4f} edge {running['edge']:.4f}) | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"elapsed {time.time() - t0:.1f}s"
        )

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for lr_img, gt_img in val_loader:
                    lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                    pred = model(lr_img)
                    l, _ = criterion(pred, gt_img)
                    val_loss += l.item()
            val_loss /= max(len(val_loader), 1)
            msg += f" | val_loss {val_loss:.4f}"
            running["val_loss"] = val_loss

        print(msg)
        history.append({"epoch": epoch, **running})

        # Select the checkpoint on validation loss when a validation set exists.
        score = running.get("val_loss", running["total"])
        if score < best_loss:
            best_loss = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scale": args.scale,
                    "base_channels": args.base_channels,
                    "n_rdb": args.n_rdb,
                    "epoch": epoch,
                    "loss": best_loss,
                },
                args.out,
            )

    with open(args.log, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best checkpoint saved to {args.out} (loss={best_loss:.4f})")
    print(f"Training curve saved to {args.log}")


if __name__ == "__main__":
    main()
