"""
Dataset utilities.

No real inspection images are used anywhere in this repository. Two sources:

    1. `SyntheticChipDataset` -- procedurally generates grayscale images with
       periodic grids / stripes / diagonal patterns and a few blob "defects".
       This is the only data the shipped checkpoint was trained and evaluated on.
    2. `FolderDataset` -- loads grayscale images from a directory (untested on
       real inspection data; included so the pipeline can be pointed at it).

Both return (degraded, ground_truth); degradation (degradation.py) is applied
on the fly. `SyntheticChipDataset` is *stateful*: with a fixed seed the
sequence of items is reproducible, but ds[i] depends on call order.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .degradation import degrade

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_grayscale(path: str) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read image: {path}")
        return img.astype(np.float32) / 255.0
    from PIL import Image  # fallback

    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


# Backwards-compatible alias
_load_grayscale = load_grayscale


def list_images(folder: str) -> list[str]:
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


class FolderDataset(Dataset):
    """Loads ground-truth crops from a folder of images."""

    def __init__(
        self,
        folder: str,
        patch_size: int = 128,
        scale: int = 2,
        speckle_sigma: float = 0.15,
        gaussian_sigma: float = 0.02,
        train: bool = True,
    ):
        self.paths = list_images(folder)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {folder}")
        self.patch_size = patch_size
        self.scale = scale
        self.speckle_sigma = speckle_sigma
        self.gaussian_sigma = gaussian_sigma
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        ps = self.patch_size
        if h < ps or w < ps:
            pad_h, pad_w = max(0, ps - h), max(0, ps - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = img.shape
        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        return img[top : top + ps, left : left + ps]

    def __getitem__(self, idx: int):
        img = load_grayscale(self.paths[idx])
        crop = self._random_crop(img) if self.train else img
        gt = torch.from_numpy(crop).unsqueeze(0).float()
        lr = degrade(
            gt,
            scale=self.scale,
            speckle_sigma=self.speckle_sigma,
            gaussian_sigma=self.gaussian_sigma,
        )
        return lr, gt


class SyntheticChipDataset(Dataset):
    """
    Procedurally generates grayscale patterns that mimic the *structural*
    properties relevant to this task -- repeating periodic columns/rows
    (DRAM-like arrays), dense parallel fins (FinFET-like), and isolated
    defect blobs -- no real inspection data is used.

    This exists purely so the training/inference/evaluation pipeline is
    demonstrably runnable end-to-end. Swap in `FolderDataset` pointed at
    a real dataset (not validated here).
    """

    def __init__(
        self,
        length: int = 512,
        patch_size: int = 128,
        scale: int = 2,
        speckle_sigma: float = 0.15,
        gaussian_sigma: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.length = length
        self.patch_size = patch_size
        self.scale = scale
        self.speckle_sigma = speckle_sigma
        self.gaussian_sigma = gaussian_sigma
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def _make_pattern(self) -> np.ndarray:
        ps = self.patch_size
        kind = self.rng.integers(0, 3)
        yy, xx = np.meshgrid(np.arange(ps), np.arange(ps), indexing="ij")

        if kind == 0:
            # DRAM-like periodic grid of cells
            period = self.rng.integers(6, 16)
            grid = ((xx % period < period // 2).astype(np.float32) * 0.5) + (
                (yy % period < period // 2).astype(np.float32) * 0.5
            )
            img = 0.25 + 0.5 * grid
        elif kind == 1:
            # FinFET-like dense parallel fins (vertical stripes)
            period = self.rng.integers(4, 10)
            img = 0.3 + 0.5 * (xx % period < period // 2).astype(np.float32)
        else:
            # Diagonal / mixed periodic structure
            period = self.rng.integers(6, 14)
            img = 0.3 + 0.5 * (((xx + yy) % period) < period // 2).astype(np.float32)

        # Random defect blobs (small dark/bright spots) -- the "thing we
        # must not lose" during restoration.
        n_defects = self.rng.integers(0, 4)
        for _ in range(n_defects):
            cy, cx = self.rng.integers(0, ps, size=2)
            r = self.rng.integers(2, 5)
            val = float(self.rng.choice([0.05, 0.95]))
            yg, xg = np.ogrid[:ps, :ps]
            mask = (yg - cy) ** 2 + (xg - cx) ** 2 <= r * r
            img[mask] = val

        img = img.astype(np.float32)
        img += self.rng.normal(0, 0.02, size=img.shape).astype(np.float32)
        return np.clip(img, 0.0, 1.0)

    def __getitem__(self, idx: int):
        pattern = self._make_pattern()
        gt = torch.from_numpy(pattern).unsqueeze(0).float()
        lr = degrade(
            gt,
            scale=self.scale,
            speckle_sigma=self.speckle_sigma,
            gaussian_sigma=self.gaussian_sigma,
            rng=self.rng,
        )
        return lr, gt


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker (and each epoch) its own RNG.

    Without this, every worker gets a copy of the dataset's RNG in the same state,
    so workers emit duplicate samples and every epoch repeats the same samples.
    """
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if hasattr(ds, "rng"):
        ds.rng = np.random.default_rng(torch.initial_seed() % 2**32)
