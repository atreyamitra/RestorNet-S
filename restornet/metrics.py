"""PSNR / SSIM used by every evaluation path in this repo.

Images are single-channel float arrays in [0, 1] (data_range=1.0), computed
per image with scikit-image defaults (SSIM: 7x7 uniform window, no padding
-> border pixels are cropped by the window) and then averaged over images.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr_ssim(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    return (
        float(peak_signal_noise_ratio(target, pred, data_range=1.0)),
        float(structural_similarity(target, pred, data_range=1.0)),
    )
