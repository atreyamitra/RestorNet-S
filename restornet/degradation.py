"""
Synthetic degradation (a simple model, NOT calibrated against any real sensor):
    1. Multiplicative speckle noise: I + I*N(0, 0.15^2)
    2. Additive Gaussian noise: N(0, 0.02^2)
    3. Bicubic downsampling by `scale` (torch, align_corners=False)
    4. Clamp to [0, 1]

Used for on-the-fly training pairs and for building the evaluation set.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def add_speckle_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Multiplicative speckle noise: I' = I + I * N(0, sigma^2).

    Deliberately NOT clamped before the additive-noise stage, so pixels can
    be pushed outside [0, 1] -- exactly the "out-of-range pixel" behaviour
    called out in the problem statement. Clamping happens once, at the end
    of the full degradation pipeline.
    """
    noise = torch.randn_like(img) * sigma
    return img + img * noise


def add_gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    return img + torch.randn_like(img) * sigma


def bicubic_downsample(img: torch.Tensor, scale: int) -> torch.Tensor:
    """img: (C, H, W) or (N, C, H, W) tensor in [0, 1]."""
    squeeze = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    h, w = img.shape[-2:]
    out = F.interpolate(
        img, size=(h // scale, w // scale), mode="bicubic", align_corners=False
    )
    return out.squeeze(0) if squeeze else out


def degrade(
    img: torch.Tensor,
    scale: int = 2,
    speckle_sigma: float = 0.15,
    gaussian_sigma: float = 0.02,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Full degradation pipeline:
    speckle -> additive Gaussian -> bicubic downsample -> clamp.

    Args:
        img: ground-truth image tensor, shape (1, H, W) or (N, 1, H, W),
             values in [0, 1].
        scale: downsampling factor (spatial resolution reduction).
        speckle_sigma: std-dev of the multiplicative speckle term.
        gaussian_sigma: std-dev of the additive Gaussian term.

    Returns:
        Degraded, low-resolution tensor, same rank as input, values
        clamped to [0, 1].
    """
    if rng is not None:
        # NOTE: reseeds torch's *global* RNG from `rng` (deterministic, but global).
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    noisy = add_speckle_noise(img, speckle_sigma)
    noisy = add_gaussian_noise(noisy, gaussian_sigma)
    low_res = bicubic_downsample(noisy, scale)
    return torch.clamp(low_res, 0.0, 1.0)
