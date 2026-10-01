"""Hybrid loss: L1 + multi-scale SSIM (mean of 1-SSIM over a 3-level bilinear
pyramid; a simplification of MS-SSIM, not the Wang et al. product form) +
Sobel-gradient L1 edge loss. Defaults: weights 1.0 / 1.0 / 0.5.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g


def _ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    g1d = _gaussian_window(window_size, 1.5).to(img1.device)
    window = (g1d[:, None] @ g1d[None, :]).unsqueeze(0).unsqueeze(0)
    channels = img1.shape[1]
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim


class MSSSIMLoss(nn.Module):
    """Multi-scale SSIM loss (1 - MS-SSIM), averaged over a small pyramid."""

    def __init__(self, scales: tuple[float, ...] = (1.0, 0.5, 0.25)):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for s in self.scales:
            if s != 1.0:
                size = (int(pred.shape[-2] * s), int(pred.shape[-1] * s))
                if min(size) < 11:
                    continue
                p = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
                t = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
            else:
                p, t = pred, target
            total = total + (1 - _ssim_map(p, t).mean())
        return total / len(self.scales)


class EdgeLoss(nn.Module):
    """Sobel-gradient L1 loss to keep edges (defect boundaries) sharp."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._grad(pred), self._grad(target))


class HybridLoss(nn.Module):
    """
    Weighted sum: L1 + MS-SSIM + edge-aware.
    The edge term penalizes gradient (high-frequency) mismatch.
    """

    def __init__(self, w_l1: float = 1.0, w_ssim: float = 1.0, w_edge: float = 0.5):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_edge = w_l1, w_ssim, w_edge
        self.ssim = MSSSIMLoss()
        self.edge = EdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim(pred, target)
        edge = self.edge(pred, target)
        total = self.w_l1 * l1 + self.w_ssim * ssim + self.w_edge * edge
        return total, {"l1": l1.item(), "ms_ssim": ssim.item(), "edge": edge.item()}
