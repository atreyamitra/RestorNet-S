# RestorNet-S

A small PyTorch CNN (868K parameters) for **joint denoising + 2× super-resolution** of grayscale images, with a reproducible CPU benchmark against a bicubic baseline. **Everything here is trained and evaluated on synthetic images with synthetic degradation; no real inspection data was used.**

[![CI](https://github.com/atreyamitra/RestorNet-S/actions/workflows/ci.yml/badge.svg)](https://github.com/atreyamitra/RestorNet-S/actions)

## What it explores

Restoring an image that is both noisy and low-resolution in a single forward pass, instead of denoise-then-upscale. The repo is an engineering exercise: model, data generator, training loop, inference CLI, deterministic evaluation, tests and CI.

## Architecture (as implemented in `restornet/model.py`)

Input `(N,1,H,W)` in [0,1] → output `(N,1,2H,2W)`. It is **not** an encoder-decoder: there is no downsampling path; all feature processing happens at input resolution.

1. 3×3 conv head (1→48 channels)
2. 6 × Residual Dense Block (4 dense 3×3 conv layers, growth 32, 1×1 fuse, squeeze-and-excitation **channel attention**, local skip)
3. Dilated-conv residual branch (dilation 2 and 4)
4. Long skip around the trunk (residual-in-residual)
5. **PixelShuffle** upsampling head (3×3 conv → `PixelShuffle(2)`)
6. 3×3 conv tail, then add a **fixed bicubic upsample of the input** (global residual: the network learns a correction to bicubic) and clamp to [0,1]

Loss (`restornet/losses.py`): L1 + (1 − SSIM) averaged over a 3-level pyramid + Sobel-gradient L1 (weights 1 / 1 / 0.5). No ablations were run, so the contribution of individual components (attention, dilated branch, each loss term) is **untested**.

## Data and degradation (all synthetic)

- **Clean images**: `SyntheticChipDataset` generates procedural grayscale images: periodic grids, vertical stripes or diagonal stripes (period 4–16 px), 0–3 saturated circular blobs, plus σ=0.02 Gaussian texture noise. The family is small and simple.
- **Degradation** (`restornet/degradation.py`): speckle `I + I·N(0,0.15²)` → additive `N(0,0.02²)` → bicubic ↓2 → clamp [0,1]. Fixed parameters; not calibrated against any real sensor.
- **Training**: shipped checkpoint, 17 epochs of 320 synthetic patches (96 px), seed 42 (`weights/training_log.json`).
- **Evaluation set**: 50 freshly generated 160 px images (80 px after degradation), seed 999. "Held-out" here only means different random draws (and a different patch size from training, so no image is identical to a training image) from **the same generator**; it is in-distribution.

## Evaluation methodology

`scripts/benchmark.py` is a single deterministic command. Per image: degrade → quantize to 8-bit → (a) **baseline**: `cv2.resize(INTER_CUBIC)` ×2, no denoising; (b) **RestorNet-S** forward pass. Both outputs are quantized to 8-bit and compared with the 8-bit ground truth using `skimage` PSNR/SSIM (`data_range=1`, per image, single channel, default 7×7 SSIM window), then averaged.

## Reproduced results

Reproduced from scratch on CPU (`python scripts/benchmark.py`, run twice with byte-identical JSON output; matches the numbers previously claimed in this repo exactly). Checkpoint SHA-256 `eb5ea71b…0926`.

| Method | Evaluation set | PSNR (dB) | SSIM |
|---|---|---|---|
| Bicubic ×2 (no denoising) | 50 synthetic images, synthetic degradation, seed 999 | 20.63 ± 2.11 | 0.8835 ± 0.0459 |
| RestorNet-S | 50 synthetic images, synthetic degradation, seed 999 | 24.04 ± 2.17 | 0.9413 ± 0.0396 |

(mean ± std across images; model PSNR beats bicubic on 50/50 images.) Per-image values: `eval_results/benchmark_metrics.json`. These numbers say nothing about real images. The baseline does no denoising, so part of the gap is simply that the model denoises and bicubic does not; a "denoise then bicubic" baseline was not tested.

## Qualitative examples

`eval_results/benchmark_grid.png`: the first 8 evaluation images by index (no selection): degraded | bicubic | RestorNet-S | ground truth. Illustrative only, not statistical evidence. `sample_outputs/` holds an older 6-image set (seed 123, `scripts/make_samples.py`).

## Quick start

```bash
pip install -r requirements.txt            # CPU is enough; Python 3.10+
python -m pytest -q                        # 9 tests, ~7 s on CPU
python scripts/benchmark.py                # reproduces the table above (~5 s)
```

## Inference

```bash
python scripts/inference.py --input path/to/degraded --output path/to/out \
    --weights weights/restornet_s_final.pth     # CUDA used automatically if present, else CPU
```

Input: folder of images (read as grayscale; any size; `--tile N` for tiled inference). Output: `<name>_restored.png` at 2× size. Missing paths give a clear error.

## Training

```bash
python scripts/train.py --dataset synthetic --epochs 30 --steps-per-epoch 40 \
    --batch-size 8 --patch-size 96 --scale 2 --out weights/restornet_s_final.pth
```

**Caveat:** the shipped checkpoint was produced by an earlier version of this script. That version had a bug: with `num_workers=2` the synthetic dataset's RNG was copied into each worker, so (verified) half of every epoch was duplicated and every epoch repeated the same images, i.e. roughly 160 unique training images in total. This is fixed (`seed_worker`; synthetic runs also now use a fixed validation set for checkpoint selection), but the checkpoint was **not retrained**, so `train.py` as committed is not guaranteed to reproduce the shipped weights. The benchmark above is unaffected: it measures the checkpoint as shipped.

## Testing / CI

`tests/` (9 tests): forward shape and 2× scaling on CPU, input validation, deterministic degradation, dataset reproducibility, worker-seeding regression, metric sanity, checkpoint load + inference, benchmark smoke test, and a test that the full 50-image benchmark reproduces the committed numbers. GitHub Actions (`.github/workflows/ci.yml`) installs CPU torch, runs pytest and the benchmark. No GPU and no training in CI.

## Limitations

- Synthetic images and synthetic degradation only; domain shift to real images is unmeasured and likely large. No real inspection data validation.
- Small, simple generator; train and eval come from the same distribution. 50 evaluation images.
- Lightly trained checkpoint (17 short epochs, trained with the data-duplication bug above); no hyperparameter search, no ablations, no comparison against other learned models or a denoise+bicubic baseline.
- PSNR/SSIM are pixel/structure fidelity measures and can favor smooth outputs; no perceptual (LPIPS) or task-level (defect detection) metric was run.
- No latency, memory or deployment measurements were made or are claimed.

## Repository map

```
restornet/   model.py (network, load_checkpoint) · degradation.py · losses.py · dataset.py · metrics.py
scripts/     benchmark.py (evaluation) · train.py · inference.py · evaluate.py (folder-vs-folder metrics) · make_samples.py
tests/       test_restornet.py          .github/workflows/ci.yml
weights/     restornet_s_final.pth (3.5 MB) · training_log.json
eval_results/  benchmark_metrics.json · benchmark_table.md · benchmark_grid.png
```

MIT license.
