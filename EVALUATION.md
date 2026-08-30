# Evaluation: RestorNet-S vs. Bicubic Baseline

## Method

Both RestorNet-S and a bicubic-upsampling baseline were evaluated on the
**same 50 held-out synthetic test images** (seed=999, not used during
training), using the repo's own `scripts/evaluate.py` for both, so PSNR
and SSIM are computed identically for each — the only variable is the
restoration method itself.

- **Baseline**: the degraded (80×80) input, upsampled to 160×160 via
  OpenCV bicubic interpolation (`cv2.INTER_CUBIC`) — no denoising, no
  learned model, the standard classical baseline for this task.
- **RestorNet-S**: the trained model (`weights/restornet_s_final.pth`)
  performing joint denoising + 2× super-resolution in a single forward
  pass.

Reproduce with:
```bash
python scripts/make_samples.py --weights weights/restornet_s_final.pth --n 50 --seed 999 --out-dir eval_50
# then upsample eval_50/degraded/*.png via bicubic into a separate folder, and:
python scripts/evaluate.py --restored eval_50/restored     --ground-truth eval_50/ground_truth --no-lpips
python scripts/evaluate.py --restored <bicubic_folder>     --ground-truth eval_50/ground_truth --no-lpips
```

## Results (50-image test set)

| Method                | PSNR (dB) | SSIM   |
|------------------------|-----------|--------|
| Bicubic baseline        | 20.63     | 0.8835 |
| **RestorNet-S**         | **24.04** | **0.9413** |
| **Improvement**         | **+3.41 dB** | **+0.058 (+6.5% relative)** |

## Interpretation

RestorNet-S improves PSNR by 3.41 dB and SSIM by 6.5% (relative) over
bicubic upsampling alone, on a held-out set the model was not evaluated
against during development. The gain is consistent with the model's
design goal: bicubic upsampling only resizes and does nothing about the
speckle/Gaussian noise in the degraded input, while RestorNet-S performs
denoising and super-resolution jointly in one forward pass.
