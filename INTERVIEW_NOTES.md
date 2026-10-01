# Interview notes: RestorNet-S (supporting project)

Scope to state up front: a small PyTorch project on **synthetic** images with **synthetic** degradation. It shows I can build, evaluate and honestly report an ML experiment; it is not a claim about real-world imaging.

**Why combine denoising and super-resolution?** The degradation (noise, then downsampling) is one pipeline, so one network can invert both in a single pass and avoid a separate denoiser's errors being amplified by an upscaler. I did not compare against a two-stage pipeline, so I can't claim it's better.

**Why this architecture?** Residual dense blocks with channel attention are a common SR design (RDN/RCAN family). I used it because it is well understood and small (868K params). It is not an encoder-decoder (no downsampling). I ran no ablations, so I can't say attention or the dilated branch helps.

**PixelShuffle?** Sub-pixel convolution: a conv produces r² × C channels at low resolution and PixelShuffle rearranges them into an r× larger spatial grid. Learned upsampling with no checkerboard-prone transposed conv.

**Residual learning?** The network predicts a correction, here to a fixed bicubic upsample of the input (global skip), plus skips inside the trunk. Easier optimization, since identity-like mapping is the default.

**Losses?** L1 for pixel fidelity, a multi-scale SSIM term for structure, a Sobel edge term to penalize blurred gradients. Weights 1/1/0.5 were chosen by hand; not tuned or ablated. The SSIM term is a simplified pyramid average, not true MS-SSIM.

**PSNR / SSIM?** PSNR = 10·log10(1/MSE) for images in [0,1]; pixelwise error. SSIM compares local luminance, contrast and structure in windows.

**Why can they mislead?** PSNR rewards smooth, blurry outputs and ignores perception; SSIM is window-based and insensitive to some artifacts. Neither measures whether a downstream task (e.g. defect detection) still works. Also, a model with a bicubic skip can score well on synthetic data that is easy.

**Why bicubic baseline?** It's the standard no-learning upscaler, cheap and unambiguous: same 80×80 input, `cv2` bicubic to 160×160. It does no denoising, so it's a weak baseline for a noisy input; a denoise-then-bicubic baseline would be fairer and I haven't run it.

**What was synthetic?** Both the clean images (procedural grids/stripes/blobs) and the degradation (speckle σ=0.15, Gaussian σ=0.02, bicubic ↓2). Training and test come from the same generator.

**What does the evaluation prove?** On 50 in-distribution synthetic images the shipped checkpoint scores 24.04 dB / 0.9413 vs bicubic 20.63 dB / 0.8835, winning on 50/50 images, and the result is reproducible byte-for-byte on CPU. **Not proven:** generalization to real images, benefit of any component, perceptual quality, speed.

**What would I need for real usefulness?** Real paired (or realistically simulated, sensor-calibrated) data, held-out by wafer/session, a downstream-task metric, comparison with stronger baselines, and out-of-distribution tests.

**Improve generalization?** Train on real or calibrated-noise data, randomize degradation parameters, augmentation, a validation split matching deployment, ablations, and early stopping on validation.

**Deploy/optimize?** Fully convolutional so any size; tiled inference exists. Next steps would be ONNX/TorchScript export, fp16/int8, batching, and benchmarking latency on target hardware. I have not measured any of that.

**A bug I found and fixed?** With multi-worker DataLoaders the synthetic dataset's RNG was copied per worker, so samples were duplicated and repeated every epoch (verified by counting unique samples). Fixed with a `worker_init_fn` and a regression test; I documented that the shipped checkpoint predates the fix and wasn't retrained.
