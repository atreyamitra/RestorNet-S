# Resume / recruiter material (evidence-backed only)

**Resume bullets**
- Built a PyTorch residual-dense-block network (channel attention, PixelShuffle head, bicubic global skip; 868K params) for joint denoising + 2× super-resolution, with a hybrid L1/SSIM/edge loss and CPU inference CLI.
- Made the evaluation reproducible: deterministic benchmark vs. bicubic on 50 synthetically degraded images (24.04 vs 20.63 dB PSNR, 0.941 vs 0.884 SSIM), plus pytest suite and GitHub Actions CI; found and fixed a DataLoader RNG bug that duplicated training samples.

**LinkedIn bullets**
- Implemented and evaluated a small PyTorch image-restoration model (denoising + 2× super-resolution) against a bicubic baseline on synthetic data, reporting results as synthetic-only.
- Added tests, CPU-only CI and a one-command reproducible benchmark; audited and corrected overstated README claims.

**One-line description:** PyTorch denoising + 2× super-resolution CNN with a reproducible, CI-tested benchmark against bicubic on synthetic data.

**30-second pitch:** "RestorNet-S is a small side project where I built a PyTorch model that denoises and 2×-upscales grayscale images in one pass, and I focused on doing the evaluation honestly. Everything is synthetic: generated images and simulated noise. On 50 held-out synthetic images it beats bicubic by about 3.4 dB PSNR, and anyone can reproduce that with one command on CPU. I also audited the repo, corrected claims that said more than the evidence supported, found a data-loader bug that was duplicating training samples, and added tests and CI. I wouldn't claim it works on real images; that would need real data."

**Do NOT use (removed from the old README):** "KLA track / inspection-ready" framing; "physics-matched / matching real inspection-camera degradation"; "hierarchical residual encoder-decoder" (no encoder-decoder exists); "+6.5% relative SSIM" and "improvement on held-out set the model was not evaluated against during development" phrasing; "Target metrics SSIM ≥ 0.90–0.94, PSNR ≥ 30–34 dB, <few seconds, zero extra hardware, deployable on existing tooling" (never measured on real data); "learnable global residual" (it's fixed bicubic); "preserves DRAM/FinFET periodic structures" (untested).
