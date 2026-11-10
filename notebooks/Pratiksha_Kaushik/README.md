# Part 2: Fine-tuning Stable Diffusion 1.5 on Naruto images

This part fine-tunes Stable Diffusion 1.5 on the Naruto BLIP Captions dataset in two runs: LoRA with rank 4 and LoRA with rank 32. Both runs are compared with the base model on 20 fixed prompts.

## Notebooks (run in this order)

| # | Notebook | What it does |
|---|---|---|
| 1 | [p2_baseline.ipynb](p2_baseline.ipynb) | 2.1 data preparation, writes the 20 fixed prompts, generates base-model images, computes baseline metrics |
| 2 | [p2_run1_v1.ipynb](p2_run1_v1.ipynb) | Run 1: trains LoRA rank 4, then generates its images and computes its metrics |
| 3 | [p2_run2_v2.ipynb](p2_run2_v2.ipynb) | Run 2: trains LoRA rank 32 with the same setup, then generates its images and computes its metrics |

Notebooks 2 and 3 both read the files that notebook 1 saves, so run notebook 1 first. Notebooks 2 and 3 don't depend on each other.

## Setup

- **Environment:** Google Colab with a Tesla T4 GPU. Files are saved to Google Drive under `MyDrive/data266-lab2/`.
- **Packages:** each notebook installs its pinned versions in its first code cell: `diffusers==0.31.0`, `transformers==4.46.3`, `peft==0.13.2`, `accelerate==1.1.1`, `datasets==3.1.0`, `huggingface_hub==0.26.2`, `torchmetrics==1.6.0`, `torch-fidelity==0.3.0`, `lpips==0.1.4`. The baseline notebook also installs `imagehash==4.3.1`.
- **Seed:** 266 for the split, training and data loading. Images are generated with seeds 0 and 1.
- **Retraining:** `RETRAIN = False` reloads the adapter saved in Drive. Set it to `True` to train again (about 75 to 85 minutes per run on a T4).
- **Hugging Face upload:** set `HF_USER` to your username and `UPLOAD = True`, and store a `HF_TOKEN` in Colab secrets.

## 2.1 Data preparation

- **Dataset:** [`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions), 1,221 images with captions written automatically by BLIP.
- **License:** the dataset card doesn't state one. The images come from narutopedia.com and Naruto is copyrighted, so this work is for non-commercial research use only.

| Step | Count |
|---|---|
| Original images | 1,221 |
| Corrupted or empty caption | 0 |
| Exact duplicates (MD5) | 0 |
| Near duplicates (perceptual hash ≤ 2 bits) | 0 |
| Train / test after 90/10 split (seed 266) | 1,098 / 123 |
| Train images removed by NSFW filter | 239 (21.8%) |
| **Final train / test** | **859 / 123** |

All images are center-cropped and resized to 512×512. The NSFW filter is the Stable Diffusion safety checker. It is known to be conservative on drawn characters, so many of the 239 flags are probably false positives.

## 2.2 Fine-tuning

Both runs use the same data, steps and settings. LoRA is applied to the UNet attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`). The text encoder and VAE are frozen.

| | Run 1 | Run 2 |
|---|---|---|
| Method | LoRA r=4, alpha 4 | LoRA r=32, alpha 32 |
| Trainable parameters | 797,184 | 6,377,472 |
| Learning rate | 1e-4, cosine schedule, 100 warmup steps | same |
| Steps / batch size | 3,000 / 4 | 3,000 / 4 |
| Peak GPU memory (train) | 3.54 GB | 3.61 GB |
| Training time | 76.5 min (1.275 GPU-hours) | 82.3 min (1.372 GPU-hours) |
| GPU | Tesla T4 | Tesla T4 |

Memory savers: fp16 base weights, fp32 LoRA weights, gradient checkpointing and mixed precision.

## 2.3 Image generation

- **Prompts:** 20 fixed prompts saved to `eval/p2_prompts.json` before any training. 10 are in-domain (captions from the test set), 5 are out-of-domain and 5 are compositional.
- **Sampler:** DPM-Solver++, 30 steps, CFG 7.5, 512×512, seeds 0 and 1. This gives 40 images per model.
- **Grids:** side-by-side base vs run grids for 8 prompts (ids 0, 1, 2, 3, 10, 12, 15, 16) are in notebooks 2 and 3.

## 2.4 Results

| Metric | Base | Run 1 (r=4) | Run 2 (r=32) |
|---|---|---|---|
| KID vs test images, lower is better | 0.1337 ± 0.0122 | 0.0389 ± 0.0077 | **0.0217 ± 0.0056** |
| CLIP score, all prompts | **31.54** | 30.35 | 29.84 |
| CLIP, in-domain | **31.19** | 29.25 | 28.02 |
| CLIP, out-of-domain | **32.31** | 30.57 | 31.73 |
| CLIP, compositional | 31.46 | **32.35** | 31.59 |
| LPIPS between seeds (diversity), higher is more varied | **0.808** | 0.776 | 0.749 |
| NSFW-flagged generations (of 40) | 1 | 2 | 2 |
| Median seconds per image | 5.08 | 4.99 | 5.39 |
| Peak GPU memory (generate) | 2.62 GB | 2.62 GB | 2.63 GB |

KID uses the 20 in-domain generations against the 123 test images, with 100 subsets. With this few images it is noisy, so the standard deviation is shown.

**Summary:** both runs move much closer to the Naruto style than the base model (KID drops by 71% for Run 1 and 84% for Run 2). The cost is lower CLIP prompt match and lower diversity. Run 2 matches the style better but loses more diversity and in-domain prompt match. It keeps more of the out-of-domain prompt match than Run 1.

## Status

Done:
- [x] 2.1 Data cleaning, split, resize and NSFW filter
- [x] 2.2 Two training runs with parameters, LR, steps, batch size, peak memory and time
- [x] 2.3 40 images per model and side-by-side grids
- [x] 2.4 KID, CLIP, LPIPS, GPU-hours and peak memory

Still to do:
- [ ] Push both adapters, with model cards, and all generated images to Hugging Face, then add the links here
- [ ] Blinded human evaluation: 25 pairs, 3 or more raters, win rate
- [ ] Memorization check of generated images against training images
- [ ] Write up two failure modes and which run is chosen, and why
- [ ] Re-run inside a git checkout so the reproducibility cell prints the commit (it currently prints "no git repo here")
