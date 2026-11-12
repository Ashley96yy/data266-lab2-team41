# Part 2: Fine-tuning Stable Diffusion 1.5 on Naruto images

DATA 266 Lab 2, Team 41, Part 2. This part fine-tunes Stable Diffusion 1.5 on the Naruto BLIP Captions dataset with two LoRA runs (rank 4 and rank 32), compares both with the base model on 20 fixed prompts, and publishes the adapters and all generated images on Hugging Face.

**Final model: Run 2, LoRA rank 32.** It has the best domain style (KID 0.0217 vs 0.1337 for the base model) at almost the same cost as rank 4.

## Hugging Face

| What | Link |
|---|---|
| Final model: LoRA rank 32 | [PratikshaK-1/naruto-sd15-lora-r32](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r32) |
| Comparison model: LoRA rank 4 | [PratikshaK-1/naruto-sd15-lora-r4](https://huggingface.co/PratikshaK-1/naruto-sd15-lora-r4) |
| All 120 generated images (base, run 1, run 2) with prompt, seed, CLIP score and NSFW flag | [PratikshaK-1/naruto-sd15-p2-generations](https://huggingface.co/datasets/PratikshaK-1/naruto-sd15-p2-generations) |

All notebook outputs (cleaned data, adapters, images, evaluation files and logs) are also in the [Part 2 Google Drive folder](https://drive.google.com/drive/folders/1f7xlLwBXvq8Yw1EkglcBFuaAUblfBniU?usp=drive_link).

Both model repos include a model card with the training settings, evaluation results, usage code and limitations.

### Use the final model

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("PratikshaK-1/naruto-sd15-lora-r32")
image = pipe("a man with blue hair and a headband").images[0]
```

## Notebooks

| Notebook | Status | Contents |
|---|---|---|
| [p2_v4.ipynb](p2_v4.ipynb) | **Submission, executed** | The whole of Part 2 in one notebook: baseline, Run 1, Run 2, final choice, grids, human and AI evaluation, failure modes, safety, model cards, Hugging Face upload |
| [p2_v3.ipynb](p2_v3.ipynb) | Earlier draft, not executed | Combined notebook before the final section was finished |
| [p2_v2.ipynb](p2_v2.ipynb) | Earlier draft, executed | First combined notebook (baseline, Run 1, Run 2) |
| [p2_baseline.ipynb](p2_baseline.ipynb) | Development, executed | 2.1 data preparation, fixed prompts, base-model metrics |
| [p2_run1_v1.ipynb](p2_run1_v1.ipynb) | Development, executed | Run 1 training and metrics |
| [p2_run2_v2.ipynb](p2_run2_v2.ipynb) | Development, executed | Run 2 training and metrics |

Graders should use **p2_v4.ipynb**. The other notebooks are kept to show how the work developed.

## How to run

1. Open `p2_v4.ipynb` in Google Colab with a GPU runtime (a Tesla T4 was used).
2. Optional, for uploading: add a Hugging Face write token as the Colab secret `HF_TOKEN`, and set `HF_USER` and `UPLOAD = True` in the config cell.
3. Run **Runtime → Restart session and run all**.

Notes:
- **Storage:** all outputs are saved to Google Drive under `MyDrive/data266-lab2/` ([shared Drive folder](https://drive.google.com/drive/folders/1f7xlLwBXvq8Yw1EkglcBFuaAUblfBniU?usp=drive_link)). The `eval/` and `work/` folders mentioned below are inside that Drive folder, not in this repository.
- **Reuse:** with `REUSE = True`, finished steps (cleaned data, trained adapters, generated images, human-eval pairs) are loaded from Drive. Metrics are recomputed on every run.
- **Retraining:** set `RETRAIN = True` to train again. Each run takes about 75 to 85 minutes on a T4.
- **Choosing the final run:** the notebook picks the run with the lowest KID. Set `FINAL_OVERRIDE = "run1"` or `"run2"` to choose the other one.

**Pinned packages** (installed by the first code cell): `diffusers==0.31.0`, `transformers==4.46.3`, `peft==0.13.2`, `accelerate==1.1.1`, `datasets==3.1.0`, `huggingface_hub==0.26.2`, `torchmetrics==1.6.0`, `torch-fidelity==0.3.0`, `lpips==0.1.4`, `imagehash==4.3.1`.

**Seeds:** 266 for the split, training and data loading. Images are generated with seeds 0 and 1.

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

Both runs use the same data, steps and settings. Only the LoRA rank differs. LoRA is applied to the UNet attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`). The text encoder and VAE are frozen.

| | Run 1 | Run 2 (final) |
|---|---|---|
| Method | LoRA r=4, alpha 4 | LoRA r=32, alpha 32 |
| Trainable parameters | 797,184 | 6,377,472 |
| Learning rate | 1e-4, cosine schedule, 100 warmup steps | same |
| Steps / batch size | 3,000 / 4 | 3,000 / 4 |
| Peak GPU memory (train) | 3.54 GB | 3.61 GB |
| Training time | 76.5 min | 82.3 min |
| GPU-hours (train) | 1.275 | 1.372 |
| GPU | Tesla T4 | Tesla T4 |

Memory savers: fp16 base weights, fp32 LoRA weights, gradient checkpointing and mixed precision.

## 2.3 Image generation

- **Prompts:** 20 fixed prompts saved to `eval/p2_prompts.json` before any training. 10 are in-domain (captions from the test set), 5 are out-of-domain and 5 are compositional.
- **Sampler:** DPM-Solver++, 30 steps, CFG 7.5, 512×512, seeds 0 and 1. This gives 40 images per model and 120 in total.
- **Grids:** side-by-side base vs Run 1 vs Run 2 grids for 8 prompts (ids 0, 1, 2, 3, 10, 12, 15, 16) are in `p2_v4.ipynb`.
- **Images:** all 120 are on Hugging Face (link above).

## 2.4 Evaluation

### Results: all three models

| Metric | Base | Run 1 (r=4) | Run 2 (r=32, final) |
|---|---|---|---|
| KID vs test images, lower is better | 0.1337 ± 0.0122 | 0.0389 ± 0.0077 | **0.0217 ± 0.0056** |
| CLIP score, all 20 prompts | **31.54** | 30.35 | 29.84 |
| CLIP, in-domain | **31.19** | 29.25 | 28.02 |
| CLIP, out-of-domain | **32.31** | 30.57 | 31.73 |
| CLIP, compositional | 31.46 | **32.35** | 31.59 |
| LPIPS between seeds (diversity), higher is more varied | **0.808** | 0.776 | 0.749 |
| NSFW-flagged generations (of 40) | 1 | 2 | 2 |
| Median seconds per image | 5.08 | 4.99 | 5.39 |
| GPU-hours (train) | — | 1.275 | 1.372 |
| Peak GPU memory, train / generate | — / 2.62 GB | 3.54 / 2.62 GB | 3.61 / 2.63 GB |

Notes on the metrics:
- **KID:** compares the 20 in-domain generations with the 123 test images, over 100 subsets. With this few images it is noisy, so the standard deviation is shown.
- **CLIP score:** uses `openai/clip-vit-base-patch16`.
- **LPIPS:** uses AlexNet, comparing seed 0 with seed 1 for each prompt.

### Human and AI-judge evaluation (final run vs base)

25 blinded pairs. Left and right are randomized with seed 266, and the answer key is kept separately. Raters choose the better image for **style** (more like Naruto) and for **prompt match**.

| Judge | Style win rate vs base | Prompt-match win rate vs base |
|---|---|---|
| Human raters (3 or more) | pending | pending |
| AI judge (CLIP ViT-L/14, not a human) | 0.92 | 0.20 |

The base model's win rate is 0.50 by definition. The AI judge is reported separately and doesn't replace the human ratings.

### Choice of final run

Run 2 (rank 32) is the final model:
- **Domain style:** it has the lowest KID, 0.0217 vs 0.0389 for Run 1 and 0.1337 for the base model (6.2× lower than base). Matching the Naruto style is the main goal of the fine-tune.
- **Trade-off:** its CLIP score is 0.52 lower than Run 1's (29.84 vs 30.35), and its diversity is lower (LPIPS 0.749 vs 0.776).
- **Cost:** it needed only slightly more compute: 1.37 vs 1.28 GPU-hours, 3.61 vs 3.54 GB peak memory, about 1.08× the training time.

### Failure modes

1. **Weaker prompt match on in-domain prompts.** In-domain CLIP falls from 31.19 to 28.02 (−3.17). The worst case is "a woman in a pink dress and brown jacket" (CLIP 23.6).
2. **Style leaking into out-of-domain prompts.** Out-of-domain CLIP drops by 0.59 (32.31 to 31.73), so prompts with no Naruto content (cars, lighthouse, fruit) follow the text slightly less well.
3. **Weak compositional prompts.** "A man holding a sword in his left hand and a scroll in his right hand" scores CLIP 26.4, below the Run 2 median of 29.1.
4. **Less diversity between seeds.** Mean LPIPS falls from 0.808 to 0.749. The least varied prompt is p12 (0.624 vs 0.716 for base).
5. **Dark side bands.** 4.8% of training images have black borders, and 7 of 40 Run 2 generations have a dark band on one side (8 for Run 1, 10 for base).

## Safety

- **License:**
  - Dataset: not stated on the dataset card. The images show copyrighted Naruto characters, so this work is non-commercial research only.
  - Base model: CreativeML OpenRAIL-M, which forbids harmful uses.
- **NSFW filter:** the SD safety checker removed 239 of 1,098 training images and was also run on all 120 generations (base 1, Run 1 2, Run 2 2 flagged). The flag is stored in the Hugging Face image dataset.
- **Memorization check:** for the 10 in-domain Run 2 generations (seed 0), the closest training image has CLIP ViT-L/14 similarity between 0.614 and 0.889 (median 0.752). Real test images, which the model never saw, reach up to 0.967 (median 0.891). None of the 10 generations is closer to the training set than the closest test image, and none exceeds 0.95. **No sign of memorization.**
- **Train/test overlap:** 4 of the 123 test images have a near duplicate in the training set (similarity > 0.95). The perceptual-hash de-duplication didn't catch this small overlap.

## Limitations

- Captions are generated by BLIP, short (about 9 words) and generic. They rarely name the character, so the model learns the art style more than character identities.
- The data is unbalanced. "A man" begins 424 of 859 training captions, and most images are single-character close-ups, which makes multi-character and compositional prompts harder.
- KID is computed from only 20 generated images, so small differences between runs should be read with care.

## Outputs saved by the notebook (in Google Drive)

All of these files are in the [Part 2 Google Drive folder](https://drive.google.com/drive/folders/1f7xlLwBXvq8Yw1EkglcBFuaAUblfBniU?usp=drive_link).

| File | Contents |
|---|---|
| `eval/p2_data_cleaning.json` | Cleaning counts, removed ids, split seed, license |
| `eval/p2_prompts.json` | The 20 fixed prompts |
| `eval/p2_sampler.json` | Sampler settings |
| `eval/p2_training_table.csv` | Training table for both runs |
| `eval/p2_results_all_models.csv` | Results table for base, Run 1 and Run 2 |
| `eval/p2_human_key.csv` | Answer key for the 25 blinded pairs |
| `eval/p2_human_ratings.csv` | Human ratings (to be filled in) |
| `eval/p2_analysis.md` | Failure modes, final choice, memorization and safety notes |
| `eval/p2_model_card.md` | Model card for the final run |
| `logs/p2_run_log.txt` | Time-stamped log of every step |

## Remaining work

- [ ] Collect the human ratings (3 or more raters, 25 pairs) in `eval/p2_human_ratings.csv`, re-run from the ratings cell, and fill in the human win rates above and in the model card.
- [ ] Make the three Hugging Face repos public, or share access with the graders. The notebook creates them as private.
