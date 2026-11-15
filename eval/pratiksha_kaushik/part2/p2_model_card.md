---
license: creativeml-openrail-m
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
datasets:
- lambdalabs/naruto-blip-captions
tags: [stable-diffusion, lora, text-to-image, diffusers]
---

# Naruto-style LoRA for Stable Diffusion 1.5 (rank 32)

LoRA adapter on the SD 1.5 UNet attention layers (q, k, v, out), trained on 859 cleaned images of
`lambdalabs/naruto-blip-captions` (90/10 split, seed 266). DATA 266 Lab 2, Part 2.

## Training
| | |
|---|---|
| trainable params | 6,377,472 |
| learning rate | 0.0001 (cosine, 100 warmup) |
| steps / batch | 3000 / 4 |
| resolution | 512 |
| GPU | Tesla T4 |
| GPU-hours | 1.372 |
| peak memory | 3.61 GB |

## Evaluation (DPMSolverMultistepScheduler, 30 steps, CFG 7.5, seeds 0 and 1)
| metric | base | this model |
|---|---|---|
| KID in-domain (lower better) | 0.1337 | 0.0217 |
| CLIP score (20 prompts) | 31.54 | 29.84 |
| LPIPS diversity | 0.808 | 0.749 |
| human win rate vs base (style / prompt) | - | 0.83 / 0.40 |

KID uses only 20 generated images, so it is noisy.

## Use
```python
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("PratikshaK-1/naruto-sd15-lora-r32")
pipe("a man with blue hair and a headband").images[0]
```

## Limitations and safety
- Training captions are BLIP-generated and generic; prompt following on out-of-domain and multi-character prompts is weak.
- Dataset license: not stated on the dataset card; images from narutopedia.com (Naruto is copyrighted), non-commercial research use only.
- Training images were screened with the SD safety checker (239 removed); a CLIP nearest-neighbour memorization check found: no sign of memorization: generations are no closer to the training set than unseen test images are.
