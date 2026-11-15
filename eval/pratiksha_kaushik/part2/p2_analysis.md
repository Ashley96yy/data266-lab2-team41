## Failure modes

1. Weaker prompt alignment on in-domain prompts. CLIP in-domain falls from 31.19 (base) to 28.02 (-3.18), so images follow the caption text less closely after fine-tuning; the worst case is "a woman in a pink dress and brown jacket" (CLIP 23.6).
2. Style leaking into out-of-domain prompts. CLIP on out-of-domain prompts drops by 0.59 (32.31 to 31.73), so prompts with no Naruto content (cars, lighthouse, fruit) follow the text slightly less well than with the base model.
3. Weak compositional prompts. The lowest-scoring compositional prompt for run2 is "a man holding a sword in his left hand and a scroll in his right hand" (CLIP 26.4, below the run2 median of 29.1); compositional mean 31.59 vs base 31.46. See the compositional grid above.
4. Less diversity between seeds. Mean LPIPS between seed 0 and seed 1 falls from 0.808 to 0.749; the least varied prompt is p12 (LPIPS 0.624 vs 0.716 for base).

## Chosen run and why

- Final run: run2 (LoRA rank 32, 6,377,472 trainable parameters).
- Domain style: KID 0.0217 for run2 vs 0.0389 for run1 and 0.1337 for base (6.2x lower than base).
- Prompt match (CLIP, all 20 prompts): 29.84 for run2 vs 30.35 for run1.
- Diversity (LPIPS): 0.749 for run2 vs 0.776 for run1.
- Cost: 1.37 vs 1.28 GPU-hours, 3.61 vs 3.54 GB peak memory.
- Trade-off: run2 has the lowest KID (best Naruto style, the main goal of the fine-tune) but 0.52 lower CLIP than run1; it took 1.08x the training time of run1.
- Human evaluation: raters preferred run2 over base in 83% of votes for style and 40% for prompt match.
- AI judge (CLIP, automatic, not a human rater): run2 preferred over base in 92% of pairs for style and 20% for prompt match; it agrees with the human majority on 84% (style) and 64% (prompt) of pairs.

## Memorization

For the 10 in-domain generations of run2 (seed 0), the closest training image has CLIP similarity between 0.614 and 0.889 (median 0.752). Real test images, which the model never saw, reach up to 0.967 (median 0.891). 0 of 10 generations are closer to the training set than the closest test image, and 0 exceed 0.95. Conclusion: no sign of memorization: generations are no closer to the training set than unseen test images are. Note: 4 test images have a near-duplicate in the training set (sim > 0.95), a small overlap the perceptual-hash de-duplication did not catch.

## Safety

License: not stated on the dataset card; images from narutopedia.com (Naruto is copyrighted), non-commercial research use only. Base model: CreativeML OpenRAIL-M. NSFW: the SD safety checker removed 239 of 1098 training images (21.8%). This rate is high for an anime dataset; the CLIP-based checker is known to be conservative on drawn characters, so some flags are likely false positives (not verified by hand). Generations flagged: base 1/40, run1 2/40, run2 2/40.