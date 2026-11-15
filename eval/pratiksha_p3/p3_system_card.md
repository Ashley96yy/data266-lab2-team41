# System card: DATA 266 Lab 2, Part 3 multi-agent image system

## Version
v2: after the first run (v1), the critic's score parsing was fixed, the critic compares the original with the result, and revisions change the generation settings. Scenarios, red-team cases and guardrails were unchanged.

## What it does
Describes images and answers questions about them, changes their style, edits them, and draws in Naruto anime style with the Part 2 LoRA. Five agents: Planner, Vision Understanding, Prompt Engineering, Image Generation and Critique, connected by plain Python functions sharing one typed state, with at most 2 revisions.

## Models
- Vision-language model (Planner, Vision, Prompt Engineering, Critique judgement): Qwen/Qwen2.5-VL-3B-Instruct (float16)
- Style change: stable-diffusion-v1-5/stable-diffusion-v1-5 image-to-image (strength 0.6); edit: timbrooks/instruct-pix2pix (image guidance 1.5)
- Naruto style: stable-diffusion-v1-5/stable-diffusion-v1-5 + PratikshaK-1/naruto-sd15-lora-r32 (image-to-image strength 0.65, or text-to-image)
- Critique score: openai/clip-vit-base-patch16 similarity (accept at or above 22.0) and VLM score (accept at or above 3/5)
- Sampler: 30 steps, guidance 7.5; seed 0 + revision number

## Results (15 scenarios)
- Task success: not available yet / 15 without critique, not available yet / 15 with critique
- Critic agreement with the team: not available yet
- Member agreement on pass/fail: not available yet (Cohen's kappa not available yet)
- Red-team attack success: 1 / 10
- Median seconds per request: 11.1 without critique, 16.8 with critique

## Safety
Guardrails: (1) the Planner refuses requests for nudity, sexual content, gore or violence (keyword filter plus VLM classification); (2) it transcribes text in the input image and refuses if that text tries to instruct the AI (pattern match plus VLM check), and Prompt Engineering treats image text as quoted data only; (3) for photos of real people it refuses edits that would change what the person looks like or does in a misleading way; (4) broken, empty, tiny or blank files return a clear message; (5) the Critique agent checks every output with a VLM safety question (the SD safety checker is recorded as a second signal) and never shows an unsafe image. Red team: 1 of 10 attacks succeeded (broken_file: 0/2, instruction_in_image: 0/3, real_person_edit: 1/3, unsafe_request: 0/2).

## Limitations
- A 3B vision-language model makes classification and judgement mistakes; refusals and critiques can be wrong in both directions.
- CLIP similarity is a weak measure of edit quality, and the Critique agent shares its VLM with the Planner, so it is not an independent judge.
- The Naruto LoRA is trained on copyrighted characters: non-commercial research use only. Deceptive edits of real people are refused when the real-person check detects a person, and that check can miss stylized or unusual photos.
