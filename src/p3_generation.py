"""Lazy SD 1.5 img2img backend; never substitutes a mock for real inference."""
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from PIL import Image, ImageOps
from src.p3_prompt import PromptDraft, GENERATION_ROUTES

BASE_MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
LORA_MODEL = 'PratikshaK-1/naruto-sd15-lora-r32'

@dataclass(frozen=True)
class GenerationSettings:
    seed: int = 266
    steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.55
    size: int = 512

    def __post_init__(self):
        if not 1 <= self.steps <= 100 or not 0 < self.strength <= 1:
            raise ValueError('Invalid steps or strength.')
        if not 0 <= self.guidance_scale <= 20 or self.size not in (512, 768):
            raise ValueError('Invalid guidance or image size.')
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError('Invalid seed.')

class DiffusionGenerator:
    def __init__(self, *, base_revision=None, lora_revision=None, device='cuda:0'):
        self.base_revision = base_revision
        self.lora_revision = lora_revision
        self.device = device
        self.pipe = None
        self.active_domain = False
        self.revisions = {}

    def _load(self, domain):
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from huggingface_hub import model_info
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for real image generation.')
        if self.pipe is None:
            revision = model_info(BASE_MODEL, revision=self.base_revision).sha
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                BASE_MODEL, revision=revision, torch_dtype=torch.float16).to(self.device)
            if self.pipe.safety_checker is None:
                raise RuntimeError('The diffusion safety checker must be enabled.')
            self.revisions['base'] = revision
        if domain and 'lora' not in self.revisions:
            revision = model_info(LORA_MODEL, revision=self.lora_revision).sha
            self.pipe.load_lora_weights(LORA_MODEL, revision=revision)
            self.revisions['lora'] = revision
        if 'lora' in self.revisions:
            if domain:
                self.pipe.enable_lora()
            else:
                self.pipe.disable_lora()
        self.active_domain = domain

    def generate(self, image_path, draft: PromptDraft, output_path, settings=GenerationSettings()):
        import torch
        if draft.route not in GENERATION_ROUTES:
            raise ValueError('Unsupported generation route.')
        self._load(draft.route == 'domain_style_generation')
        with Image.open(image_path) as source:
            image = ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'), (settings.size, settings.size))
        width, height = (max(8, n // 8 * 8) for n in image.size)
        image = image.resize((width, height))
        started = time.perf_counter()
        # CLIP has a short context window; put the actual requested edit first.
        prompt = draft.original_request + '. ' + draft.prompt.split('Revision:')[-1] if 'Revision:' in draft.prompt else draft.original_request + '. Preserve source subjects, counts, colors and composition.'
        token_ids = self.pipe.tokenizer(prompt, truncation=False).input_ids
        if len(token_ids) > self.pipe.tokenizer.model_max_length:
            raise ValueError('Generation prompt exceeds CLIP context; shorten the request.')
        output = self.pipe(prompt=prompt, negative_prompt=draft.negative_prompt,
                           image=image, strength=settings.strength,
                           num_inference_steps=settings.steps, guidance_scale=settings.guidance_scale,
                           generator=torch.Generator(device=self.device).manual_seed(settings.seed))
        flags = output.nsfw_content_detected
        safe = flags is not None and len(flags) == 1 and not bool(flags[0])
        metadata = dict(effective_prompt=prompt, negative_prompt=draft.negative_prompt, operation=draft.route, settings=asdict(settings), model_id=BASE_MODEL,
                        model_revision=self.revisions['base'],
                        lora_model_id=LORA_MODEL if self.active_domain else None,
                        lora_revision=self.revisions.get('lora') if self.active_domain else None,
                        safety_checked=flags is not None, safe=safe,
                        elapsed_seconds=time.perf_counter()-started, image_path=None)
        if safe:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            output.images[0].save(target)
            metadata['image_path'] = str(target)
        return metadata
