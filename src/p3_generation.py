"""Lazy SD 1.5 img2img backend; never substitutes a mock for real inference."""
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from PIL import Image, ImageOps
from src.p3_prompt import PromptDraft, GENERATION_ROUTES, compile_diffusion_prompt, monochrome_requested

BASE_MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
LORA_MODEL = 'PratikshaK-1/naruto-sd15-lora-r32'

@dataclass(frozen=True)
class GenerationSettings:
    seed: int = 266
    steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.7
    size: int = 512
    domain_strength: float = 0.55
    lora_scale: float = 0.65

    def __post_init__(self):
        if not 1 <= self.steps <= 100 or not 0 < self.strength <= 1:
            raise ValueError('Invalid steps or strength.')
        if not 0 <= self.guidance_scale <= 20 or self.size not in (512, 768):
            raise ValueError('Invalid guidance or image size.')
        if not 0<self.domain_strength<=1 or not 0<=self.lora_scale<=1.5:
            raise ValueError('Invalid domain strength or LoRA scale.')
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError('Invalid seed.')

class DiffusionGenerator:
    def __init__(self, *, base_revision=None, lora_revision=None, device='cuda:0',
                 cache_dir='.runtime/model-cache'):
        self.base_revision = base_revision
        self.lora_revision = lora_revision
        self.device = device
        self.cache_dir = str(cache_dir)
        self.pipe = None
        self.active_domain = False
        self.revisions = {}

    def _load(self, domain, lora_scale=0.65):
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from huggingface_hub import model_info
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for real image generation.')
        if self.pipe is None:
            revision = self.base_revision if self.base_revision and len(self.base_revision)==40 else model_info(BASE_MODEL, revision=self.base_revision).sha
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                BASE_MODEL, revision=revision, torch_dtype=torch.float16,
                variant='fp16', use_safetensors=True, cache_dir=self.cache_dir).to(self.device)
            if self.pipe.safety_checker is None:
                raise RuntimeError('The diffusion safety checker must be enabled.')
            self.revisions['base'] = revision
        if domain and 'lora' not in self.revisions:
            revision = self.lora_revision if self.lora_revision and len(self.lora_revision)==40 else model_info(LORA_MODEL, revision=self.lora_revision).sha
            self.pipe.load_lora_weights(LORA_MODEL, revision=revision,
                                        cache_dir=self.cache_dir,adapter_name='team_domain')
            self.revisions['lora'] = revision
        if 'lora' in self.revisions:
            if domain:
                self.pipe.set_adapters('team_domain',adapter_weights=lora_scale)
                self.pipe.enable_lora()
            else:
                self.pipe.disable_lora()
        self.active_domain = domain

    def generate(self, image_path, draft: PromptDraft, output_path, settings=GenerationSettings()):
        import torch
        if draft.route not in GENERATION_ROUTES:
            raise ValueError('Unsupported generation route.')
        self._load(draft.route == 'domain_style_generation',settings.lora_scale)
        with Image.open(image_path) as source:
            original = ImageOps.exif_transpose(source).convert('RGB')
        started = time.perf_counter()
        compiled=compile_diffusion_prompt(draft,self.pipe.tokenizer)
        prompt=compiled['effective_prompt']
        operations=[]
        postprocessing=[]
        effective_strength=settings.domain_strength if draft.route=='domain_style_generation' else settings.strength
        if draft.route=='enhancement':
            import numpy as np
            from src.p3_enhancement import enhance_image
            enhanced,operations=enhance_image(original,draft.original_request,draft.revision_index)
            with torch.inference_mode():
                _,flags=self.pipe.run_safety_checker(np.asarray(enhanced,dtype=np.float32)[None]/255,
                                                    torch.device(self.device),torch.float16)
            images=[enhanced]
        else:
            image=ImageOps.contain(original,(settings.size,settings.size))
            width,height=(max(8,n//8*8) for n in image.size)
            image=image.resize((width,height))
            output = self.pipe(prompt=prompt, negative_prompt=draft.negative_prompt,
                           image=image, strength=effective_strength,
                           num_inference_steps=settings.steps, guidance_scale=settings.guidance_scale,
                           generator=torch.Generator(device=self.device).manual_seed(settings.seed))
            flags=output.nsfw_content_detected
            images=output.images
            if flags is not None and len(flags)==1 and not bool(flags[0]) and monochrome_requested(draft.original_request):
                import numpy as np
                images=[ImageOps.grayscale(images[0]).convert('RGB')]
                _,flags=self.pipe.run_safety_checker(np.asarray(images[0],dtype=np.float32)[None]/255,
                                                    torch.device(self.device),torch.float16)
                postprocessing.append('monochrome_explicitly_requested_by_user')
        safe = flags is not None and len(flags) == 1 and not bool(flags[0])
        metadata = dict(**compiled, negative_prompt=draft.negative_prompt, operation=draft.route, settings=asdict(settings),
                        backend='pillow_photometric' if draft.route=='enhancement' else 'stable_diffusion_img2img',
                        applied_diffusion_steps=0 if draft.route=='enhancement' else max(1,int(settings.steps*effective_strength)),
                        effective_strength=effective_strength if draft.route!='enhancement' else None,
                        lora_scale=settings.lora_scale if self.active_domain else None,postprocessing=postprocessing,
                        enhancement_operations=operations,revision_index=draft.revision_index,
                        model_id=BASE_MODEL if draft.route!='enhancement' else None,
                        model_revision=self.revisions['base'] if draft.route!='enhancement' else None,
                        safety_checker_model_id=BASE_MODEL,safety_checker_model_revision=self.revisions['base'],
                        lora_model_id=LORA_MODEL if self.active_domain else None,
                        lora_revision=self.revisions.get('lora') if self.active_domain else None,
                        safety_checked=flags is not None, safe=safe,
                        elapsed_seconds=time.perf_counter()-started, image_path=None)
        if safe:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(target)
            metadata['image_path'] = str(target)
        return metadata
