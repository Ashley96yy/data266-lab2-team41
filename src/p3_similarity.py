"""CPU CLIP image/text cosine similarity; diagnostic, not a safety/pass label."""
from pathlib import Path
from PIL import Image

CLIP_MODEL = 'openai/clip-vit-base-patch32'
CLIP_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'


class ClipSimilarity:
    def __init__(self, cache_dir='.runtime/part1-model-cache'):
        self.cache_dir = Path(cache_dir)
        self.processor = self.model = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        snapshot = self.cache_dir / ('models--'+CLIP_MODEL.replace('/','--')) / 'snapshots' / CLIP_REVISION
        location = str(snapshot) if snapshot.is_dir() else CLIP_MODEL
        self.processor = CLIPProcessor.from_pretrained(location, revision=CLIP_REVISION, cache_dir=str(self.cache_dir), use_fast=False)
        self.model = CLIPModel.from_pretrained(location, revision=CLIP_REVISION, cache_dir=str(self.cache_dir), use_safetensors=False).to('cpu').eval()

    def score(self, image_path, text):
        import torch
        self._load()
        tokens = self.processor.tokenizer(text, truncation=False).input_ids
        limit = self.processor.tokenizer.model_max_length
        with Image.open(image_path) as source:
            inputs = self.processor(text=[text], images=[source.convert('RGB')],
                return_tensors='pt', padding=True, truncation=True, max_length=limit)
        with torch.inference_mode():
            output = self.model(**inputs)
            similarity = (output.image_embeds * output.text_embeds).sum(-1).item()
        return dict(clip_cosine_similarity=float(similarity), clip_model_id=CLIP_MODEL,
                    clip_model_revision=CLIP_REVISION, clip_device='cpu',
                    clip_text=text, clip_text_truncated=len(tokens)>limit,
                    clip_text_token_limit=limit,
                    interpretation='Unscaled cosine similarity; no acceptance or safety threshold.')

    def get_tokenizer(self):
        self._load()
        return self.processor.tokenizer
