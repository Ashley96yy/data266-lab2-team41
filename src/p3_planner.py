"""First Planner checkpoint: validate input, screen requests, and choose a route.

This is a transparent CPU-only routing baseline. Vision-derived image text is
untrusted data and is never added to the user's request or used as an order.
The Vision, generation, and critique agents are implemented in later steps.
"""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_REQUEST_CHARS = 2_000
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}

_CREATE_OR_EDIT = re.compile(
    r"\b(create|generate|draw|render|make|edit|turn|transform|convert|change|"
    r"add|correct|reduce|brighten|enhance|denoise|sharpen|improve)\b",
    re.I,
)
_UNSAFE_CONTENT = re.compile(
    r"\b(sexually explicit|explicit sexual|sexual content|porn(?:ography|ographic)?|"
    r"sexuali[sz]ed|nude|nudity|nsfw|graphic gore|graphic violence|gore|"
    r"dismember\w*|mutilat\w*)\b",
    re.I,
)
_DECEPTIVE_CLAIM = re.compile(
    r"\b(arrest\w*|mugshot|criminal record|endorse\w*|fake testimonial|"
    r"confess\w*)\b",
    re.I,
)
_PERSON_REFERENCE = re.compile(
    r"\b(real person|this person|someone|person|portrait|photo|him|her|they)\b",
    re.I,
)
_DOMAIN_STYLE = re.compile(r"\b(naruto|shinobi|ninja anime)\b", re.I)
_STYLE = re.compile(
    r"\b(watercolor|ink sketch|sketch|painting|clay|stop.motion|cartoon|"
    r"illustration|art style|style change)\b",
    re.I,
)
_ENHANCE = re.compile(
    r"\b(brighten|lighting|color cast|colour cast|color correct\w*|"
    r"colour correct\w*|denois\w*|noise|exposure|contrast|white balance|"
    r"saturation|sharpen|enhanc\w*)\b",
    re.I,
)
_VISION = re.compile(
    r"\b(caption|describe|identify|read|what|which|who|where|when|"
    r"how many|how much|is there|are there)\b",
    re.I,
)


@dataclass(frozen=True)
class PlannerDecision:
    route: str
    status: str
    message: str
    image_info: dict[str, Any] | None
    untrusted_image_text_present: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable output for the later agent trace."""
        return asdict(self)


def _inspect_image(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "Image file is missing. Please upload a PNG, JPEG, or WebP image."
    size = path.stat().st_size
    if size == 0:
        return None, "The image file is empty. Please upload a valid image."
    if size > MAX_IMAGE_BYTES:
        return None, "The image file is too large for this demo."

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
                if width * height > MAX_IMAGE_PIXELS:
                    return None, "The image dimensions are too large for this demo."
                if image_format not in SUPPORTED_FORMATS:
                    return None, "Unsupported image format. Use PNG, JPEG, or WebP."
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return None, "The file is not a readable image. Please upload a valid image."

    return {
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": size,
    }, None


def plan_request(
    request: str,
    image_path: str | Path | None,
    *,
    image_text: str | None = None,
    real_person: bool | None = None,
) -> PlannerDecision:
    """Screen a request and return the next agent route without generating.

    ``image_text`` is supplied by a later OCR/Vision step. Its content is
    deliberately ignored for routing and safety authority. ``real_person``
    can be set by a trusted human or future vision check; unknown is treated
    conservatively for deceptive edit requests.
    """
    started = time.perf_counter()
    untrusted_text_present = bool(image_text)

    def decide(
        route: str,
        status: str,
        message: str,
        image_info: dict[str, Any] | None = None,
    ) -> PlannerDecision:
        return PlannerDecision(
            route=route,
            status=status,
            message=message,
            image_info=image_info,
            untrusted_image_text_present=untrusted_text_present,
            elapsed_seconds=round(time.perf_counter() - started, 6),
        )

    if not isinstance(request, str) or not request.strip():
        return decide("clarification", "needs_clarification", "What would you like me to do with the image?")
    request = " ".join(request.split())
    if len(request) > MAX_REQUEST_CHARS:
        return decide("input_error", "input_error", "Please shorten the request and try again.")

    asks_to_create = bool(_CREATE_OR_EDIT.search(request))
    if asks_to_create and _UNSAFE_CONTENT.search(request):
        return decide("refuse", "refused", "I cannot create that unsafe image content.")
    if (
        asks_to_create
        and _DECEPTIVE_CLAIM.search(request)
        and real_person is not False
        and (_PERSON_REFERENCE.search(request) or real_person is True)
    ):
        return decide("refuse", "refused", "I cannot create a deceptive edit of a real person.")

    if image_path is None:
        return decide("input_error", "input_error", "Please upload an image first.")
    try:
        info, error = _inspect_image(Path(image_path))
    except (OSError, TypeError, ValueError):
        return decide("input_error", "input_error", "The image file could not be opened.")
    if error:
        return decide("input_error", "input_error", error)

    if _DOMAIN_STYLE.search(request) and asks_to_create:
        return decide("domain_style_generation", "ready", "Use the team Part 2 style model.", info)
    if _STYLE.search(request) and asks_to_create:
        return decide("style_change_img2img", "ready", "Apply the requested visual style.", info)
    if _ENHANCE.search(request) and asks_to_create:
        return decide("enhancement", "ready", "Enhance the image while preserving its subject.", info)
    if _VISION.search(request) or ("?" in request and not asks_to_create):
        return decide("vision_only", "ready", "Send the image and question to Vision Understanding.", info)
    if asks_to_create:
        return decide(
            "clarification", "needs_clarification",
            "What specific style or edit would you like? No image has been generated.", info,
        )
    return decide(
        "unsupported", "unsupported",
        "I can caption or answer questions about an image, change its style, or enhance it.", info,
    )
