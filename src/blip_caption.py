"""
BLIP image captioning for video frames.

BLIP (Bootstrapping Language-Image Pretraining) generates natural-language
captions describing what's in an image. We caption each sampled video frame,
then embed those captions with MiniLM into the main 384-dim collection.

This is the complementary signal to raw CLIP embeddings:
    - CLIP path:    text query → CLIP text embedding → CLIP frame embedding
    - BLIP path:    BLIP caption → MiniLM embedding → text-query match in main collection

Two independent visual retrieval paths. If one misses, the other often catches.

Model: Salesforce/blip-image-captioning-base (~990 MB on first download).
"""

from typing import List
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# Max tokens for the generated caption. ~30 is enough for one descriptive sentence.
MAX_CAPTION_TOKENS = 30

# Lazy-loaded globals
_blip_processor: BlipProcessor | None = None
_blip_model: BlipForConditionalGeneration | None = None


def _load_blip():
    """Lazy-load BLIP processor + model."""
    global _blip_processor, _blip_model
    if _blip_model is None:
        print(f"[blip] Loading {BLIP_MODEL_NAME}")
        _blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        _blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
        _blip_model.eval()
        print(f"[blip] Loaded.")
    return _blip_processor, _blip_model


def caption_image(image: Image.Image, prompt: str | None = None) -> str:
    """
    Generate a caption for a single PIL image.

    Args:
        image: PIL Image (RGB).
        prompt: Optional conditioning prompt to guide the caption.
                e.g. "a photo of" makes BLIP more descriptive.
                If None, runs unconditional captioning.

    Returns:
        A natural-language caption string.
    """
    processor, model = _load_blip()
    if prompt:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_CAPTION_TOKENS)

    caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


def caption_image_batch(images: List[Image.Image], prompt: str | None = None) -> List[str]:
    """
    Generate captions for a list of PIL images.

    Slower than truly-batched inference because we call caption_image in a loop,
    but BLIP's batching with variable-length outputs is messy and the CPU
    workload dominates either way. Good enough for a demo on short videos.

    Returns:
        List of caption strings (same order as input images).
    """
    return [caption_image(img, prompt=prompt) for img in images]
