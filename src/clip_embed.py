"""
CLIP image and text encoder for cross-modal visual search.

Uses open_clip with the ViT-B-32 architecture and the openai pretrained weights.
Image and text both map to the same 512-dim normalized embedding space, so a
text query like "a person typing on a laptop" can directly retrieve image frames
without going through a caption.

This is the "raw visual signal" path. The BLIP caption path (blip_caption.py)
gives a complementary signal: BLIP describes the scene in natural language,
which then enters the main text collection. Searching both gives us two
independent ways to retrieve visual content.
"""

from typing import List, Union
import numpy as np
import torch
from PIL import Image
import open_clip


CLIP_ARCH = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_EMBEDDING_DIM = 512

# Lazy-loaded globals
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _load_clip():
    """Lazy-load CLIP model, preprocessor, and tokenizer."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        print(f"[clip] Loading {CLIP_ARCH} ({CLIP_PRETRAINED})")
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_ARCH, pretrained=CLIP_PRETRAINED
        )
        model.eval()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = open_clip.get_tokenizer(CLIP_ARCH)
        print(f"[clip] Loaded. Embedding dim: {CLIP_EMBEDDING_DIM}")
    return _clip_model, _clip_preprocess, _clip_tokenizer


def embed_image(image: Image.Image) -> np.ndarray:
    """
    Encode a PIL image to a 512-dim L2-normalized CLIP embedding.

    Normalization is critical: cosine distance only behaves correctly on
    unit-norm vectors.
    """
    model, preprocess, _ = _load_clip()
    img_tensor = preprocess(image).unsqueeze(0)  # shape (1, 3, 224, 224)
    with torch.no_grad():
        features = model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
    return features.squeeze(0).cpu().numpy().astype(np.float32)


def embed_image_batch(images: List[Image.Image]) -> np.ndarray:
    """
    Encode a list of PIL images to 512-dim normalized embeddings.

    More efficient than calling embed_image in a loop because of batched
    GPU/CPU tensor ops. Returns array of shape (N, 512).
    """
    if not images:
        return np.zeros((0, CLIP_EMBEDDING_DIM), dtype=np.float32)

    model, preprocess, _ = _load_clip()
    tensors = torch.stack([preprocess(img) for img in images])  # (N, 3, 224, 224)
    with torch.no_grad():
        features = model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def embed_clip_text(text: Union[str, List[str]]) -> np.ndarray:
    """
    Encode text into CLIP's 512-dim space (shared with image embeddings).

    Use this for text-to-image search ONLY. Do not use this for searching
    the main 384-dim text collection — that uses MiniLM (see embeddings.py).

    Args:
        text: A single string or list of strings.

    Returns:
        - single string: 1D array shape (512,)
        - list of N strings: 2D array shape (N, 512)
    """
    model, _, tokenizer = _load_clip()
    is_single = isinstance(text, str)
    inputs = [text] if is_single else text
    tokens = tokenizer(inputs)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    arr = features.cpu().numpy().astype(np.float32)
    return arr[0] if is_single else arr
