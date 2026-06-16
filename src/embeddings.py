"""
Text embedding module using sentence-transformers/all-MiniLM-L6-v2.

This is the core embedding function used across the entire system:
- Text files are embedded directly
- Audio is transcribed by Whisper, then transcripts are embedded here
- Video frames go through CLIP separately, but video captions/transcripts
  also flow through here

Model: all-MiniLM-L6-v2
Output: 384-dimensional dense vectors (numpy array)
"""

from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer

# Module-level model cache — load once, reuse forever.
# Loading is expensive (~2-5s); embedding a string is fast (~10ms on CPU).
_model: SentenceTransformer | None = None

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model on first use."""
    global _model
    if _model is None:
        print(f"[embeddings] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[embeddings] Model loaded. Embedding dim: {EMBEDDING_DIM}")
    return _model


def embed_text(text: Union[str, List[str]]) -> np.ndarray:
    """
    Embed a string or list of strings into 384-dim vectors.

    Args:
        text: A single string OR a list of strings.

    Returns:
        - If input is a single string: 1D numpy array of shape (384,)
        - If input is a list of N strings: 2D numpy array of shape (N, 384)
    """
    model = get_model()
    # convert_to_numpy=True gives us a numpy array directly (no torch tensors).
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding
