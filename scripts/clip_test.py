"""Test CLIP: load model, embed an image and a text query, check similarity."""
import numpy as np
from src.video_ingest import sample_frames
from src.clip_embed import embed_image, embed_clip_text, CLIP_EMBEDDING_DIM

print("Sampling one frame from video...")
frames = sample_frames("data/video/workstation_ai.mp4")
ts, img = frames[len(frames) // 2]  # middle frame
print(f"Picked frame at t={ts:.1f}s, size={img.size}")

print()
print("Embedding image with CLIP...")
img_emb = embed_image(img)
print(f"Image embedding shape: {img_emb.shape}")
print(f"Image embedding norm: {np.linalg.norm(img_emb):.4f}  (should be ~1.0)")

print()
print("Embedding text queries with CLIP...")
queries = [
    "a person at a workstation with a computer",
    "a cat sleeping on a couch",
    "a bowl of pasta",
]
text_embs = embed_clip_text(queries)
print(f"Text embeddings shape: {text_embs.shape}")

print()
print("Cosine similarity (image vs each text):")
for q, t_emb in zip(queries, text_embs):
    sim = float(np.dot(img_emb, t_emb))  # both unit-norm -> dot == cosine sim
    print(f"  sim={sim:+.4f}  '{q}'")

print()
print("Higher similarity = better match. The workstation text should win.")
