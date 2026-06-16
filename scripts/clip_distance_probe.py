"""One-off: print raw CLIP distances for a known-relevant query."""
from src.clip_embed import embed_clip_text
from src.vectorstore import query_visual

q = "a person at a desk with a laptop"
clip_emb = embed_clip_text(q).tolist()
results = query_visual(clip_emb, n_results=5)
print("CLIP distances for query:", q)
for i in range(5):
    md = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    print(f"  rank {i+1}: dist={dist:.4f}  source={md['source']} @ t={md['timestamp_sec']:.1f}s")
