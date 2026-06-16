"""Round-trip sanity test: embed 3 docs, store, query, verify ranking."""
from src.embeddings import embed_text
from src.vectorstore import add_documents, query, count, reset

reset()

docs = [
    "The cat sat on the warm windowsill in the afternoon sun.",
    "Stock markets fell sharply after the central bank raised interest rates.",
    "Pasta carbonara is made with eggs, cheese, pancetta, and black pepper.",
]
ids = ["doc1", "doc2", "doc3"]
embs = [embed_text(d).tolist() for d in docs]
metas = [{"modality": "text", "source": f"sample_{i}.txt"} for i in range(3)]

add_documents(ids, embs, docs, metas)
print("Total docs in store:", count())

q = "recipe with eggs and bacon"
q_emb = embed_text(q).tolist()
results = query(q_emb, n_results=3)

print()
print("Query:", q)
print("Top matches (closest first):")
for i in range(3):
    print(f"  {i+1}. id={results['ids'][0][i]} dist={results['distances'][0][i]:.4f}")
    print(f"     {results['documents'][0][i]}")
