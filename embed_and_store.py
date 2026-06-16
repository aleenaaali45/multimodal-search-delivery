import sys
import json
from src.embeddings import embed_text
from src.vectorstore import add_documents

data = json.loads(sys.argv[1])
chunks = data["chunks"]
metadatas = data["metadatas"]
ids = data["ids"]
embeddings = [embed_text(c).tolist() for c in chunks]
add_documents(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
print("OK")