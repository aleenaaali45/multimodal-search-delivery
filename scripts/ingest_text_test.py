"""Ingest the data/text folder and confirm count."""
from src.vectorstore import reset, count
from src.ingest import ingest_text_folder

reset()
ingest_text_folder("data/text")
print()
print("Final doc count in ChromaDB:", count())
