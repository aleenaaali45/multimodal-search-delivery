"""Test audio ingestion. Does NOT reset ChromaDB — adds to existing text docs."""
from src.vectorstore import count
from src.audio_ingest import ingest_audio_folder

print("Doc count BEFORE audio ingest:", count())
ingest_audio_folder("data/audio")
print()
print("Doc count AFTER audio ingest:", count())
