"""
Ingestion pipeline for text files.

Reads .txt files from disk, splits into chunks if needed, embeds each chunk,
and stores in ChromaDB with metadata indicating source file and modality.

Audio and video ingestion modules will be added later (they convert their
content to text first — transcripts/captions — then call into this same
flow).
"""

from pathlib import Path
from typing import List
import hashlib

from src.embeddings import embed_text
from src.vectorstore import add_documents


# Chunk size in characters. ~500 chars is roughly 100 tokens for English,
# well within MiniLM's 256-token context. Overlap preserves context across
# chunk boundaries.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.

    Simple character-based chunking. Good enough for short documents;
    for production we'd use a sentence-aware splitter.
    """
    text = text.strip()
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _make_id(source: str, chunk_idx: int) -> str:
    """Stable unique ID based on source path + chunk index."""
    h = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    return f"text_{h}_{chunk_idx}"


def ingest_text_file(path: str | Path) -> int:
    """
    Ingest a single .txt file into the vector store.

    Returns:
        Number of chunks added.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected .txt file, got: {path.suffix}")

    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    ids = [_make_id(str(path), i) for i in range(len(chunks))]
    embeddings = [embed_text(c).tolist() for c in chunks]
    metadatas = [
        {
            "modality": "text",
            "source": path.name,
            "source_path": str(path),
            "chunk_index": i,
            "n_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    add_documents(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"[ingest] {path.name}: {len(chunks)} chunk(s) added")
    return len(chunks)


def ingest_text_folder(folder: str | Path) -> int:
    """
    Ingest all .txt files in a folder (non-recursive).

    Returns:
        Total number of chunks added across all files.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    total = 0
    files = sorted(folder.glob("*.txt"))
    if not files:
        print(f"[ingest] No .txt files found in {folder}")
        return 0

    print(f"[ingest] Found {len(files)} .txt file(s) in {folder}")
    for f in files:
        total += ingest_text_file(f)
    print(f"[ingest] Done. Total chunks: {total}")
    return total
