"""
Audio ingestion pipeline using OpenAI Whisper (local, base model).

Flow:
    audio file → Whisper transcribes → text → embed_text() → ChromaDB

The transcript itself is stored as the document text, with metadata
indicating the original audio source. This means audio becomes searchable
by any text query, and (later) by transcribed spoken queries.

Whisper supports many audio/video container formats via ffmpeg, including
mp3, wav, m4a, mp4, flac, ogg, webm.
"""

from pathlib import Path
from typing import List
import hashlib

import whisper

from src.embeddings import embed_text
from src.vectorstore import add_documents
from src.ingest import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP


# Whisper model size. "base" is ~140 MB, runs on CPU, good accuracy/speed tradeoff.
# Other options: tiny (39 MB, faster, less accurate), small/medium/large (slower).
WHISPER_MODEL = "tiny"

# Supported extensions (Whisper/ffmpeg can handle all of these)
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".webm", ".aac"}

# Lazy-loaded global Whisper model
_whisper_model = None


def get_whisper_model():
    """Lazy-load the Whisper model on first use."""
    global _whisper_model
    if _whisper_model is None:
        print(f"[audio] Loading Whisper model: {WHISPER_MODEL}")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print(f"[audio] Whisper loaded.")
    return _whisper_model


def transcribe_audio(path: str | Path) -> str:
    """
    Transcribe an audio file to text using Whisper.

    Returns the full transcript as a single string.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported: {sorted(SUPPORTED_AUDIO_EXTS)}"
        )

    import subprocess, sys
    print(f"[audio] Transcribing: {path.name} (this may take 10-60s on CPU)")
    res = subprocess.run([sys.executable, "whisper_transcribe.py", str(path)], capture_output=True, text=True)
    transcript = res.stdout.strip()
    print(f"[audio] Transcript ({len(transcript)} chars): {transcript[:120]}{'...' if len(transcript) > 120 else ''}")
    return transcript


def _make_id(source: str, chunk_idx: int) -> str:
    """Stable unique ID based on source path + chunk index."""
    h = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    return f"audio_{h}_{chunk_idx}"


def ingest_audio_file(path: str | Path) -> int:
    """
    Ingest a single audio file: transcribe, chunk, embed, store.

    Returns:
        Number of chunks added.
    """
    path = Path(path)
    transcript = transcribe_audio(path)
    if not transcript:
        print(f"[audio] Empty transcript for {path.name}, skipping.")
        return 0

    chunks = chunk_text(transcript, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    ids = [_make_id(str(path), i) for i in range(len(chunks))]
    embeddings = [embed_text(c).tolist() for c in chunks]
    metadatas = [
        {
            "modality": "audio",
            "source": path.name,
            "source_path": str(path),
            "chunk_index": i,
            "n_chunks": len(chunks),
            "transcript_full_length": len(transcript),
        }
        for i in range(len(chunks))
    ]

    add_documents(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"[audio] {path.name}: {len(chunks)} chunk(s) added")
    return len(chunks)


def ingest_audio_folder(folder: str | Path) -> int:
    """
    Ingest all supported audio files in a folder (non-recursive).

    Returns:
        Total number of chunks added.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    files = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ]
    if not files:
        print(f"[audio] No supported audio files found in {folder}")
        return 0

    print(f"[audio] Found {len(files)} audio file(s) in {folder}")
    total = 0
    for f in files:
        total += ingest_audio_file(f)
    print(f"[audio] Done. Total chunks: {total}")
    return total
