"""
Video ingestion pipeline.

A video produces three searchable signals:
    1. Audio transcript (Whisper)           → main collection [type=transcript]
    2. CLIP frame embeddings (512-dim)      → visual collection [type=frame]
    3. BLIP frame captions (text → 384-dim) → main collection [type=caption]

This file handles the video-mechanics layer:
    - Sampling frames at fixed time intervals (default every 2 seconds)
    - Extracting the audio track to a temp .wav for Whisper

CLIP and BLIP embedding logic lives in separate modules (clip_embed.py,
blip_caption.py) added in subsequent steps. The end-to-end ingest function
that ties everything together is ingest_video_file() at the bottom — but
the CLIP/BLIP parts are stubbed out for now and will be wired in next.
"""

from pathlib import Path
from typing import List, Tuple
import hashlib
import subprocess
import tempfile

import cv2
from PIL import Image

from src.embeddings import embed_text
from src.clip_embed import embed_image_batch, CLIP_EMBEDDING_DIM
from src.blip_caption import caption_image_batch
from src.vectorstore import add_documents, add_visual_documents


# Sample one frame every N seconds. 2.0 is a reasonable tradeoff between
# coverage and processing time on CPU. For a 15s video this gives ~7-8 frames.
FRAME_INTERVAL_SEC = 2.0

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def sample_frames(video_path: str | Path, interval_sec: float = FRAME_INTERVAL_SEC) -> List[Tuple[float, Image.Image]]:
    """
    Sample one frame every `interval_sec` seconds from a video.

    Returns:
        List of (timestamp_in_seconds, PIL.Image) tuples.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video metadata: fps={fps}, frames={total_frames}")

    print(f"[video] {video_path.name}: {duration_sec:.1f}s, {fps:.1f} fps, {total_frames} frames")

    frame_step = int(fps * interval_sec)
    if frame_step < 1:
        frame_step = 1

    sampled: List[Tuple[float, Image.Image]] = []
    for frame_idx in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        # OpenCV returns BGR; PIL expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        timestamp = frame_idx / fps
        sampled.append((timestamp, img))

    cap.release()
    print(f"[video] Sampled {len(sampled)} frame(s) at {interval_sec}s intervals")
    return sampled


def extract_audio(video_path: str | Path, output_wav: str | Path | None = None) -> Path:
    """
    Extract the audio track from a video to a 16kHz mono WAV file using ffmpeg.

    16kHz mono is what Whisper expects internally — pre-converting saves time
    during transcription.

    Returns:
        Path to the extracted .wav file.
    """
    video_path = Path(video_path)
    if output_wav is None:
        # Use a temp file that survives until the caller is done with it.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        output_wav = Path(tmp.name)
    else:
        output_wav = Path(output_wav)

    cmd = [
        "ffmpeg",
        "-y",                # overwrite output
        "-i", str(video_path),
        "-vn",               # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",      # 16 kHz sample rate (Whisper's expected input)
        "-ac", "1",          # mono
        str(output_wav),
    ]
    print(f"[video] Extracting audio: {video_path.name} → {output_wav.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    print(f"[video] Audio extracted ({output_wav.stat().st_size} bytes)")
    return output_wav


def _make_id(prefix: str, source: str, idx: int) -> str:
    """Stable unique ID for a video-derived document."""
    h = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}_{idx}"


def ingest_video_file(path: str | Path) -> dict:
    """
    Full end-to-end ingestion for one video file.

    Produces three independent signals stored across two ChromaDB collections:

      Main 384-dim collection (text embeddings via MiniLM):
        1. Whisper transcript of the audio track       [type=transcript]
        2. BLIP captions of sampled frames             [type=caption]

      Visual 512-dim collection (CLIP image embeddings):
        3. CLIP frame embeddings                       [type=frame]

    All entries are tagged with modality='video' and the source filename.

    Returns:
        Dict with counts: {'transcript_chunks': N, 'frames': M, 'captions': M}
    """
    # Import here to avoid loading Whisper at module-import time
    from src.audio_ingest import transcribe_audio
    from src.ingest import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
        raise ValueError(
            f"Unsupported video format: {path.suffix}. "
            f"Supported: {sorted(SUPPORTED_VIDEO_EXTS)}"
        )

    print(f"\n{'='*60}\n[video] INGESTING: {path.name}\n{'='*60}")

    # ─── PATH A: Audio transcript ────────────────────────────────────────
    print(f"\n[video] Path A: Audio transcript")
    audio_wav = extract_audio(path)
    transcript = transcribe_audio(audio_wav)

    transcript_chunks_added = 0
    if transcript:
        chunks = chunk_text(transcript, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        ids = [_make_id("vidtrans", str(path), i) for i in range(len(chunks))]
        embs = [embed_text(c).tolist() for c in chunks]
        metas = [
            {
                "modality": "video",
                "type": "transcript",
                "source": path.name,
                "source_path": str(path),
                "chunk_index": i,
                "n_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]
        add_documents(ids=ids, embeddings=embs, documents=chunks, metadatas=metas)
        transcript_chunks_added = len(chunks)
        print(f"[video] +{transcript_chunks_added} transcript chunk(s) -> main collection")
    else:
        print(f"[video] No transcript (silent or empty audio)")

    # Clean up temp audio file
    try:
        audio_wav.unlink()
    except Exception:
        pass

    # ─── PATH B + C: Frame sampling, then CLIP + BLIP ────────────────────
    print(f"\n[video] Paths B+C: Frame sampling for CLIP + BLIP")
    frames = sample_frames(path)
    if not frames:
        print(f"[video] No frames sampled, skipping visual paths")
        return {
            "transcript_chunks": transcript_chunks_added,
            "frames": 0,
            "captions": 0,
        }

    timestamps = [ts for ts, _ in frames]
    images = [img for _, img in frames]

    # PATH B: CLIP frame embeddings -> visual collection (512-dim)
    print(f"\n[video] Path B: CLIP embeddings for {len(images)} frame(s)")
    clip_embs = embed_image_batch(images)  # (N, 512) numpy array
    frame_ids = [_make_id("vidframe", str(path), i) for i in range(len(images))]
    frame_docs = [
        f"Frame at t={ts:.1f}s of {path.name}" for ts in timestamps
    ]
    frame_metas = [
        {
            "modality": "video",
            "type": "frame",
            "source": path.name,
            "source_path": str(path),
            "timestamp_sec": float(ts),
            "frame_index": i,
            "n_frames": len(images),
        }
        for i, ts in enumerate(timestamps)
    ]
    add_visual_documents(
        ids=frame_ids,
        embeddings=clip_embs.tolist(),
        documents=frame_docs,
        metadatas=frame_metas,
    )
    print(f"[video] +{len(images)} CLIP embedding(s) -> visual collection")

    # PATH C: BLIP captions -> embed with MiniLM -> main collection (384-dim)
    print(f"\n[video] Path C: BLIP captions for {len(images)} frame(s)")
    captions = caption_image_batch(images, prompt="a photo of")
    for ts, cap in zip(timestamps, captions):
        print(f"  t={ts:5.2f}s  {cap!r}")

    caption_ids = [_make_id("vidcap", str(path), i) for i in range(len(captions))]
    caption_embs = [embed_text(c).tolist() for c in captions]
    caption_metas = [
        {
            "modality": "video",
            "type": "caption",
            "source": path.name,
            "source_path": str(path),
            "timestamp_sec": float(ts),
            "frame_index": i,
            "n_frames": len(captions),
        }
        for i, ts in enumerate(timestamps)
    ]
    add_documents(
        ids=caption_ids,
        embeddings=caption_embs,
        documents=captions,
        metadatas=caption_metas,
    )
    print(f"[video] +{len(captions)} caption(s) -> main collection")

    summary = {
        "transcript_chunks": transcript_chunks_added,
        "frames": len(images),
        "captions": len(captions),
    }
    print(f"\n[video] DONE: {path.name}  {summary}")
    return summary


def ingest_video_folder(folder: str | Path) -> dict:
    """
    Ingest all supported video files in a folder (non-recursive).

    Returns:
        Aggregate counts: {'transcript_chunks': total, 'frames': total, 'captions': total}
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    files = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS
    ]
    if not files:
        print(f"[video] No supported video files found in {folder}")
        return {"transcript_chunks": 0, "frames": 0, "captions": 0}

    print(f"[video] Found {len(files)} video file(s) in {folder}")
    totals = {"transcript_chunks": 0, "frames": 0, "captions": 0}
    for f in files:
        s = ingest_video_file(f)
        for k in totals:
            totals[k] += s[k]
    print(f"\n[video] ALL DONE. Totals: {totals}")
    return totals
