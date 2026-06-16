"""
Audio → Text search.

Takes a path to a short audio file containing a spoken query, transcribes
it with Whisper, then runs the resulting text through the standard
multi-modal search pipeline (MiniLM main collection + CLIP visual collection
with RRF fusion).

This satisfies the brief's Audio→Text example:
    [User speaks a phrase] → most relevant document containing the phrase

The query audio is NOT ingested into ChromaDB — it's transcribed once and
discarded. To index audio as searchable content, use ingest_audio_test.py
instead.

Usage:
    python -m scripts.search_by_audio path\\to\\spoken_query.mp4
    python -m scripts.search_by_audio query.mp3 --k 10
    python -m scripts.search_by_audio query.wav --modality text
    python -m scripts.search_by_audio query.mp4 --no-visual --debug
"""

import argparse
import sys
from pathlib import Path

from src.audio_ingest import transcribe_audio, SUPPORTED_AUDIO_EXTS
from src.vectorstore import count, visual_count

# Reuse all the search infrastructure we already built
from scripts.search import search, format_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search the multi-modal vector store using a spoken audio query.",
    )
    parser.add_argument(
        "audio_path",
        help="Path to a short audio file containing the spoken query.",
    )
    parser.add_argument("--k", type=int, default=5, help="Final results to return.")
    parser.add_argument(
        "--modality", choices=["text", "audio", "video"], default=None,
        help="Restrict results to a single modality.",
    )
    parser.add_argument(
        "--no-visual", action="store_true",
        help="Skip the CLIP visual collection (main only).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show RRF score breakdown for each result.",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"[search-audio] File not found: {audio_path}")
        return 1
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        print(
            f"[search-audio] Unsupported audio format: {audio_path.suffix}. "
            f"Supported: {sorted(SUPPORTED_AUDIO_EXTS)}"
        )
        return 1

    total_main = count()
    total_visual = visual_count()
    if total_main == 0 and total_visual == 0:
        print("[search-audio] Both collections are empty. Ingest some files first.")
        return 1

    # ─── Step 1: Transcribe the spoken query ─────────────────────────
    print(f"[search-audio] Audio file: {audio_path}")
    print(f"[search-audio] Transcribing query...")
    transcript = transcribe_audio(audio_path)

    if not transcript:
        print("[search-audio] Whisper returned an empty transcript. Cannot search.")
        return 1

    print()
    print(f"[search-audio] You said: {transcript!r}")
    print(f"[search-audio] Collections: main={total_main}, visual={total_visual}")
    if args.modality:
        print(f"[search-audio] Modality filter: {args.modality}")
    if args.no_visual:
        print(f"[search-audio] Visual path: DISABLED")
    print()

    # ─── Step 2: Run the standard text search ────────────────────────
    results = search(
        transcript,
        k=args.k,
        modality=args.modality,
        use_visual=not args.no_visual,
        debug=args.debug,
    )

    if not results:
        print("[search-audio] No matches.")
        return 0

    for i, r in enumerate(results, start=1):
        print(format_result(i, r, debug=args.debug))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
