"""
Command-line search interface for the multi-modal vector store.

Solution B architecture: queries BOTH ChromaDB collections and fuses results
using Reciprocal Rank Fusion (RRF).

  Main collection (384-dim, MiniLM):
    - text chunks
    - audio transcripts (modality=audio)
    - video audio transcripts (modality=video, type=transcript)
    - video BLIP captions (modality=video, type=caption)

  Visual collection (512-dim, CLIP):
    - video CLIP frame embeddings (modality=video, type=frame)

For each query:
  1. Embed query with MiniLM -> search main collection
  2. Embed query with CLIP text encoder -> search visual collection
  3. RRF-merge the two ranked lists into one final ranking

This means a query like "person at a computer" can be retrieved via THREE
independent paths:
  - matching a BLIP caption ("...sitting at a desk with a laptop")
  - matching a Whisper transcript ("This is my workstation...")
  - matching the raw CLIP visual signal of the frame itself

Usage:
    python -m scripts.search "your natural language query"
    python -m scripts.search "your query" --k 10
    python -m scripts.search "your query" --modality video
    python -m scripts.search "your query" --no-visual    # skip CLIP path
    python -m scripts.search "your query" --debug        # show RRF score breakdown
"""

import argparse
import sys
from typing import Dict, List, Any, Tuple

from src.embeddings import embed_text
from src.clip_embed import embed_clip_text
from src.vectorstore import query, query_visual, count, visual_count


# RRF constant. 60 is the standard value from the original RRF paper
# (Cormack et al. 2009). Higher k = flatter rank weighting.
RRF_K = 60

# Maximum acceptable CLIP cosine distance for a visual result to enter RRF
# fusion. Calibrated from observed distances on this dataset:
#   - relevant matches:   0.74-0.78 (rank 1-3 of the workstation video for
#                         the "person at desk" query)
#   - presumed irrelevant: 0.78+ (rank 4-5 of that same video, plus the
#                         0.65+ band we previously saw for unrelated queries)
# 0.78 sits at the conservative edge of the observed relevant band, dropping
# borderline matches that are more likely to be noise than signal. For
# production, a relative-to-best-score gate or a learned reranker would be
# more robust than a static threshold.
CLIP_DISTANCE_THRESHOLD = 0.78


def rrf_merge(
    rankings: List[List[Dict[str, Any]]],
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion merge over multiple ranked lists.

    Each ranking is a list of result dicts (already in rank order, rank 1 first).
    Results are merged by their 'id' field. RRF score for a doc:
        sum over each list of 1 / (k + rank_in_that_list)

    Returns:
        Single merged list sorted by RRF score descending. Each dict gets
        added fields: 'rrf_score', 'rrf_sources' (which lists contributed),
        and 'per_list_ranks' for debugging.
    """
    # doc_id -> aggregated record
    merged: Dict[str, Dict[str, Any]] = {}

    for list_idx, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            doc_id = item["id"]
            contribution = 1.0 / (k + rank)
            if doc_id not in merged:
                # First time we see this doc -> copy its fields
                merged[doc_id] = {**item, "rrf_score": 0.0, "rrf_sources": [], "per_list_ranks": {}}
            merged[doc_id]["rrf_score"] += contribution
            merged[doc_id]["rrf_sources"].append(list_idx)
            merged[doc_id]["per_list_ranks"][list_idx] = rank

    out = list(merged.values())
    out.sort(key=lambda x: x["rrf_score"], reverse=True)
    return out


def chroma_to_items(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten ChromaDB query() output (which uses nested lists for batched
    queries) into a clean list-of-dicts for a single query.
    """
    if not results.get("ids") or not results["ids"][0]:
        return []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [
        {"id": i, "document": d, "metadata": m, "distance": dist}
        for i, d, m, dist in zip(ids, docs, metas, dists)
    ]


def search(
    query_text: str,
    k: int = 5,
    modality: str | None = None,
    use_visual: bool = True,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    End-to-end multi-modal search with RRF fusion.

    Args:
        query_text: Natural-language query.
        k: How many final results to return.
        modality: Optional filter ('text', 'audio', 'video'). Applies to main
                  collection; visual collection is always 'video' so it's only
                  included when modality is None or 'video'.
        use_visual: If False, skip the CLIP path entirely. Useful for ablation
                    or when you only want text/audio results.
        debug: If True, attach RRF score breakdown to each result.

    Returns:
        List of result dicts ordered by RRF score (best first).
    """
    # Pull more results per source than the user asked for, so the fusion
    # has room to rerank meaningfully. 2*k per source is a sane default.
    per_source_k = max(k * 2, 10)

    rankings: List[List[Dict[str, Any]]] = []

    # ─── Main collection (MiniLM 384-dim) ───────────────────────────
    where_main = {"modality": modality} if modality else None
    main_emb = embed_text(query_text).tolist()
    main_raw = query(main_emb, n_results=per_source_k, where=where_main)
    main_items = chroma_to_items(main_raw)
    rankings.append(main_items)

    # ─── Visual collection (CLIP 512-dim) ───────────────────────────
    include_visual = use_visual and visual_count() > 0 and modality in (None, "video")
    if include_visual:
        clip_emb = embed_clip_text(query_text).tolist()
        visual_raw = query_visual(clip_emb, n_results=per_source_k)
        visual_items = chroma_to_items(visual_raw)
        # Drop weakly-matched visual results so they don't pollute RRF
        n_raw = len(visual_items)
        visual_items = [v for v in visual_items if v["distance"] <= CLIP_DISTANCE_THRESHOLD]
        n_dropped = n_raw - len(visual_items)
        if debug:
            print(f"[debug] visual: {n_raw} candidates, dropped {n_dropped} above dist {CLIP_DISTANCE_THRESHOLD}, kept {len(visual_items)}")
        if visual_items:
            rankings.append(visual_items)

    # ─── Fuse ────────────────────────────────────────────────────────
    fused = rrf_merge(rankings)
    return fused[:k]


def format_result(idx: int, r: Dict[str, Any], debug: bool = False) -> str:
    """Pretty-print one result."""
    meta = r.get("metadata", {}) or {}
    modality = meta.get("modality", "?")
    rtype = meta.get("type", "")
    source = meta.get("source", "?")
    timestamp = meta.get("timestamp_sec")

    # Tag combines modality + sub-type for clarity
    if rtype:
        tag = f"{modality}/{rtype}"
    else:
        tag = modality

    snippet = (r.get("document") or "").strip().replace("\n", " ")
    if len(snippet) > 200:
        snippet = snippet[:200] + "..."

    ts_str = f" @ t={timestamp:.1f}s" if timestamp is not None else ""
    rrf = r.get("rrf_score", 0.0)
    head = f"#{idx}  [{tag}] {source}{ts_str}  (rrf: {rrf:.4f})"

    lines = [head, f"    {snippet}"]
    if debug:
        sources = r.get("rrf_sources", [])
        ranks = r.get("per_list_ranks", {})
        src_names = {0: "main", 1: "visual"}
        details = ", ".join(f"{src_names.get(s,s)}#{ranks[s]}" for s in sources)
        lines.append(f"    [debug] contributed by: {details}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search the multi-modal vector store (RRF over MiniLM main + CLIP visual).",
    )
    parser.add_argument("query", help="Natural language query string.")
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

    total_main = count()
    total_visual = visual_count()
    if total_main == 0 and total_visual == 0:
        print("[search] Both collections are empty. Ingest some files first.")
        return 1

    print(f"[search] Collections: main={total_main}, visual={total_visual}")
    print(f"[search] Query: {args.query!r}")
    if args.modality:
        print(f"[search] Modality filter: {args.modality}")
    if args.no_visual:
        print(f"[search] Visual path: DISABLED")
    print()

    results = search(
        args.query,
        k=args.k,
        modality=args.modality,
        use_visual=not args.no_visual,
        debug=args.debug,
    )

    if not results:
        print("[search] No matches.")
        return 0

    for i, r in enumerate(results, start=1):
        print(format_result(i, r, debug=args.debug))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
