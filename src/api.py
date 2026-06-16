"""
FastAPI HTTP layer wrapping the multi-modal search system.

All endpoints are thin wrappers around functions already implemented in
src/ingest.py, src/audio_ingest.py, src/video_ingest.py, and scripts/search.py.

Endpoints:
    GET  /                 — API info
    GET  /stats            — collection sizes
    POST /search           — text query → results
    POST /search/audio     — multipart audio query → results
    POST /ingest/text      — multipart .txt upload → ingest
    POST /ingest/audio     — multipart audio upload → ingest
    POST /ingest/video     — multipart video upload → ingest

Run with:
    uvicorn src.api:app --reload --port 8000

Interactive docs at:
    http://localhost:8000/docs
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.vectorstore import count, visual_count
from src.ingest import ingest_text_file
from src.audio_ingest import ingest_audio_file, transcribe_audio, SUPPORTED_AUDIO_EXTS
from src.video_ingest import ingest_video_file, SUPPORTED_VIDEO_EXTS

# We reuse the search() function from scripts/search.py (it lives there
# alongside the CLI but is just a regular function).
from scripts.search import search as run_search


# ─── Upload destinations ──────────────────────────────────────────────────────
# Uploaded files land in the same data folders the rest of the system uses,
# so they show up to anyone running ingest_*_folder() later. The system is
# additive — uploading the same file twice will duplicate it in ChromaDB.

DATA_ROOT = Path("data")
TEXT_DIR = DATA_ROOT / "text"
AUDIO_DIR = DATA_ROOT / "audio"
VIDEO_DIR = DATA_ROOT / "video"

# Ensure target dirs exist (idempotent)
for d in (TEXT_DIR, AUDIO_DIR, VIDEO_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Modal AI Embedding System",
    description=(
        "Cross-modal semantic search over text, audio, and video. "
        "Built on ChromaDB + MiniLM + Whisper + CLIP + BLIP, with RRF "
        "fusion across two collections."
    ),
    version="0.1.0",
)

# Permissive CORS for the local Streamlit demo (and any local frontend).
# Tighten this for any non-local deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic request/response models ─────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query.")
    k: int = Field(5, ge=1, le=50, description="Max results to return.")
    modality: Optional[str] = Field(
        None,
        description="Restrict to one modality: 'text', 'audio', or 'video'.",
    )
    use_visual: bool = Field(
        True,
        description="Include the CLIP visual collection in the search.",
    )
    debug: bool = Field(
        False,
        description="Include per-source RRF rank breakdown in each result.",
    )


class SearchResult(BaseModel):
    rank: int
    id: str
    modality: str
    type: Optional[str] = None
    source: str
    timestamp_sec: Optional[float] = None
    document: str
    rrf_score: float
    rrf_sources: Optional[List[int]] = None
    per_list_ranks: Optional[Dict[str, int]] = None


class SearchResponse(BaseModel):
    query: str
    k: int
    n_results: int
    results: List[SearchResult]


class IngestResponse(BaseModel):
    filename: str
    saved_to: str
    counts: Dict[str, int]


class TranscribeAndSearchResponse(BaseModel):
    transcript: str
    search: SearchResponse


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    """Save an UploadFile to dest_dir using its original filename."""
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")
    dest = dest_dir / Path(upload.filename).name  # strip any path traversal
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _results_to_response(query_text: str, k: int, raw_results: List[Dict[str, Any]]) -> SearchResponse:
    """Convert internal search() result dicts into a SearchResponse."""
    items: List[SearchResult] = []
    for i, r in enumerate(raw_results, start=1):
        meta = r.get("metadata") or {}
        items.append(SearchResult(
            rank=i,
            id=r["id"],
            modality=meta.get("modality", "?"),
            type=meta.get("type"),
            source=meta.get("source", "?"),
            timestamp_sec=meta.get("timestamp_sec"),
            document=r.get("document", ""),
            rrf_score=r.get("rrf_score", 0.0),
            rrf_sources=r.get("rrf_sources"),
            per_list_ranks={str(k): v for k, v in (r.get("per_list_ranks") or {}).items()} or None,
        ))
    return SearchResponse(query=query_text, k=k, n_results=len(items), results=items)


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Multi-Modal AI Embedding System",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": [
            "GET /stats",
            "POST /search",
            "POST /search/audio",
            "POST /ingest/text",
            "POST /ingest/audio",
            "POST /ingest/video",
        ],
    }


@app.get("/stats")
def stats():
    return {
        "main_collection": count(),
        "visual_collection": visual_count(),
    }


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    """Run a text query through the standard RRF-fused search pipeline."""
    if req.modality is not None and req.modality not in ("text", "audio", "video"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid modality: {req.modality}. Must be text/audio/video or omitted.",
        )

    raw = run_search(
        req.query,
        k=req.k,
        modality=req.modality,
        use_visual=req.use_visual,
        debug=req.debug,
    )
    return _results_to_response(req.query, req.k, raw)


@app.post("/search/audio", response_model=TranscribeAndSearchResponse)
async def search_audio_endpoint(
    audio: UploadFile = File(..., description="Short audio file with a spoken query."),
    k: int = Form(5),
    modality: Optional[str] = Form(None),
    use_visual: bool = Form(True),
    debug: bool = Form(False),
):
    """Transcribe an uploaded audio query with Whisper, then run text search on it."""
    if Path(audio.filename or "").suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported: {sorted(SUPPORTED_AUDIO_EXTS)}",
        )

    # Save to a temp file (do NOT persist to data/queries from the API by default).
    suffix = Path(audio.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        transcript = transcribe_audio(tmp_path)
        if not transcript:
            raise HTTPException(status_code=422, detail="Whisper returned an empty transcript.")
        raw = run_search(
            transcript,
            k=k,
            modality=modality,
            use_visual=use_visual,
            debug=debug,
        )
        return TranscribeAndSearchResponse(
            transcript=transcript,
            search=_results_to_response(transcript, k, raw),
        )
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported on /ingest/text.")
    dest = _save_upload(file, TEXT_DIR)
    ingest_text_file(dest)
    return IngestResponse(
        filename=file.filename,
        saved_to=str(dest),
        counts={"main_collection": count(), "visual_collection": visual_count()},
    )


@app.post("/ingest/audio", response_model=IngestResponse)
async def ingest_audio_endpoint(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in SUPPORTED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Supported: {sorted(SUPPORTED_AUDIO_EXTS)}",
        )
    dest = _save_upload(file, AUDIO_DIR)
    ingest_audio_file(dest)
    return IngestResponse(
        filename=file.filename,
        saved_to=str(dest),
        counts={"main_collection": count(), "visual_collection": visual_count()},
    )


@app.post("/ingest/video", response_model=IngestResponse)
async def ingest_video_endpoint(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in SUPPORTED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported: {sorted(SUPPORTED_VIDEO_EXTS)}",
        )
    dest = _save_upload(file, VIDEO_DIR)
    ingest_video_file(dest)
    return IngestResponse(
        filename=file.filename,
        saved_to=str(dest),
        counts={"main_collection": count(), "visual_collection": visual_count()},
    )
