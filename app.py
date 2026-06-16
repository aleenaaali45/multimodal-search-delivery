"""
Streamlit demo UI for the Multi-Modal AI Embedding System.

Run with:
    streamlit run app.py

Requires the FastAPI backend running on http://localhost:8000:
    uvicorn src.api:app --reload
"""

from typing import Optional

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
GITHUB_URL = "https://github.com/ZarmanSattar/multimodal-search"

# --- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Modal Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS -------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #6d28d9 50%, #be185d 100%);
            padding: 1.6rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.4rem;
        }
        .main-header h1 {
            color: white;
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .main-header p {
            color: rgba(255,255,255,0.85);
            margin: 0.4rem 0 0 0;
            font-size: 0.92rem;
        }
        .result-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }
        .footer {
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: #6b7280;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.06);
            margin-top: 3rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Presentation -----------------------------------------------------------
MODALITY_ICONS = {"text": "📄", "audio": "🎵", "video": "🎬"}
MODALITY_COLORS = {"text": "#3b82f6", "audio": "#10b981", "video": "#f59e0b"}

EXAMPLE_QUERIES = [
    ("📄 Text", "how do I make spaghetti"),
    ("🎵 Audio", "greenhouse gases and fossil fuels"),
    ("🎬 Video", "a person at a desk with a laptop"),
    ("🧠 AI history", "early artificial intelligence research"),
]


def modality_badge(modality: str, sub_type: Optional[str] = None) -> str:
    icon = MODALITY_ICONS.get(modality, "•")
    color = MODALITY_COLORS.get(modality, "#6b7280")
    label = modality.upper()
    if sub_type:
        label += f" · {sub_type}"
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-size:0.78rem;font-weight:600;">'
        f"{icon} {label}</span>"
    )


# --- Backend calls ----------------------------------------------------------
def get_stats() -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def search_text(query: str, k: int, modality: Optional[str], use_visual: bool) -> dict:
    body = {
        "query": query,
        "k": k,
        "modality": modality,
        "use_visual": use_visual,
        "debug": False,
    }
    r = requests.post(f"{API_BASE}/search", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def search_audio(audio_file, k: int, modality: Optional[str], use_visual: bool) -> dict:
    files = {
        "audio": (
            audio_file.name,
            audio_file.getvalue(),
            audio_file.type or "application/octet-stream",
        )
    }
    data = {
        "k": str(k),
        "use_visual": str(use_visual).lower(),
        "debug": "false",
    }
    if modality:
        data["modality"] = modality
    r = requests.post(f"{API_BASE}/search/audio", files=files, data=data, timeout=180)
    r.raise_for_status()
    return r.json()


def ingest_file(endpoint: str, file) -> dict:
    files = {
        "file": (
            file.name,
            file.getvalue(),
            file.type or "application/octet-stream",
        )
    }
    r = requests.post(f"{API_BASE}/{endpoint}", files=files, timeout=600)
    r.raise_for_status()
    return r.json()


# --- Result rendering -------------------------------------------------------
def render_results(results: list) -> None:
    if not results:
        st.info("No results.")
        return
    for r in results:
        with st.container(border=True):
            cols = st.columns([0.7, 5, 1.4])
            with cols[0]:
                st.markdown(f"### #{r['rank']}")
            with cols[1]:
                st.markdown(
                    modality_badge(r["modality"], r.get("type")),
                    unsafe_allow_html=True,
                )
                st.markdown(f"**📁 `{r['source']}`**")
                if r.get("timestamp_sec") is not None:
                    st.caption(f"⏱ Timestamp: {r['timestamp_sec']}s")
            with cols[2]:
                st.metric("RRF score", f"{r['rrf_score']:.4f}")
            doc = r.get("document", "")
            if doc:
                preview = doc if len(doc) <= 500 else doc[:500] + "..."
                st.markdown(
                    f"<div style='color:#cbd5e1;padding-top:0.4rem;font-size:0.92rem;line-height:1.5;'>"
                    f"{preview}</div>",
                    unsafe_allow_html=True,
                )


# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔎 Multi-Modal Search")
    st.caption("Cross-modal semantic search over text, audio, and video.")
    st.divider()

    st.markdown("#### Collection stats")
    stats = get_stats()
    if stats is None:
        st.error(
            "Backend offline.\n\nStart it with:\n\n`uvicorn src.api:app --reload`"
        )
    else:
        c1, c2 = st.columns(2)
        c1.metric("Main", stats.get("main_collection", "?"))
        c2.metric("Visual", stats.get("visual_collection", "?"))
        st.caption(
            "**Main** (384d MiniLM): text + audio transcripts + video transcripts + BLIP captions.  "
            "**Visual** (512d CLIP): video frame embeddings."
        )
    if st.button("↻ Refresh stats", use_container_width=True):
        st.rerun()

    st.divider()

    with st.expander("🏗 Architecture", expanded=False):
        st.markdown(
            """
**Pipeline:**
Text  ──► MiniLM (384d) ──┐
Audio ──► Whisper ──► MiniLM ──┤
Video ──► Whisper transcript ──┤──► Main collection
└─► BLIP captions ───────┘
└─► CLIP frames (512d) ──► Visual collection

**Search:**
Query
├─► MiniLM ──► Main collection ──┐
└─► CLIP text ──► Visual ──┐     │
│     │
distance>0.78 dropped
│     │
└──► RRF (k=60) ──► ranked results

**Models:** MiniLM-L6-v2 · Whisper-base · CLIP ViT-B-32 · BLIP-base
**Store:** ChromaDB persistent · 2 collections
**Backend:** FastAPI · **UI:** Streamlit
"""
        )

    st.divider()
    st.caption(f"[💻 View on GitHub]({GITHUB_URL})")


# --- Header -----------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🔎 Multi-Modal AI Embedding System</h1>
        <p>Cross-modal semantic search · text · audio · video · RRF fusion across MiniLM + CLIP collections</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_text, tab_audio, tab_ingest = st.tabs(
    ["🔍 Search by text", "🎤 Search by audio", "📥 Ingest files"]
)

# --- Tab 1: Text query ------------------------------------------------------
with tab_text:
    st.markdown("#### Search by text")
    st.caption(
        "Natural-language query. Searches text, audio transcripts, video transcripts, "
        "BLIP captions, and CLIP frame embeddings; results fused via RRF."
    )

    query = st.text_input(
        "Query",
        placeholder="e.g. how do I make spaghetti",
        key="text_query",
    )

    st.caption("Or try an example:")
    ex_cols = st.columns(len(EXAMPLE_QUERIES))
    example_clicked: Optional[str] = None
    for i, (label, q) in enumerate(EXAMPLE_QUERIES):
        if ex_cols[i].button(label, key=f"ex_{i}", use_container_width=True, help=q):
            example_clicked = q
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        k = st.slider("Top-k", 1, 20, 5, key="text_k")
    with c2:
        modality_choice = st.selectbox(
            "Filter by modality",
            ["(all)", "text", "audio", "video"],
            key="text_modality",
        )
    with c3:
        use_visual = st.checkbox(
            "Include visual search (CLIP)", value=True, key="text_visual"
        )
    search_clicked = st.button("🔍 Search", type="primary", key="text_search")

    effective_query = example_clicked if example_clicked else (query if search_clicked else None)

    if effective_query is not None:
        if not effective_query.strip():
            st.warning("Enter a query.")
        else:
            modality = None if modality_choice == "(all)" else modality_choice
            try:
                with st.spinner(f"Searching for: {effective_query}"):
                    resp = search_text(effective_query, k, modality, use_visual)
                st.success(f"{resp['n_results']} result(s) for: \"{effective_query}\"")
                render_results(resp["results"])
            except Exception as e:
                st.error(f"Search failed: {e}")

# --- Tab 2: Audio query -----------------------------------------------------
with tab_audio:
    st.markdown("#### Search by audio")
    st.caption(
        "Upload a short audio file. Whisper transcribes it, then the transcript is "
        "used as a text query."
    )
    audio_file = st.file_uploader(
        "Audio file",
        type=["mp3", "mp4", "m4a", "wav", "ogg", "flac"],
        key="audio_upload",
    )
    if audio_file is not None:
        st.audio(audio_file)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        k_a = st.slider("Top-k", 1, 20, 5, key="audio_k")
    with c2:
        modality_choice_a = st.selectbox(
            "Filter by modality",
            ["(all)", "text", "audio", "video"],
            key="audio_modality",
        )
    with c3:
        use_visual_a = st.checkbox(
            "Include visual search (CLIP)", value=True, key="audio_visual"
        )
    if st.button("🎤 Transcribe & search", type="primary", key="audio_search"):
        if audio_file is None:
            st.warning("Upload an audio file first.")
        else:
            modality = None if modality_choice_a == "(all)" else modality_choice_a
            try:
                with st.spinner("Transcribing with Whisper, then searching..."):
                    resp = search_audio(audio_file, k_a, modality, use_visual_a)
                st.info(f"📝 **Transcript:** _{resp['transcript']}_")
                search_resp = resp["search"]
                st.success(f"{search_resp['n_results']} result(s)")
                render_results(search_resp["results"])
            except Exception as e:
                st.error(f"Audio search failed: {e}")

# --- Tab 3: Ingest ----------------------------------------------------------
with tab_ingest:
    st.markdown("#### Ingest new files")
    st.caption(
        "Add content to the searchable collections. Uploads land in "
        "data/{text,audio,video} and are embedded into ChromaDB."
    )
    st.warning(
        "Each upload adds new documents — uploading the same file twice duplicates "
        "entries.",
        icon="⚠️",
    )
    ing_text, ing_audio, ing_video = st.columns(3)
    with ing_text:
        st.markdown("**📄 Text (.txt)**")
        f_text = st.file_uploader("text", type=["txt"], key="ing_text_up", label_visibility="collapsed")
        if st.button("Ingest text", key="ing_text_btn", disabled=(f_text is None), use_container_width=True):
            try:
                with st.spinner("Ingesting..."):
                    out = ingest_file("ingest/text", f_text)
                st.success(
                    f"✓ Main: {out['counts']['main_collection']} · "
                    f"Visual: {out['counts']['visual_collection']}"
                )
            except Exception as e:
                st.error(f"Failed: {e}")
    with ing_audio:
        st.markdown("**🎵 Audio**")
        f_audio = st.file_uploader(
            "audio",
            type=["mp3", "mp4", "m4a", "wav", "ogg", "flac"],
            key="ing_audio_up",
            label_visibility="collapsed",
        )
        if st.button("Ingest audio", key="ing_audio_btn", disabled=(f_audio is None), use_container_width=True):
            try:
                with st.spinner("Transcribing & embedding..."):
                    out = ingest_file("ingest/audio", f_audio)
                st.success(
                    f"✓ Main: {out['counts']['main_collection']} · "
                    f"Visual: {out['counts']['visual_collection']}"
                )
            except Exception as e:
                st.error(f"Failed: {e}")
    with ing_video:
        st.markdown("**🎬 Video**")
        f_video = st.file_uploader(
            "video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="ing_video_up",
            label_visibility="collapsed",
        )
        if st.button("Ingest video", key="ing_video_btn", disabled=(f_video is None), use_container_width=True):
            try:
                with st.spinner("Extracting frames + transcript + captions (slow)..."):
                    out = ingest_file("ingest/video", f_video)
                st.success(
                    f"✓ Main: {out['counts']['main_collection']} · "
                    f"Visual: {out['counts']['visual_collection']}"
                )
            except Exception as e:
                st.error(f"Failed: {e}")

# --- Footer -----------------------------------------------------------------
st.markdown(
    f"""
    <div class="footer">
        Multi-Modal AI Embedding System · SPS Internship Project ·
        <a href="{GITHUB_URL}" style="color:#9ca3af;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
