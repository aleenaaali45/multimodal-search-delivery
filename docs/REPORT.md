# Multi-Modal AI Embedding System
## Project Report

**Author:** Zarman Sattar
**Project:** SPS Internship Capstone — Task 1
**Repository:** https://github.com/ZarmanSattar/multimodal-search
**Date:** December 2025

---

## 1. Executive Summary

This report describes the design, implementation, and evaluation of a **Multi-Modal AI Embedding System** — a proof-of-concept that ingests text, audio, and video files, transforms them into dense vector embeddings, and enables natural-language semantic search across all modalities.

The system successfully demonstrates all four cross-modal query types required by the project brief: text → text, text → audio, text → video, and audio → text. It is built entirely on free, locally-hosted models with no paid API dependencies, and exposes its functionality through three interfaces: a FastAPI REST backend, a Streamlit web UI, and a command-line interface.

The central engineering contribution is a **dual-collection retrieval architecture** that combines a semantic text embedding model (MiniLM, 384d) with a joint image-text embedding model (CLIP ViT-B-32, 512d), fusing their independent ranked outputs using **Reciprocal Rank Fusion (RRF)** with a calibrated CLIP distance threshold to suppress noise.

---

## 2. Problem Statement

The project brief required a system capable of:

1. Accepting text, audio, and video as input modalities.
2. Generating and persistently storing vector embeddings for each.
3. Enabling cross-modal semantic search — a single natural-language query must return relevant content regardless of the original modality.
4. Supporting at least these example queries:
   - *"Find this particular text."* (Text → Text)
   - *"Find the audio of a song by XYZ."* (Text → Audio)
   - *"Find the video where a human is drinking milk."* (Text → Video)
   - *[User speaks a phrase]* → relevant text document (Audio → Text)

The core difficulty is that text, audio, and video are fundamentally different signal types. A naive approach — embedding everything into a single space — discards modality-specific signal. A more robust approach is needed: one that respects the strengths of different embedding models while still allowing them to be queried jointly.

---

## 3. Architecture

### 3.1 High-Level Design

The system is organised into three layers:

| Layer | Responsibility | Components |
|---|---|---|
| **Ingestion** | Convert raw files into embeddings + metadata | Whisper, CLIP, BLIP, MiniLM, OpenCV, FFmpeg |
| **Storage** | Persist embeddings, enable nearest-neighbour search | ChromaDB (two collections) |
| **Retrieval** | Accept query, run dual search, fuse results | RRF, CLIP distance gate, FastAPI, Streamlit |

### 3.2 Ingestion Pipeline

Each modality follows a different path, but all converge on one or both ChromaDB collections:

- **Text files** are read, chunked at paragraph boundaries, and embedded with `sentence-transformers/all-MiniLM-L6-v2` (384d). Each chunk becomes one document in the main collection.

- **Audio files** are transcribed using OpenAI's `Whisper` (`base` model). The resulting transcript is embedded with MiniLM and stored alongside the original filename, treating the audio as a text-equivalent document.

- **Video files** undergo a three-path ingestion:
  1. The audio track is extracted via FFmpeg, transcribed by Whisper, and embedded with MiniLM (stored as `type: transcript`).
  2. Keyframes are sampled at fixed intervals (every 2 seconds) using OpenCV. Each frame is embedded with CLIP's image encoder (512d) and stored in the **visual collection** with a timestamp.
  3. Each sampled frame is also captioned by BLIP (`Salesforce/blip-image-captioning-base`), and the resulting caption text is embedded with MiniLM (stored as `type: caption`) in the main collection.

This deliberate redundancy means a single video can be retrieved via three independent semantic paths.

### 3.3 Storage Layer

Two ChromaDB collections are maintained:

| Collection | Embedding Dim | Embedding Model | Contents |
|---|---|---|---|
| `multimodal` | 384 | MiniLM | text chunks, audio transcripts, video transcripts, BLIP captions |
| `multimodal_visual` | 512 | CLIP ViT-B-32 | video keyframe image embeddings |

Each document includes metadata fields: `source` (original filename), `modality` (text/audio/video), `type` (transcript/caption/null), and `timestamp_sec` (for video frames).

### 3.4 Retrieval Pipeline

A search query flows through the following pipeline:

1. The query string is embedded **twice** — once by MiniLM (for the main collection) and once by CLIP's text encoder (for the visual collection). CLIP's text and image encoders share an embedding space, enabling direct text-to-image similarity.

2. Both collections are queried independently, returning two ranked lists.

3. The CLIP-derived list is **gated**: any result with cosine distance > 0.78 is dropped. This threshold was calibrated empirically against the test dataset (see §4.3).

4. The two filtered lists are fused using **Reciprocal Rank Fusion (RRF)** with constant `k = 60`:

    RRF_score(d) = Sum over lists i of 1 / (k + rank_i(d))

5. Final results are sorted by RRF score and returned with metadata.

### 3.5 Interface Layer

Three independent access modes are provided:

- **Streamlit UI** (`app.py`) — three tabs (text search, audio search, ingestion), with example query buttons, modality filters, and rendered result cards.
- **FastAPI REST API** (`src/api.py`) — seven endpoints with auto-generated Swagger documentation at `/docs`.
- **Command-line interface** (`scripts/search.py`, `scripts/search_by_audio.py`) — for scripted use and debugging.

---

## 4. Technical Challenges and Solutions

### 4.1 Challenge: Combining Heterogeneous Embedding Spaces

**Problem:** MiniLM produces 384-dimensional semantic embeddings optimised for sentence similarity. CLIP produces 512-dimensional joint image-text embeddings optimised for visual-linguistic alignment. They are not compatible — their cosine distances live on different scales and represent different things.

**Initial approach considered:** Project both into a common space using a learned linear adapter. Rejected as outside the project scope and overkill for a proof of concept.

**Solution adopted:** Maintain two separate collections and combine them at the **rank level** using Reciprocal Rank Fusion. RRF is robust to score-distribution differences because it uses only the rank of each result within its source list, not the raw similarity scores. The constant `k = 60` follows the original RRF paper's recommendation and damps the influence of low-rank items.

### 4.2 Challenge: CLIP "Always Returns Something"

**Problem:** CLIP's text-to-image retrieval returns a ranked list for *any* query, even queries that have no good visual match (e.g., *"how do I make spaghetti"* against a workstation video). When fused naively, these spurious matches contaminated the final results.

**Diagnostic approach:** Wrote a probe script (`scripts/clip_distance_probe.py`) to record CLIP cosine distances for known-good and known-bad query/video pairs. Found a clear bimodal distribution: good matches clustered around 0.65–0.75, weak matches around 0.80–0.90.

**Solution adopted:** Added a **CLIP distance gate at 0.78** — any CLIP result with distance above this threshold is dropped *before* RRF fusion. This threshold was calibrated against the test dataset; for a new corpus, it would require recalibration. This step turns CLIP from a noisy contributor into a precise one.

### 4.3 Challenge: Video Retrieval Robustness

**Problem:** A single video has multiple semantic dimensions — what's *shown* (visual), what's *said* (transcript), what's *describable about it* (caption). Relying on only one of these creates blind spots: pure visual matching fails for narration-heavy videos; pure transcript matching fails for silent or visual-heavy videos.

**Solution adopted:** Index every video through **three parallel paths** — Whisper transcript, CLIP frame embeddings, and BLIP captions. RRF then fuses whichever path produces a high-ranking match. The query *"a person at a desk with a laptop"* against the demo video produces high-ranked hits via all three paths simultaneously, validating the design.

### 4.4 Challenge: Cross-Modal Audio Queries

**Problem:** The brief requires that a *spoken* query (audio file) be matched against text documents. Audio embeddings (e.g., Wav2Vec2, CLAP) live in a different space from text embeddings, making direct similarity search impractical for this dataset size.

**Solution adopted:** Use Whisper as a bridge — transcribe the spoken query into text, then run the standard text-search pipeline on the transcript. This is pragmatic (Whisper is reliable for short queries on clear audio) and lets the audio-as-query feature reuse all the existing retrieval infrastructure. The `/search/audio` endpoint returns both the transcript and the search results, providing transparency to the user.

### 4.5 Challenge: Reproducibility and Local Hosting

**Problem:** Many embedding/transcription solutions rely on paid APIs (OpenAI, Pinecone, Supabase). These introduce cost, rate limits, network dependence, and reproducibility issues for an academic submission.

**Solution adopted:** Every model in the stack is open-source and runs locally on CPU:
- MiniLM (~90 MB)
- Whisper base (~140 MB)
- CLIP ViT-B-32 (~605 MB)
- BLIP base (~990 MB)

Total cold-start download is ~1.8 GB, cached in the user's `~/.cache/` directory. ChromaDB persists to a local folder. The system has zero external dependencies after initial setup.

---

## 5. Evaluation

### 5.1 Functional Verification

All four required cross-modal queries were verified end-to-end through both the CLI and Streamlit UI:

| Query | Expected Result | Observed Rank | Status |
|---|---|---|---|
| *"how do I make spaghetti"* | `cooking_pasta.txt` | #1 | Pass |
| *"greenhouse gases and fossil fuels"* | `recording_climate.mp4` (audio) | #2 | Pass |
| *"a person at a desk with a laptop"* | `workstation_ai.mp4` (video) | #1 (via 3 paths) | Pass |
| *spoken: "Tell me about climate change"* | `climate_change.txt` | #1 | Pass |

### 5.2 Quantitative Notes

The test dataset is small (16 documents in the main collection, 5 frame embeddings in the visual collection), so traditional IR metrics (mAP, nDCG) would be over-fit to the dataset. Functional pass/fail at rank-1 or top-3 was used as the success criterion, mirroring the brief's expectations.

### 5.3 Performance

On a typical CPU laptop:
- Text query end-to-end: < 300 ms
- Audio query (Whisper + search): 2–5 seconds for a 5-second clip
- Video ingestion: 30–120 seconds per minute of source video (BLIP captioning dominates)

These figures are acceptable for a single-user proof-of-concept. Scaling to many concurrent users or large corpora would require GPU acceleration and an approximate-NN index.

---

## 6. Limitations and Future Work

### 6.1 Known Limitations

- **Linear scan.** ChromaDB's default index is exact; performance will degrade above ~100k documents. Migration to HNSW or IVF indexes (both supported by Chroma) would be required for production scale.
- **CLIP threshold is dataset-specific.** The 0.78 distance gate was calibrated against this dataset. A different corpus would require recalibration, ideally via a small held-out validation set.
- **Whisper accuracy on noisy audio.** The `base` model degrades with strong accents, low-quality microphones, or background noise. A larger Whisper variant or domain fine-tuning could help.
- **Single-user, single-machine.** No authentication, no concurrency controls, no rate limiting. ChromaDB's persistent client locks the database, preventing parallel processes.
- **BLIP captions are generic.** Captions like "a photo of a man sitting at a desk" are accurate but low in specificity. Domain-adapted captioning would improve precision.

### 6.2 Future Work

- **Native audio embeddings** (Wav2Vec2 or CLAP) as a third collection, enabling direct audio-to-audio search (e.g., find similar-sounding clips).
- **Temporal segment retrieval** — return the specific timestamp range of a video that matches, not just the closest frame.
- **Learned RRF weighting** — currently each list contributes equally; weights could be tuned per query type.
- **Approximate-NN index** for scalability past 100k documents.
- **Multi-language support** — the current pipeline assumes English; multilingual MiniLM variants exist.

---

## 7. Conclusion

This project delivers a working proof-of-concept that exceeds the brief's functional requirements: all four cross-modal query types succeed against the test dataset, three independent interfaces (CLI, REST, web UI) are operational, and the entire system runs locally on free models with no external dependencies.

The principal engineering insight is that **multi-modal retrieval does not require a single unified embedding space.** By maintaining two purpose-built collections and combining them at the rank level with RRF and a calibrated distance gate, the system retains the strengths of each underlying model while presenting a unified query interface to the user.

---

## Appendix A — Repository Structure

See the project structure section in `README.md` at the repository root.

## Appendix B — References

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR.
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* (CLIP paper.)
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* (Foundation of MiniLM.)
- Li, J. et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training.*
- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* (Whisper paper.)

---

*End of report.*
