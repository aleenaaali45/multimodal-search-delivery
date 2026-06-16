# Multimodal Search Delivery - Project Report

## 1. Architecture

The system supports three modalities:
- Text: chunked and embedded using MiniLM (384-dim) into ChromaDB
- Audio: transcribed with Whisper, then embedded with MiniLM
- Video: three signals extracted:
  - Whisper transcript (MiniLM -> main collection)
  - CLIP frame embeddings (512-dim -> visual collection)
  - BLIP captions (MiniLM -> main collection)

Search uses Reciprocal Rank Fusion (RRF) to merge results from both collections.

## 2. Technical Challenges & Solutions

### Challenge 1: numba/llvmlite crash on Windows
Whisper's timing.py imports numba which caused an access violation.
Solution: Replaced timing.py with a stub that returns segments unchanged.

### Challenge 2: Memory conflict between Whisper and SentenceTransformer
Loading both models in the same process caused silent crashes on 8GB RAM.
Solution: Whisper transcription runs in a separate subprocess via whisper_transcribe.py.

### Challenge 3: ChromaDB duplicate IDs
Running ingest scripts multiple times caused silent failures.
Solution: Used reset_all() to clear collections before re-ingestion.

## 3. Technologies Used
- ChromaDB: vector database
- sentence-transformers/all-MiniLM-L6-v2: text embeddings
- OpenAI Whisper: audio transcription
- CLIP (ViT-B/32): visual frame embeddings
- BLIP: image captioning
- Streamlit: web interface
- ffmpeg: audio/video processing