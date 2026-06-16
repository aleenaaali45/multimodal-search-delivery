"""End-to-end video ingestion test. Runs Whisper + CLIP + BLIP in one pass."""
from src.vectorstore import count, visual_count
from src.video_ingest import ingest_video_file

print("BEFORE ingest:")
print(f"  main collection:   {count()}")
print(f"  visual collection: {visual_count()}")
print()

summary = ingest_video_file("data/video/workstation_ai.mp4")

print()
print("AFTER ingest:")
print(f"  main collection:   {count()}")
print(f"  visual collection: {visual_count()}")
print(f"  per-path summary:  {summary}")
