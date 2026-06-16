"""Test frame sampling + audio extraction. No CLIP/BLIP yet."""
from pathlib import Path
from src.video_ingest import sample_frames, extract_audio

video = "data/video/workstation_ai.mp4"

print("=" * 60)
print("FRAME SAMPLING TEST")
print("=" * 60)
frames = sample_frames(video)
print(f"\nGot {len(frames)} frames:")
for ts, img in frames:
    print(f"  t={ts:5.2f}s  size={img.size}  mode={img.mode}")

print()
print("=" * 60)
print("AUDIO EXTRACTION TEST")
print("=" * 60)
audio_path = extract_audio(video)
print(f"\nExtracted audio: {audio_path}")
print(f"File size: {Path(audio_path).stat().st_size} bytes")
