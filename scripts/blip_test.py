"""Test BLIP: caption a video frame and check the output is sensible."""
from src.video_ingest import sample_frames
from src.blip_caption import caption_image

print("Sampling one frame from video...")
frames = sample_frames("data/video/workstation_ai.mp4")
ts, img = frames[len(frames) // 2]  # middle frame
print(f"Picked frame at t={ts:.1f}s")

print()
print("Generating BLIP caption (first run downloads ~990 MB)...")
caption = caption_image(img)
print()
print(f"Caption: {caption!r}")

print()
print("Also trying with a prompt for richer description...")
caption2 = caption_image(img, prompt="a photo of")
print(f"Prompted caption: {caption2!r}")
