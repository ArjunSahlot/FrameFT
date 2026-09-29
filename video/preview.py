"""Contact sheet of a rendered scene, one frame shortly after each bookmark.   python preview.py SCENE [QUALITY_DIR]

Writes build/preview/<SCENE>.png, for checking layout and sync without watching the whole video.
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"


def main():
    scene = sys.argv[1]
    quality = sys.argv[2] if len(sys.argv) > 2 else "480p15"
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    video = next((BUILD / "media" / "videos").glob(f"*/{quality}/{scene}.mp4"))
    timeline = json.loads((BUILD / "timeline" / quality / f"{scene}.json").read_text())
    timings = json.loads((BUILD / "audio" / "timings.json").read_text())
    shots = [("start", 0.8)]
    for seg in timeline["segments"]:
        for mark, t in timings[seg["id"]]["marks"].items():
            shots.append((f"{seg['id']}:{mark}", seg["t"] + t + delay))
        shots.append((f"{seg['id']}:end", seg["t"] + timings[seg["id"]]["duration"] - 0.2))
    tmp = BUILD / "preview" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (name, t) in enumerate(shots):
        t = min(t, timeline["duration"] - 0.1)
        out = tmp / f"{i:03d}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=640:-1", str(out)], check=True)
        frames.append((name, t, Image.open(out).convert("RGB")))
    cols = 4
    w, h = frames[0][2].size
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 22)), "black")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (name, t, im) in enumerate(frames):
        x, y = (i % cols) * w, (i // cols) * (h + 22)
        sheet.paste(im, (x, y + 22))
        draw.text((x + 4, y + 4), f"{name} @ {t:.1f}s", fill="white", font=font)
    out = BUILD / "preview" / f"{scene}.png"
    sheet.save(out)
    print(out, len(frames), "frames")


if __name__ == "__main__":
    main()
