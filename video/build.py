"""Assemble the final video from the rendered scenes.   python build.py

1. checks every scene rendered and that its length matches the timeline Manim logged,
2. concatenates the scenes,
3. lays each narration paragraph onto one audio track at the exact time its scene logged,
4. writes captions (SRT, plus a styled ASS version that is burned in) and chapter markers,
5. encodes out/frameft_explained.mp4.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from narration import CHAPTERS

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
OUT = HERE / "out"
QUALITY = "1080p30"
SR = 24000
SCENES = ["S01Intro", "S02Problem", "S03Landscape", "S04Frames", "S05aFusion", "S05bFusion", "S06Build", "S07Update",
          "S08Knobs", "S09Training", "S10Results", "S11Experiment", "S12Recap"]
GAP = 0.0  # scenes already fade out and in


def run(cmd):
    subprocess.run(cmd, check=True)


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def ts_srt(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def ts_ass(t):
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Inter,46,&H00F3EDE9,&H000000FF,&H24302119,&H00000000,0,0,0,0,100,100,0,0,3,14,0,2,230,230,28,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def main():
    OUT.mkdir(exist_ok=True)
    final = BUILD / "final"
    final.mkdir(exist_ok=True)
    timings = json.loads((BUILD / "audio" / "timings.json").read_text())

    # ---- 1. scenes, durations, offsets
    videos, offsets, t = [], {}, 0.0
    for name in SCENES:
        video = next((BUILD / "media" / "videos").glob(f"*/{QUALITY}/{name}.mp4"))
        timeline = json.loads((BUILD / "timeline" / QUALITY / f"{name}.json").read_text())
        dur = probe_duration(video)
        if abs(dur - timeline["duration"]) > 0.05:
            raise SystemExit(f"{name}: video is {dur:.3f}s but its timeline says {timeline['duration']:.3f}s; re-render it")
        videos.append(video)
        offsets[name] = (t, timeline)
        t += dur
    total = t
    print(f"{len(videos)} scenes, {total / 60:.2f} min")

    # ---- 2. concatenate video
    listing = final / "scenes.txt"
    listing.write_text("".join(f"file '{v}'\n" for v in videos))
    silent = final / "video_silent.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(silent)])

    # ---- 3. narration track
    audio = np.zeros(int((total + 1) * SR), dtype=np.float32)
    cues, seen = [], set()
    for name in SCENES:
        start, timeline = offsets[name]
        for seg in timeline["segments"]:
            sid = seg["id"]
            if sid in seen:
                raise SystemExit(f"segment {sid} narrated twice")
            seen.add(sid)
            wav, sr = sf.read(BUILD / "audio" / f"{sid}.wav", dtype="float32")
            assert sr == SR
            at = int(round((start + seg["t"]) * SR))
            if np.any(audio[at:at + len(wav)] != 0):
                raise SystemExit(f"segment {sid} overlaps the previous one")
            audio[at:at + len(wav)] += wav
            for c in timings[sid]["captions"]:
                cues.append([start + seg["t"] + c["start"], start + seg["t"] + c["end"], c["text"]])
    missing = [sid for sid in timings if sid not in seen]
    if missing:
        raise SystemExit(f"never narrated: {missing}")
    audio = audio[:int(total * SR)]
    voice = final / "voice.wav"
    sf.write(voice, audio, SR, subtype="PCM_16")

    # ---- 4. captions and chapters
    cues.sort()
    for i, cue in enumerate(cues):  # hold each caption a moment, never into the next one
        nxt = cues[i + 1][0] if i + 1 < len(cues) else total
        cue[1] = min(cue[1] + 0.35, nxt - 0.04)
    srt = "".join(f"{i + 1}\n{ts_srt(a)} --> {ts_srt(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(cues))
    (OUT / "frameft_explained.srt").write_text(srt)
    ass = ASS_HEADER + "".join(
        f"Dialogue: 0,{ts_ass(a)},{ts_ass(b)},Cap,,0,0,0,,{text.replace(chr(10), ' ')}\n" for a, b, text in cues)
    ass_path = final / "captions.ass"
    ass_path.write_text(ass)

    chapter_titles = dict(CHAPTERS)
    first_scene = {}
    for name in SCENES:
        first = offsets[name][1]["segments"][0]["id"].rsplit("_", 1)[0]
        first_scene.setdefault(first, offsets[name][0])
    meta = [";FFMETADATA1", "title=FrameFT, explained", "artist=FrameFT explainer"]
    keys = [k for k, _ in CHAPTERS]
    for i, key in enumerate(keys):
        a = first_scene[key]
        b = first_scene[keys[i + 1]] if i + 1 < len(keys) else total
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(a * 1000)}", f"END={int(b * 1000)}",
                 f"title={i + 1:02d} {chapter_titles[key]}"]
    meta_path = final / "chapters.txt"
    meta_path.write_text("\n".join(meta) + "\n")

    # the same chapter list, with timestamps, in README.md
    table = ["| Time | Chapter |", "| :--- | :--- |"]
    for i, key in enumerate(keys):
        a = int(first_scene[key])
        table.append(f"| {a // 60}:{a % 60:02d} | {i + 1:02d} · {chapter_titles[key]} |")
    readme = HERE / "README.md"
    text = readme.read_text()
    start, end = "<!-- chapters:start -->", "<!-- chapters:end -->"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        readme.write_text(head + start + "\n" + "\n".join(table) + "\n" + end + tail)

    # ---- 5. encode
    out = OUT / "frameft_explained.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(voice), "-i", str(meta_path),
         "-map", "0:v", "-map", "1:a", "-map_metadata", "2", "-map_chapters", "2",
         "-vf", f"ass={ass_path}", "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-tune", "animation",
         "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", str(out)])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {len(cues)} captions")


if __name__ == "__main__":
    main()
