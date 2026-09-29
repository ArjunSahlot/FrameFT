"""Check audio/visual sync in the finished video.   python qa_sync.py

Every animation waits for a bookmark at (scene start + paragraph start + bookmark offset). This script
transcribes the final MP4's soundtrack with word timestamps and checks that the word right after each
bookmark is actually heard at that moment, i.e. that the voice landed where the animations expect it.
"""
import json
import re
from pathlib import Path

from build import BUILD, OUT, QUALITY, SCENES, probe_duration
from narration import SCRIPT, spoken_text

MARK = re.compile(r"\[\[(\w+)\]\]")


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def main():
    from faster_whisper import WhisperModel
    timings = json.loads((BUILD / "audio" / "timings.json").read_text())
    expected, t0 = [], 0.0
    for name in SCENES:
        video = next((BUILD / "media" / "videos").glob(f"*/{QUALITY}/{name}.mp4"))
        tl = json.loads((BUILD / "timeline" / QUALITY / f"{name}.json").read_text())
        for seg in tl["segments"]:
            chapter, idx = seg["id"].rsplit("_", 1)
            spoken = spoken_text(SCRIPT[chapter][int(idx)])
            for m in MARK.finditer(spoken):
                nxt = re.findall(r"[A-Za-z]+", MARK.sub("", spoken[m.end():]))[:1]
                if nxt:
                    expected.append((seg["id"], m.group(1), norm(nxt[0]), t0 + seg["t"] + timings[seg["id"]]["marks"][m.group(1)]))
        t0 += probe_duration(video)

    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(OUT / "frameft_explained.mp4"), word_timestamps=True, beam_size=5)
    heard = [(norm(w.word), w.start) for s in segs for w in s.words]

    errors = []
    for sid, mark, word, t in expected:
        near = [ts for w, ts in heard if abs(ts - t) < 2.0 and (w.startswith(word[:4]) or word.startswith(w[:4]) and w)]
        if near:
            errors.append(min(near, key=lambda ts: abs(ts - t)) - t)
        else:
            print(f"  not matched: {sid}:{mark} ({word!r} at {t:.2f}s)")
    errors.sort()
    n = len(errors)
    print(f"{n}/{len(expected)} bookmarks matched; offset of heard word vs. scheduled time (s): "
          f"median {errors[n // 2]:+.3f}, 5th pct {errors[n // 20]:+.3f}, 95th pct {errors[-(n // 20) - 1]:+.3f}")


if __name__ == "__main__":
    main()
