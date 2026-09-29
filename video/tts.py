"""Synthesize the voiceover with Kokoro and record exact timings.   python tts.py [--only SEGMENT_ID ...]

For every paragraph in narration.py this writes build/audio/<segment>.wav and an entry in
build/audio/timings.json:
    duration      seconds of audio
    marks         {bookmark: seconds from the paragraph start}
    captions      [{start, end, text}] caption chunks, seconds from the paragraph start
Sentences are synthesized one at a time, trimmed, and joined with fixed pauses, so the timing of every
sentence and bookmark is known exactly instead of being estimated.
"""
import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from narration import SCRIPT, display_text, segments, split_sentences, spoken_text

SR = 24000
VOICE, SPEED = "af_heart", 1.07
LEAD_IN, SENTENCE_GAP, TAIL = 0.12, 0.34, 0.10
MAX_CAPTION_CHARS = 84

OUT = Path(__file__).resolve().parent / "build" / "audio"
MARK = re.compile(r"\[\[(\w+)\]\]")


def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def trim(audio, threshold=0.004):
    """Cut leading and trailing near-silence, keeping 30 ms of air on each side."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if len(loud) == 0:
        return audio, 0.0
    pad = int(0.03 * SR)
    a, b = max(0, loud[0] - pad), min(len(audio), loud[-1] + pad)
    return audio[a:b], a / SR


def caption_chunks(markup):
    """Split one sentence (markup form) into caption-sized pieces at commas, colons and dashes."""
    if len(display_text(markup)) <= MAX_CAPTION_CHARS:
        return [markup]
    # candidate split points: after , : ; outside braces
    points, depth = [], 0
    for i, ch in enumerate(markup):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0 and ch in ",:;" and i + 1 < len(markup) and markup[i + 1] == " ":
            points.append(i + 1)
    chunks, start = [], 0
    while len(display_text(markup[start:])) > MAX_CAPTION_CHARS:
        good = [p for p in points if p > start and len(display_text(markup[start:p])) <= MAX_CAPTION_CHARS]
        # prefer the split that balances the two halves of what remains
        if good:
            rest = len(display_text(markup[start:]))
            cut = min(good, key=lambda p: abs(len(display_text(markup[start:p])) - min(rest / 2, MAX_CAPTION_CHARS)))
        else:  # no punctuation: split at the space closest to the middle
            spaces = [m.start() for m in re.finditer(" ", markup) if m.start() > start
                      and markup[start:m.start()].count("{") == markup[start:m.start()].count("}")
                      and len(display_text(markup[start:m.start()])) <= MAX_CAPTION_CHARS]
            if not spaces:
                break
            mid = start + len(markup[start:]) / 2
            cut = min(spaces, key=lambda p: abs(p - mid))
        chunks.append(markup[start:cut].strip())
        start = cut
    chunks.append(markup[start:].strip())
    return [c for c in chunks if c]


class Aligner:
    """Maps character positions in the spoken text to times, using Kokoro's word timestamps.

    Kokoro's tokens don't always spell words exactly as the script does, so the two character streams
    are aligned with difflib, and unmatched characters are interpolated between their neighbours.
    """

    def __init__(self, tokens, duration, spoken):
        self.duration = duration
        stream, times = "", []
        for text, t0, t1 in tokens:
            n = norm(text)
            if not n or t0 is None:
                continue
            for j, _ in enumerate(n):
                times.append(t0 + (t1 - t0) * j / len(n))  # spread a word's characters over its duration
            stream += n
        target = norm(spoken)
        self.map = np.full(len(target) + 1, np.nan)
        if stream:
            sm = difflib.SequenceMatcher(None, target, stream, autojunk=False)
            for a, b, size in sm.get_matching_blocks():
                for j in range(size):
                    self.map[a + j] = times[b + j]
        self.map[len(target)] = duration
        if np.isnan(self.map[0]):
            self.map[0] = times[0] if times else 0.0
        known = np.flatnonzero(~np.isnan(self.map))
        self.map = np.interp(np.arange(len(self.map)), known, self.map[known])

    def time_at(self, spoken_prefix):
        """Time at which the word right after `spoken_prefix` begins."""
        return float(self.map[min(len(norm(spoken_prefix)), len(self.map) - 1)])


def synth_sentence(pipeline, spoken):
    chunks, tokens, offset = [], [], 0.0
    for r in pipeline(spoken, voice=VOICE, speed=SPEED):
        audio = r.audio.numpy()
        for tk in (r.tokens or []):
            if tk.start_ts is not None:
                tokens.append((tk.text, offset + tk.start_ts, offset + tk.end_ts))
        chunks.append(audio)
        offset += len(audio) / SR
    return np.concatenate(chunks), tokens


def build_segment(pipeline, markup):
    pieces, marks, captions = [np.zeros(int(LEAD_IN * SR), dtype=np.float32)], {}, []
    t = LEAD_IN
    for sentence in split_sentences(markup):
        spoken_marked = spoken_text(sentence)
        spoken = MARK.sub("", spoken_marked)
        audio, tokens = synth_sentence(pipeline, spoken)
        audio, cut = trim(audio)
        tokens = [(w, a - cut, b - cut) for w, a, b in tokens]
        dur = len(audio) / SR
        align = Aligner(tokens, dur, spoken)
        for m in MARK.finditer(spoken_marked):
            prefix = MARK.sub("", spoken_marked[:m.start()])
            marks[m.group(1)] = round(t + max(0.0, align.time_at(prefix) - 0.04), 3)
        # caption chunks start when their first word is spoken
        chunks = caption_chunks(sentence)
        starts, consumed = [], ""
        for c in chunks:
            starts.append(t + (0.0 if not consumed else align.time_at(MARK.sub("", spoken_text(consumed)))))
            consumed += c + " "
        for i, c in enumerate(chunks):
            end = starts[i + 1] if i + 1 < len(chunks) else t + dur
            captions.append({"start": round(starts[i], 3), "end": round(end, 3), "text": display_text(c).strip()})
        pieces.append(audio.astype(np.float32))
        t += dur
        pieces.append(np.zeros(int(SENTENCE_GAP * SR), dtype=np.float32))
        t += SENTENCE_GAP
    pieces[-1] = np.zeros(int(TAIL * SR), dtype=np.float32)
    t += TAIL - SENTENCE_GAP
    audio = np.concatenate(pieces)
    return audio, {"duration": round(len(audio) / SR, 3), "marks": marks, "captions": captions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="segment ids to (re)generate")
    args = parser.parse_args()

    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    OUT.mkdir(parents=True, exist_ok=True)
    timings_path = OUT / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    for sid, markup in segments():
        if args.only and sid not in args.only:
            continue
        audio, info = build_segment(pipeline, markup)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.89 / peak)
        sf.write(OUT / f"{sid}.wav", audio, SR, subtype="PCM_16")
        timings[sid] = info
        missing = set(MARK.findall(markup)) - set(info["marks"])
        print(f"{sid:14s} {info['duration']:6.1f}s  marks={len(info['marks'])}" + (f"  MISSING {missing}" if missing else ""))
        timings_path.write_text(json.dumps(timings, indent=1))
    total = sum(v["duration"] for v in timings.values())
    print(f"total narration {total / 60:.1f} min")


if __name__ == "__main__":
    main()
