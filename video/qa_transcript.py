"""Check the voiceover against the script with speech recognition.   python qa_transcript.py [segment ...]

Transcribes every narration paragraph with Whisper and prints the words it heard differently from the
spoken script, so mispronounced terms stand out. Numbers are compared as words ("768" vs "seven sixty-eight"
would otherwise look like errors), so the check is on the spoken form.
"""
import difflib
import re
import sys
from pathlib import Path

from narration import segments, spoken_text

HERE = Path(__file__).resolve().parent
AUDIO = HERE / "build" / "audio"
MARK = re.compile(r"\[\[\w+\]\]")

# spellings that sound the same: the script spells acronyms out letter by letter, Whisper writes them joined
JOIN = [("frame f t", "frameft"), ("frame ft", "frameft"), ("fourier f t", "fourierft"), ("fourier ft", "fourierft"),
        ("p e f t", "peft"), ("r t e", "rte"), ("m r p c", "mrpc"), ("q r", "qr"), ("g p u", "gpu"), ("jay peg", "jpeg"),
        ("roberta base", "roberta"), ("laura", "lora")]


def words(text):
    text = " " + " ".join(re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))) + " "
    for a, b in JOIN:
        text = text.replace(f" {a} ", f" {b} ")
    return text.split()


def main():
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    only = set(sys.argv[1:])
    flagged = 0
    for sid, markup in segments():
        if only and sid not in only:
            continue
        expected = words(MARK.sub("", spoken_text(markup)))
        segs, _ = model.transcribe(str(AUDIO / f"{sid}.wav"), beam_size=5, language="en")
        heard_text = " ".join(s.text for s in segs)
        heard = words(heard_text)
        sm = difflib.SequenceMatcher(None, expected, heard, autojunk=False)
        diffs = [(" ".join(expected[a:b]), " ".join(heard[c:d])) for op, a, b, c, d in sm.get_opcodes() if op != "equal"]
        # Whisper writes numbers as digits; those differences are spelling, not pronunciation
        diffs = [(e, h) for e, h in diffs if not re.search(r"\d", h)]
        ratio = sm.ratio()
        print(f"{sid:14s} match {ratio:.3f}" + ("" if not diffs else "  " + " | ".join(f"{e!r}->{h!r}" for e, h in diffs)))
        flagged += ratio < 0.9
    print(f"{flagged} paragraphs below 0.90 word match")


if __name__ == "__main__":
    main()
