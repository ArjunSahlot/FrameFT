# FrameFT, explained (animated video)

An animated walkthrough of FrameFT: what problem it solves, what a tight fusion frame is, how
`construct_real_tff` builds one, how the update `ΔW = (scale/n) · Bᵀ S B` is trained and merged,
the design choices in the code, and the frame vs. random vs. identity basis experiment in `GLUE/`.

- **Video:** [`out/frameft_explained.mp4`](out/frameft_explained.mp4) (1080p, voiceover, burned-in
  captions, chapter markers)
- **Captions:** [`out/frameft_explained.srt`](out/frameft_explained.srt)
- **Script:** [`SCRIPT.md`](SCRIPT.md) (the narration as plain text)

## Chapters

<!-- chapters:start -->
| Time | Chapter |
| :--- | :--- |
| 0:00 | 01 · FrameFT, end to end |
| 0:42 | 02 · The problem: fine-tuning |
| 1:30 | 03 · Parameter-efficient fine-tuning |
| 2:27 | 04 · Bases and frames |
| 3:42 | 05 · Fusion frames |
| 4:48 | 06 · Building the frame |
| 6:39 | 07 · The FrameFT update |
| 8:09 | 08 · Two knobs: block size and scale |
| 9:18 | 09 · Training and inference |
| 10:53 | 10 · Results from the paper |
| 11:33 | 11 · Our experiment: does the frame matter? |
| 13:30 | 12 · Recap |
<!-- chapters:end -->

## Where the pictures come from

Nothing in the video is a mock-up of the method:

- Every basis is computed by the repository's own `peft.tuners.frame.layer.build_basis`, and the 1,000
  coefficient positions use the same seeded `randperm` as `FrameLayer.update_layer` (`entry_seed=2024`,
  `share_entry`). See `data.py`.
- The "real weight matrix" is layer 1's query weight from `roberta-base` when the weights are available
  (set `ROBERTA_SAFETENSORS`), otherwise a look-alike stand-in.
- The claim that all three bases give updates with identical singular values is computed, not assumed: the
  chapter 11 plot draws `svd(Bᵀ S B)` for each basis from `data.py`.
- The experiment's per-seed dots are read from `GLUE/figures/fig_results.png`. Accuracies are multiples of
  1/277 (RTE) and 1/408 (MRPC), so each dot is an exact count of correct examples, and every group's mean
  matches the value printed on that figure. The training-loss panel is cropped from
  `GLUE/figures/fig_curves.png`.
- Paper numbers (GLUE, Llama-2-7B, ViT-L) are the tables in the repository's main `README.md`.

## How it's built

| Step | File | What it does |
| :--- | :--- | :--- |
| Script | `narration.py` | 12 chapters of narration. `{display\|spoken}` sets what the captions show vs. what the voice says; `[[mark]]` is a bookmark a scene can wait for. |
| Voice | `tts.py` | Kokoro TTS (`af_heart`). Each sentence is synthesized separately and joined with fixed pauses, and Kokoro's word timestamps give exact times for every bookmark and caption chunk. Writes `build/audio/`. |
| Pictures | `data.py` | Bases, coefficient positions, ΔW, singular values, all from the repo's code. Cached in `build/data.npz`. |
| Animation | `common.py`, `scenes_1.py` … `scenes_4.py` | Manim scenes. `self.narrate(id)` records when a paragraph starts, `self.at(tr, "mark")` waits for a spoken word, and `self.upto(tr, "mark")` makes an animation end on it. |
| Render | `render.sh` | Renders all scenes at 1080p30, three at a time. Caching is off on purpose (`manim.cfg`): Manim's clock is frame-exact only when every animation is rendered. |
| Assemble | `build.py` | Concatenates the scenes, places each paragraph's audio at the time its scene logged, writes SRT and styled ASS captions and chapter markers, and encodes the final MP4. |
| Check | `preview.py` | A contact sheet with one frame after every bookmark, for checking layout without watching the whole video. |
| Check | `qa_transcript.py` | Transcribes every paragraph with Whisper and lists words heard differently from the script (catches mispronounced terms). |
| Check | `qa_sync.py` | Transcribes the finished MP4 and measures how far each bookmarked word lands from the moment its animation was scheduled. |

### Rebuild from scratch

System packages (Ubuntu): `ffmpeg libcairo2-dev libpango1.0-dev pkg-config texlive-latex-base
texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-science cm-super dvisvgm
espeak-ng fonts-inter fonts-jetbrains-mono`.

```bash
cd video
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # CPU-only torch is fine
export ROBERTA_SAFETENSORS=/path/to/roberta-base/model.safetensors   # optional, for the real W

python data.py                            # matrices from the repo's FrameFT code
python tts.py                             # voiceover + timings (about 7 minutes on 4 CPU cores)
bash render.sh                            # all scenes at 1080p30
python build.py                           # out/frameft_explained.mp4 and .srt
```

To change a line of narration, edit `narration.py`, run `python tts.py --only <segment_id>`, then
re-render the scene that narrates it (`bash render.sh S07Update`) and run `python build.py`. Scenes wait
for the new timings automatically.

For a fast draft of one scene: `manim -r 854,480 --frame_rate 15 scenes_3.py S07Update`, then
`python preview.py S07Update`.
