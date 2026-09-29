"""Shared look, helpers, and the narration-synced Scene base class.

Timing model: every scene calls `tr = self.narrate("<segment>")`, which records the exact video time at
which that paragraph's audio will be placed. `self.at(tr, "mark")` waits until a bookmark is spoken, and
`self.upto(tr, "mark")` returns a run_time that makes an animation end on the bookmark. The clock is
Manim's own frame counter (`renderer.time`), so there is no drift. Rendering must use --disable_caching,
because cached animations advance that clock by un-rounded durations.
"""
import json
import textwrap
from pathlib import Path

import numpy as np
from manim import *  # noqa: F401,F403
from PIL import Image

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
TIMINGS = json.loads((BUILD / "audio" / "timings.json").read_text())
FPS = 30
PX_PER_UNIT = 1080 / 8

# ------------------------------------------------------------------ palette
BG = "#0D1117"
PANEL = "#151B24"
PANEL_2 = "#1C2430"
GRID = "#2A3340"
INK = "#E9EDF3"
INK_2 = "#AAB3C0"
MUTED = "#6E7887"
# the three bases keep the colours of GLUE/plot_results.py, lifted for a dark background
FRAME_C = "#4D9BFF"
RANDOM_C = "#FF7F4A"
IDENT_C = "#2FD69C"
TRAIN_C = "#FFBE4D"   # trainable things (coefficients, head)
FROZEN_C = "#8FC8FF"  # frozen things
DW_C = "#C293FF"      # delta W
W_C = "#8C9AAE"       # pretrained weights
GOOD = "#57D98B"
BAD = "#FF6B6B"

FONT = "Inter"
MONO = "JetBrains Mono"


def T(text, size=28, color=INK, weight=NORMAL, **kw):
    return Text(text, font=FONT, font_size=size, color=color, weight=weight, **kw)


def Mono(text, size=24, color=INK_2, **kw):
    return Text(text, font=MONO, font_size=size, color=color, **kw)


def M(tex, size=44, color=INK, **kw):
    return MathTex(tex, font_size=size, color=color, **kw)


def wrap(text, chars):
    return "\n".join(textwrap.wrap(text, chars))


# ------------------------------------------------------------------ images of matrices
_NEG = np.array([0x4F, 0xC3, 0xF7]) / 255  # cyan
_MID = np.array([0x10, 0x15, 0x1C]) / 255
_POS = np.array([0xFF, 0xB8, 0x6B]) / 255  # peach


def colorize(arr, vmax=None, cmap="div", gamma=0.75):
    """Map a 2D array to RGB. 'div': cyan (negative) / dark (zero) / peach (positive)."""
    arr = np.asarray(arr, dtype=np.float64)
    if vmax is None:
        vmax = np.percentile(np.abs(arr), 99.5) or 1.0
    x = np.clip(arr / vmax, -1, 1)
    if cmap == "div":
        m = np.abs(x)[..., None] ** gamma
        col = np.where(x[..., None] >= 0, _POS, _NEG)
        rgb = _MID * (1 - m) + col * m
    elif cmap == "mono":  # magnitude only, for sparse masks
        m = np.abs(x)[..., None] ** gamma
        rgb = _MID * (1 - m) + np.array(ManimColor(TRAIN_C).to_rgb()) * m
    else:
        raise ValueError(cmap)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def mat_image(arr, height, vmax=None, cmap="div", width=None, border=GRID, stroke=1.5, dilate=0, gamma=0.75, dim=1.0):
    """A crisp picture of a matrix: resized with box filtering to the exact on-screen pixel size.

    `dim` < 1 blends the picture into the background (ImageMobject opacity doesn't survive FadeIn)."""
    arr = np.asarray(arr, dtype=np.float64)
    if dilate:  # make isolated nonzeros visible at video resolution
        from scipy.ndimage import maximum_filter, minimum_filter
        pos = maximum_filter(np.maximum(arr, 0), size=2 * dilate + 1)
        neg = minimum_filter(np.minimum(arr, 0), size=2 * dilate + 1)
        arr = np.where(pos >= -neg, pos, neg)
    rgb = colorize(arr, vmax=vmax, cmap=cmap, gamma=gamma)
    if dim < 1.0:
        bg = np.array(ManimColor(BG).to_rgb()) * 255
        rgb = (bg * (1 - dim) + rgb * dim).astype(np.uint8)
    if width is None:
        width = height * arr.shape[1] / arr.shape[0]
    px_w, px_h = max(1, round(width * PX_PER_UNIT)), max(1, round(height * PX_PER_UNIT))
    img = Image.fromarray(rgb).resize((px_w, px_h), Image.BOX)
    mob = ImageMobject(np.array(img))
    mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    mob.stretch_to_fit_width(width).stretch_to_fit_height(height)
    group = Group(mob)
    if border:
        group.add(Rectangle(width=width, height=height, stroke_color=border, stroke_width=stroke))
    return group


def strip(vec, width, height=0.32, vmax=None, border=GRID):
    """A row vector drawn as a thin coloured strip."""
    return mat_image(np.asarray(vec)[None, :], height, vmax=vmax, width=width, border=border)


# ------------------------------------------------------------------ small vector widgets
def cell_matrix(values, cell=0.62, fmt="{:.2f}", size=22, fill_fn=None, stroke=GRID, text_color=INK):
    values = np.asarray(values)
    g = VGroup()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            sq = Square(cell, stroke_color=stroke, stroke_width=1.5)
            if fill_fn:
                sq.set_fill(fill_fn(v, i, j), opacity=1)
            sq.move_to([j * cell, -i * cell, 0])
            label = T(fmt.format(v) if not isinstance(v, str) else v, size, text_color)
            if label.width > cell * 0.9:
                label.scale_to_fit_width(cell * 0.86)
            label.move_to(sq)
            g.add(VGroup(sq, label))
    g.center()
    return g


def snowflake(size=0.32, color=FROZEN_C):
    arms = VGroup()
    for k in range(3):
        arms.add(Line(DOWN * size / 2, UP * size / 2, stroke_width=3, color=color).rotate(k * PI / 3))
    for k in range(6):
        d = rotate_vector(UP, k * PI / 3)
        tip = d * size * 0.33
        for s in (-1, 1):
            arms.add(Line(tip, tip + rotate_vector(d, s * PI / 4) * size * 0.14, stroke_width=2.4, color=color))
    return arms


def flame(size=0.34, color=TRAIN_C):
    outer = VMobject(stroke_width=0, fill_color=color, fill_opacity=1)
    outer.set_points_smoothly([
        [0, 1.0, 0], [0.28, 0.55, 0], [0.5, 0.05, 0], [0.38, -0.38, 0], [0, -0.52, 0],
        [-0.38, -0.38, 0], [-0.5, 0.05, 0], [-0.22, 0.42, 0], [-0.05, 0.62, 0], [0, 1.0, 0],
    ])
    inner = VMobject(stroke_width=0, fill_color="#FFF1C9", fill_opacity=1)
    inner.set_points_smoothly([
        [0.02, 0.35, 0], [0.2, -0.05, 0], [0.16, -0.34, 0], [0, -0.42, 0], [-0.16, -0.34, 0],
        [-0.2, -0.08, 0], [-0.05, 0.15, 0], [0.02, 0.35, 0],
    ])
    return VGroup(outer, inner).scale_to_fit_height(size)


def badge(kind, text=None, size=0.3):
    icon = snowflake(size) if kind == "frozen" else flame(size * 1.05)
    color = FROZEN_C if kind == "frozen" else TRAIN_C
    label = T(text or ("frozen" if kind == "frozen" else "trainable"), 20, color)
    return VGroup(icon, label).arrange(RIGHT, buff=0.12)


def pill(text, color=INK_2, size=22, fill=PANEL_2, pad=0.18):
    label = T(text, size, color)
    box = RoundedRectangle(corner_radius=0.16, width=label.width + 2 * pad + 0.1, height=label.height + 2 * pad,
                           stroke_color=color, stroke_width=1.5, fill_color=fill, fill_opacity=1)
    return VGroup(box, label)


def panel(width, height, color=GRID, fill=PANEL, radius=0.18, opacity=1):
    return RoundedRectangle(corner_radius=radius, width=width, height=height, stroke_color=color,
                            stroke_width=1.5, fill_color=fill, fill_opacity=opacity)


def term_card(term, definition, chars=34):
    eyebrow = T("TERM", 16, MUTED, weight=BOLD)
    head = T(term, 28, TRAIN_C, weight=BOLD)
    body = T(wrap(definition, chars), 21, INK_2, line_spacing=0.9)
    content = VGroup(eyebrow, head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    box = panel(max(content.width + 0.5, 3.2), content.height + 0.42, color="#3A4452", fill=PANEL_2)
    content.move_to(box)
    card = VGroup(box, content)
    card.to_corner(UR, buff=0.4).shift(DOWN * 0.55)
    return card


def chapter_tag(number, title):
    num = T(f"{number:02d}", 22, TRAIN_C, weight=BOLD)
    name = T(title.upper(), 19, MUTED, weight=BOLD)
    g = VGroup(num, name).arrange(RIGHT, buff=0.22)
    g.to_corner(UL, buff=0.42)
    return g


def live(func):
    """Like always_redraw, for pictures: `become` can't morph images, so swap the child every frame."""
    holder = Group(func())

    def swap(m):
        m.submobjects = [func()]
    holder.add_updater(swap)
    return holder


def counter_text(tracker, size=96, color=INK, fmt="{:,.0f}", **kw):
    """A number in the house font that follows a ValueTracker (Integer would use the LaTeX font)."""
    return always_redraw(lambda: T(fmt.format(tracker.get_value()), size, color, weight=BOLD, **kw))


# ------------------------------------------------------------------ narration-synced scene
class Tracker:
    def __init__(self, sid, start, info):
        self.sid, self.start = sid, start
        self.duration = info["duration"]
        self.marks = info["marks"]
        self.end = start + self.duration

    def t(self, mark):
        if mark not in self.marks:
            raise KeyError(f"{self.sid} has no mark {mark!r}; has {sorted(self.marks)}")
        return self.start + self.marks[mark]


class NScene(Scene):
    chapter = None  # (number, title)

    def setup(self):
        self.camera.background_color = BG
        self.timeline = []
        self._audio_end = 0.0
        self.tag = None
        if self.chapter:
            self.tag = chapter_tag(*self.chapter)

    @property
    def now(self):
        return self.renderer.time

    def show_tag(self, run_time=0.6):
        if self.tag is not None:
            self.play(FadeIn(self.tag, shift=RIGHT * 0.15), run_time=run_time)

    def narrate(self, sid):
        if self.now < self._audio_end:  # never let two paragraphs overlap
            self.wait(self._audio_end - self.now + 1 / config.frame_rate)
        tr = Tracker(sid, self.now, TIMINGS[sid])
        self.timeline.append({"id": sid, "t": round(self.now, 4)})
        self._audio_end = tr.end
        return tr

    def wait_until(self, t):
        dt = t - self.now
        if dt >= 1 / config.frame_rate:
            self.wait(dt)

    def at(self, tr, mark, lead=0.0):
        self.wait_until(tr.t(mark) - lead)

    def upto(self, tr, mark, minimum=0.35, lead=0.0):
        """run_time that ends an animation right on a bookmark (never shorter than `minimum`)."""
        return max(minimum, tr.t(mark) - lead - self.now)

    def until_end(self, tr, minimum=0.35, pad=0.0):
        return max(minimum, tr.end + pad - self.now)

    def finish(self, tr, pad=0.3):
        self.wait_until(tr.end + pad)

    def show_term(self, card, run_time=0.5):
        self.play(FadeIn(card, shift=LEFT * 0.25), run_time=run_time)

    def clear_all(self, run_time=0.7, keep_tag=False):
        mobs = [m for m in self.mobjects if not (keep_tag and m is self.tag)]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def tear_down(self):
        # keyed by quality, so preview renders never overwrite the timeline of the final render
        out = BUILD / "timeline" / f"{config.pixel_height}p{int(config.frame_rate)}"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{type(self).__name__}.json").write_text(
            json.dumps({"scene": type(self).__name__, "duration": round(self.now, 4), "segments": self.timeline}))
