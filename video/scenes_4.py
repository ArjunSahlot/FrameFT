"""Chapters 9-12: training and inference, the paper's results, the basis experiment, and the recap."""
import numpy as np
from PIL import Image

from common import *  # noqa: F401,F403
from data import load
from scenes_1 import model_stack
from scenes_3 import image_from_file

D = load()
B = D["B_frame"]
N = 768

# The basis experiment (GLUE/figures/fig_results.png). Accuracies are k/277 on RTE and k/408 on MRPC, so the
# per-seed dots can be read off the figure exactly; every group's mean matches the value printed on it.
RESULTS = {
    "rte_best": {"frame": [218, 220, 222], "random": [217, 220, 225], "identity": [198, 200, 204], "n": 277},
    "rte_final": {"frame": [210, 210, 214], "random": [207, 207, 205], "identity": [194, 195, 191], "n": 277},
    "mrpc_best": {"frame": [366, 365], "random": [362, 361], "identity": [358], "n": 408},
}
ARM_C = {"frame": FRAME_C, "random": RANDOM_C, "identity": IDENT_C}


def small_matrix(label, color, size=0.9, sub=None):
    sq = RoundedRectangle(corner_radius=0.06, width=size, height=size, stroke_color=color, stroke_width=2,
                          fill_color=PANEL_2, fill_opacity=1)
    t = T(label, 22, INK, weight=BOLD).move_to(sq)
    g = VGroup(sq, t)
    if sub:
        g.add(T(sub, 16, MUTED).next_to(sq, DOWN, buff=0.08))
    return g


def dot_panel(key, title, y_range, width=3.7, height=3.0, paper=None):
    """One seed per dot, a short bar at the mean; the three bases side by side."""
    data = RESULTS[key]
    lo, hi = y_range
    ax_h = height
    ax = VGroup()
    base = np.array([0, 0, 0])

    def Y(v):
        return base + UP * (v - lo) / (hi - lo) * ax_h

    for tick in np.arange(np.ceil(lo), hi + 0.01, 2 if hi - lo > 8 else 1):
        ax.add(Line(Y(tick), Y(tick) + RIGHT * width, color=GRID, stroke_width=1))
        ax.add(T(f"{tick:.0f}", 16, MUTED).next_to(Y(tick), LEFT, buff=0.1))
    arms = VGroup()
    for i, arm in enumerate(("frame", "random", "identity")):
        x = RIGHT * width * (i + 0.5) / 3
        vals = [100 * c / data["n"] for c in data[arm]]
        mean = float(np.mean(vals))
        offs = np.linspace(-0.14, 0.14, len(vals)) if len(vals) > 1 else [0]
        dots = VGroup(*[Dot(Y(v) + x + RIGHT * o, radius=0.07, color=ARM_C[arm]) for v, o in zip(vals, offs)])
        bar = Line(Y(mean) + x + LEFT * 0.3, Y(mean) + x + RIGHT * 0.3, color=ARM_C[arm], stroke_width=4)
        num = T(f"{mean:.1f}", 20, INK, weight=BOLD).next_to(bar, RIGHT, buff=0.08)
        name = T(arm, 18, ARM_C[arm]).move_to(Y(lo) + x + DOWN * 0.28)
        nlab = T(f"n={len(vals)}", 14, MUTED).next_to(name, DOWN, buff=0.04)
        arms.add(VGroup(dots, bar, num, name, nlab))
    head = T(title, 22, INK, weight=BOLD).next_to(Y(hi) + RIGHT * width / 2, UP, buff=0.15)
    g = VGroup(ax, arms, head)
    if paper is not None:
        pl = DashedLine(Y(paper), Y(paper) + RIGHT * width, color=INK_2, stroke_width=2, dash_length=0.07)
        pt = T(f"paper\n{paper}", 15, INK_2, line_spacing=0.8).next_to(Y(paper) + RIGHT * width, RIGHT, buff=0.06)
        g.add(VGroup(pl, pt))
    return g


class S09Training(NScene):
    chapter = (9, "Training and inference")

    def construct(self):
        # ---------------------------------------------------------------- training_0: where FrameFT goes
        tr = self.narrate("training_0")
        self.show_tag()
        self.at(tr, "layers", lead=0.3)
        stack = model_stack(width=2.2, bar_h=0.22, gap=0.07).move_to([-5.2, -0.35, 0])
        name = T("RoBERTa-base", 24, INK, weight=BOLD).next_to(stack, UP, buff=0.25)
        self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in stack], lag_ratio=0.05), FadeIn(name), run_time=1.0)
        pick = stack[6]
        hl = SurroundingRectangle(pick, color=TRAIN_C, buff=0.05, stroke_width=2.5)

        att = panel(4.9, 2.2, color=GRID, fill=PANEL).move_to([0.9, 0.85, 0])
        att_t = T("attention", 20, MUTED).next_to(att, UP, buff=0.08).align_to(att, LEFT)
        mats = VGroup(small_matrix("Q", W_C, sub="query"), small_matrix("K", W_C, sub="key"),
                      small_matrix("V", W_C, sub="value"), small_matrix("O", W_C, sub="output"))
        mats.arrange(RIGHT, buff=0.3).move_to(att)
        ffn = panel(3.0, 1.5, color=GRID, fill=PANEL).next_to(att, DOWN, buff=0.45).align_to(att, LEFT)
        ffn_t = T("feed-forward", 20, MUTED).next_to(ffn, UP, buff=0.08).align_to(ffn, LEFT)
        f_mats = VGroup(small_matrix("F1", W_C, 0.8), small_matrix("F2", W_C, 0.8)).arrange(RIGHT, buff=0.3).move_to(ffn)
        z1 = Line(hl.get_corner(UR), att.get_corner(UL), stroke_width=1.2, color=TRAIN_C)
        z2 = Line(hl.get_corner(DR), ffn.get_corner(DL), stroke_width=1.2, color=TRAIN_C)
        lay_t = T("one layer, zoomed in", 20, TRAIN_C).next_to(att, UP, buff=0.08).align_to(att, RIGHT)
        self.play(Create(hl), Create(z1), Create(z2), FadeIn(att), FadeIn(att_t), FadeIn(mats), FadeIn(ffn), FadeIn(ffn_t),
                  FadeIn(f_mats), FadeIn(lay_t), run_time=1.0)

        self.at(tr, "qv", lead=0.2)
        wraps = VGroup()
        for m in (mats[0], mats[2]):
            m[0].set_stroke(FRAME_C, width=3)
            wraps.add(pill("+ΔW", DW_C, 16, pad=0.08).next_to(m[0], UP, buff=0.08))
        self.play(FadeIn(wraps, shift=DOWN * 0.1), Indicate(mats[0], color=FRAME_C), Indicate(mats[2], color=FRAME_C),
                  run_time=0.8)

        self.at(tr, "query", lead=0.2)
        q_tip = T("query → what a word looks for", 22, INK_2)
        q_tip.next_to(ffn, RIGHT, buff=0.45).align_to(ffn, UP)
        self.play(FadeIn(q_tip, shift=DOWN * 0.1), run_time=0.5)
        self.at(tr, "value", lead=0.2)
        v_tip = T("value → what it passes along", 22, INK_2).next_to(q_tip, DOWN, buff=0.2).align_to(q_tip, LEFT)
        q_tip.shift(UP * 0.0)
        tips = VGroup(q_tip, v_tip)
        self.play(FadeIn(v_tip, shift=DOWN * 0.1), run_time=0.5)

        self.at(tr, "frozen", lead=0.2)
        frozen_parts = VGroup(mats[1], mats[3], f_mats)
        ice = VGroup(*[snowflake(0.22).next_to(m[0], UR, buff=-0.12) for m in (mats[1], mats[3], f_mats[0], f_mats[1])])
        self.play(frozen_parts.animate.set_opacity(0.45), FadeIn(ice),
                  VGroup(*[l[1] for l in stack]).animate.set_fill(W_C, opacity=0.35), run_time=0.7)

        self.at(tr, "coeffs", lead=0.2)
        fl = VGroup(*[VGroup(flame(0.26), T("1,000 coefficients", 16, TRAIN_C)).arrange(RIGHT, buff=0.06)
                      .next_to(m[0], DOWN, buff=0.35) for m in (mats[0], mats[2])])
        self.play(FadeIn(fl), run_time=0.6)

        self.at(tr, "head", lead=0.2)
        head = RoundedRectangle(corner_radius=0.06, width=2.2, height=0.36, stroke_color=TRAIN_C, stroke_width=2,
                                fill_color=TRAIN_C, fill_opacity=0.3).next_to(stack, UP, buff=0.15)
        head_t = T("classification head", 16, INK).move_to(head)
        self.play(name.animate.next_to(head, UP, buff=0.18), run_time=0.3)
        self.play(FadeIn(head, shift=DOWN * 0.1), FadeIn(head_t), run_time=0.5)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---------------------------------------------------------------- training_1: forward and backward
        tr = self.narrate("training_1")
        rng = np.random.default_rng(2)
        xv = rng.standard_normal(N)
        x_s = strip(xv, 1.6, 0.32).move_to([-5.5, 1.35, 0])
        x_l = T("x", 22, INK_2).next_to(x_s, UP, buff=0.1)
        w_box = mat_image(D["W"], 1.05, border=W_C, stroke=2).move_to([-2.6, 2.05, 0])
        dw_box = mat_image(D["dW_frame"], 1.05, border=DW_C, stroke=2).move_to([-2.6, 0.55, 0])
        w_l = VGroup(snowflake(0.22), M("W", 30, W_C)).arrange(RIGHT, buff=0.1).next_to(w_box, UP, buff=0.08)
        dw_l = M(r"\Delta W = \tfrac{\text{scale}}{n} B^{\top} S B", 26, DW_C).next_to(dw_box, DOWN, buff=0.1)
        plus = VGroup(Circle(0.26, color=INK, stroke_width=2.5), M("+", 36)).move_to([0.2, 1.35, 0])
        y_s = strip(xv @ D["W"], 1.6, 0.32).move_to([2.1, 1.35, 0])
        y_l = T("y", 22, INK_2).next_to(y_s, UP, buff=0.1)
        a_top = VGroup(Arrow(x_s.get_right(), w_box.get_left(), buff=0.1, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.15),
                       Arrow(w_box.get_right(), plus.get_left(), buff=0.1, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.15))
        a_bot = VGroup(Arrow(x_s.get_right(), dw_box.get_left(), buff=0.1, stroke_width=3, color=DW_C, max_tip_length_to_length_ratio=0.15),
                       Arrow(dw_box.get_right(), plus.get_left(), buff=0.1, stroke_width=3, color=DW_C, max_tip_length_to_length_ratio=0.15))
        a_out = Arrow(plus.get_right(), y_s.get_left(), buff=0.1, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.2)
        eq = M(r"y = xW + x\,\Delta W", 40, INK).move_to([5.2, 1.35, 0])

        self.at(tr, "fwd", lead=0.3)
        self.play(FadeIn(x_s), FadeIn(x_l), FadeIn(w_box), FadeIn(w_l), run_time=0.5)
        self.play(GrowArrow(a_top[0]), GrowArrow(a_top[1]), FadeIn(plus), run_time=0.6)
        self.at(tr, "plus", lead=0.2)
        self.play(FadeIn(dw_box), FadeIn(dw_l), GrowArrow(a_bot[0]), GrowArrow(a_bot[1]), run_time=0.7)
        self.play(GrowArrow(a_out), FadeIn(y_s), FadeIn(y_l), Write(eq), run_time=0.8)

        self.at(tr, "grad", lead=0.2)
        back = CurvedArrow(y_s.get_bottom() + DOWN * 0.1, dw_box.get_right() + DOWN * 0.3 + RIGHT * 0.05, angle=-TAU / 6,
                           color=BAD, stroke_width=3)
        # a gradient with some structure: low rank plus noise
        G = np.outer(rng.standard_normal(N), rng.standard_normal(N)) + 0.6 * np.outer(rng.standard_normal(N), rng.standard_normal(N))
        G += 0.8 * rng.standard_normal((N, N))
        g_img = mat_image(G, 1.5, border=BAD, stroke=2).move_to([-5.3, -1.65, 0])
        g_l = M(r"G = \frac{\partial L}{\partial \Delta W}", 30, BAD).next_to(g_img, UP, buff=0.12)
        self.play(Create(back), run_time=0.6)
        self.play(FadeIn(g_img), FadeIn(g_l), run_time=0.5)
        card = term_card("Gradient", "For every weight, which way it should move (and how much) to reduce the error, "
                                     "written L for loss.")
        card.to_corner(DR, buff=0.4).shift(UP * 1.05)
        self.show_term(card)

        self.at(tr, "gshare", lead=0.2)
        atom = np.outer(B[4], B[7])
        times = M(r"\odot", 40).next_to(g_img, RIGHT, buff=0.25)
        a_img = mat_image(atom, 1.5, border=DW_C, stroke=2).next_to(times, RIGHT, buff=0.25)
        a_l = T("its pattern", 20, DW_C).next_to(a_img, UP, buff=0.12)
        arrow_sum = Arrow(a_img.get_right(), a_img.get_right() + RIGHT * 1.5, buff=0.15, stroke_width=3, color=MUTED)
        sum_t = T("multiply,\nthen sum", 18, INK_2, line_spacing=0.85).next_to(arrow_sum, DOWN, buff=0.08)
        grad_c = M(r"\frac{\partial L}{\partial c_{ij}} = \frac{\text{scale}}{n}\; b_i\, G\, b_j^{\top}", 36, TRAIN_C)
        grad_c.next_to(arrow_sum, RIGHT, buff=0.2)
        self.play(FadeOut(card), FadeIn(times), FadeIn(a_img), FadeIn(a_l), run_time=0.6)
        self.play(GrowArrow(arrow_sum), FadeIn(sum_t), Write(grad_c), run_time=0.9)

        self.at(tr, "lrs", lead=0.2)
        opt = VGroup(T("AdamW, two learning rates (RTE):", 20, INK_2),
                     VGroup(flame(0.24), T("coefficients 0.321", 20, TRAIN_C)).arrange(RIGHT, buff=0.08),
                     VGroup(flame(0.24), T("head 0.006", 20, TRAIN_C)).arrange(RIGHT, buff=0.08)).arrange(RIGHT, buff=0.3)
        opt.move_to([0.6, -2.55, 0])
        self.play(FadeIn(opt, shift=UP * 0.1), run_time=0.6)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---------------------------------------------------------------- training_2: caching and merging
        tr = self.narrate("training_2")
        self.at(tr, "cache", lead=0.3)
        cache = VGroup(panel(4.6, 1.3, color=FRAME_C, fill=PANEL),
                       VGroup(Mono("tffs[(basis, seed, n, l)]", 20, INK), T("master copy, built once (CPU)", 18, MUTED))
                       .arrange(DOWN, buff=0.1))
        cache[1].move_to(cache[0])
        cache.move_to([-3.6, 1.55, 0])
        gpus = VGroup()
        for i in range(2):
            box = VGroup(panel(2.3, 1.0, color=GRID, fill=PANEL_2), VGroup(T(f"GPU {i}", 20, INK, weight=BOLD),
                                                                           T("one copy of B", 16, INK_2)).arrange(DOWN, buff=0.06))
            box[1].move_to(box[0])
            gpus.add(box)
        gpus.arrange(RIGHT, buff=0.6).move_to([-3.6, -0.4, 0])
        c_arrows = VGroup(*[Arrow(cache.get_bottom(), g.get_top(), buff=0.1, stroke_width=3, color=FRAME_C,
                                  max_tip_length_to_length_ratio=0.15) for g in gpus])
        layers = VGroup()
        for i, g in enumerate(gpus):
            for j in range(6):
                sq = Square(0.26, stroke_color=W_C, stroke_width=1.2).move_to(g.get_bottom() + DOWN * 0.8 + RIGHT * (j - 2.5) * 0.36)
                layers.add(VGroup(sq, Line(sq.get_top(), g.get_bottom(), stroke_width=1, color=FRAME_C, stroke_opacity=0.6)))
        l_note = T("every layer on a GPU reads that GPU's copy", 18, MUTED).next_to(layers, DOWN, buff=0.15)
        self.play(FadeIn(cache), run_time=0.5)
        self.play(GrowArrow(c_arrows[0]), GrowArrow(c_arrows[1]), FadeIn(gpus), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(l) for l in layers], lag_ratio=0.04), FadeIn(l_note), run_time=0.8)

        self.at(tr, "merge", lead=0.3)
        mw = mat_image(D["W"], 1.6, border=W_C, stroke=2).move_to([2.2, 0.6, 0])
        mdw = mat_image(D["dW_frame"], 1.6, border=DW_C, stroke=2).move_to([4.6, 0.6, 0])
        mplus = M("+", 44).move_to([3.4, 0.6, 0])
        m_lab = VGroup(M("W", 32, W_C).next_to(mw, UP, buff=0.1), M(r"\Delta W", 32, DW_C).next_to(mdw, UP, buff=0.1))
        self.play(FadeIn(mw), FadeIn(mdw), FadeIn(mplus), FadeIn(m_lab), run_time=0.5)
        merged = mat_image(D["W"] + 0.35 * D["dW_frame"] / np.abs(D["dW_frame"]).max(), 1.6, border=INK, stroke=2)
        merged.move_to([3.4, 0.6, 0])
        code = Mono("model.merge_and_unload()", 20, INK_2).move_to([3.4, -0.75, 0])
        self.play(mdw.animate.move_to(mw), mw.animate.move_to([3.4, 0.6, 0]), FadeOut(mplus), FadeOut(m_lab), run_time=0.7)
        self.remove(mdw, mw)
        self.add(merged)
        new_l = M(r"W' = W + \Delta W", 32, INK).next_to(merged, UP, buff=0.1)
        self.play(FadeIn(new_l), FadeIn(code), run_time=0.5)

        self.at(tr, "fast", lead=0.2)
        fast = VGroup(T("after merging:", 20, MUTED), T("one matrix per layer, like the original", 22, GOOD),
                      T("zero extra inference cost", 22, GOOD, weight=BOLD)).arrange(DOWN, buff=0.1).next_to(code, DOWN, buff=0.3)
        self.play(FadeIn(fast, shift=UP * 0.1), run_time=0.6)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---------------------------------------------------------------- training_3: the parameter count
        tr = self.narrate("training_3")
        self.at(tr, "count", lead=0.3)
        calc = VGroup(T("12 layers", 30, INK), T("×", 30, MUTED), T("2 matrices", 30, INK), T("×", 30, MUTED),
                      T("1,000", 30, TRAIN_C, weight=BOLD), T("=", 30, MUTED), T("24,000", 36, TRAIN_C, weight=BOLD))
        calc.arrange(RIGHT, buff=0.22).move_to([0, 2.0, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in calc], lag_ratio=0.18), run_time=1.3)

        def bar_row(label, value, text, color, i):
            x0 = -2.2
            length = (np.log10(value) - 4) / 4.2 * 8.0 + 0.25
            lab = T(label, 24, INK_2).move_to([x0 - 0.25, 0.55 - i * 0.85, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=length, height=0.46, stroke_width=0, fill_color=color, fill_opacity=0.9)
            bar.move_to([x0 + length / 2, 0.55 - i * 0.85, 0])
            num = T(text, 24, INK, weight=BOLD).next_to(bar, RIGHT, buff=0.15)
            return VGroup(lab, bar, num)

        rows = [bar_row("FrameFT", 24_000, "24 K", TRAIN_C, 0), bar_row("LoRA (rank 8)", 294_912, "≈ 300 K", INK_2, 1),
                bar_row("Full fine-tuning", 125e6, "125 M", W_C, 2)]
        scale_note = VGroup()
        for k_, lab_ in zip(range(4, 9), ("10 K", "100 K", "1 M", "10 M", "100 M")):
            x_ = -2.2 + (k_ - 4) / 4.2 * 8.0 + 0.25
            scale_note.add(DashedLine([x_, 0.95, 0], [x_, -1.5, 0], color=GRID, stroke_width=1.2, dash_length=0.06))
            scale_note.add(T(lab_, 16, MUTED).move_to([x_, -1.72, 0]))
        scale_note.set_z_index(-1)
        self.play(FadeIn(rows[0][0]), GrowFromEdge(rows[0][1], LEFT), FadeIn(rows[0][2]), FadeIn(scale_note), run_time=0.7)
        self.at(tr, "lora", lead=0.2)
        self.play(FadeIn(rows[1][0]), GrowFromEdge(rows[1][1], LEFT), FadeIn(rows[1][2]), run_time=0.7)
        self.at(tr, "ff", lead=0.2)
        self.play(FadeIn(rows[2][0]), GrowFromEdge(rows[2][1], LEFT), FadeIn(rows[2][2]), run_time=0.8)

        self.at(tr, "kb", lead=0.3)
        store = VGroup(T("24,000 × 4 bytes ≈ 96 KB", 28, TRAIN_C, weight=BOLD), T("vs. ≈ 500 MB for a full copy", 24, INK_2))
        store.arrange(RIGHT, buff=0.4).move_to([0, -2.45, 0])
        self.play(FadeOut(scale_note), FadeIn(store, shift=UP * 0.1), run_time=0.6)
        self.finish(tr, pad=0.8)
        self.clear_all()


class S10Results(NScene):
    chapter = (10, "Results from the paper")

    def construct(self):
        tr = self.narrate("results_0")
        self.show_tag()
        self.at(tr, "glue", lead=0.3)
        tasks = [("SST-2", "positive or negative?"), ("MRPC", "same meaning?"), ("CoLA", "grammatical?"),
                 ("QNLI", "answers the question?"), ("RTE", "does A imply B?"), ("STS-B", "how similar?")]
        chips = VGroup()
        for name, q in tasks:
            chips.add(VGroup(T(name, 22, INK, weight=BOLD), T(q, 18, INK_2)).arrange(DOWN, buff=0.06))
        chips.arrange_in_grid(rows=2, cols=3, buff=(0.9, 0.22)).move_to([0, 2.2, 0])
        chips.shift(DOWN * 0.15)
        title = T("GLUE: six language-understanding tasks", 20, MUTED).next_to(chips, UP, buff=0.2)
        self.play(FadeIn(title), LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in chips], lag_ratio=0.1), run_time=1.0)

        self.at(tr, "avg", lead=0.3)
        methods = [("Full fine-tuning", 85.2, "125 M", W_C), ("LoRA", 85.2, "0.3 M", INK_2), ("FourierFT", 85.0, "24 K", INK_2),
                   ("FrameFT", 86.1, "24 K", TRAIN_C)]
        lo, hi, width = 83.0, 87.0, 7.0
        x0 = -2.6
        axis = VGroup()
        for t_ in (83, 84, 85, 86, 87):
            x = x0 + (t_ - lo) / (hi - lo) * width
            axis.add(Line([x, 0.95, 0], [x, -2.25, 0], color=GRID, stroke_width=1))
            axis.add(T(str(t_), 16, MUTED).move_to([x, -2.45, 0]))
        axis_t = T("GLUE average, RoBERTa-base (axis starts at 83)", 18, MUTED).move_to([x0 + width / 2, 1.2, 0])
        bars = VGroup()
        for i, (m, v, p, col) in enumerate(methods):
            y = 0.55 - i * 0.78
            lab = T(m, 24, INK if m == "FrameFT" else INK_2, weight=BOLD if m == "FrameFT" else NORMAL)
            lab.move_to([x0 - 0.25, y, 0], aligned_edge=RIGHT)
            L = (v - lo) / (hi - lo) * width
            bar = Rectangle(width=L, height=0.44, stroke_width=0, fill_color=col, fill_opacity=0.9).move_to([x0 + L / 2, y, 0])
            num = T(f"{v}", 22, INK, weight=BOLD).next_to(bar, RIGHT, buff=0.12)
            par = T(p + " trained", 18, TRAIN_C if m == "FrameFT" else MUTED).next_to(num, RIGHT, buff=0.3)
            bars.add(VGroup(lab, bar, num, par))
        self.play(FadeIn(axis), FadeIn(axis_t), run_time=0.4)
        self.play(LaggedStart(*[AnimationGroup(FadeIn(b[0]), GrowFromEdge(b[1], LEFT), FadeIn(b[2])) for b in bars],
                              lag_ratio=0.2), run_time=1.4)
        self.at(tr, "params", lead=0.2)
        self.play(LaggedStart(*[FadeIn(b[3], shift=LEFT * 0.1) for b in bars], lag_ratio=0.15), run_time=0.8)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        tr = self.narrate("results_1")

        def table(title, rows, pos):
            head = T(title, 22, INK, weight=BOLD)
            g = VGroup()
            for i, (m, v, p) in enumerate(rows):
                hl = m == "FrameFT"
                r = VGroup(T(m, 20, TRAIN_C if hl else INK_2, weight=BOLD if hl else NORMAL),
                           T(v, 20, INK, weight=BOLD if hl else NORMAL), T(p, 18, TRAIN_C if hl else MUTED))
                for j, c in enumerate(r):
                    c.move_to([j * 1.75, -i * 0.46, 0], aligned_edge=LEFT)
                g.add(r)
            hdr = VGroup(T("method", 16, MUTED), T("avg", 16, MUTED), T("trained", 16, MUTED))
            for j, c in enumerate(hdr):
                c.move_to([j * 1.75, 0.46, 0], aligned_edge=LEFT)
            body = VGroup(hdr, g)
            out = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            box = panel(out.width + 0.5, out.height + 0.45, color=GRID, fill=PANEL).move_to(out)
            return VGroup(box, out).move_to(pos)

        llama = table("Llama-2-7B, instruction-tuned (8 tasks)",
                      [("Full FT", "63.39", "6.7 B"), ("LoRA", "63.18", "16.7 M"), ("FourierFT", "63.22", "320 K"),
                       ("FrameFT", "63.62", "320 K")], [-3.35, 0.2, 0])
        vit = table("ViT-L, image classification (8 tasks)",
                    [("Full FT", "90.20", "303 M"), ("LoRA", "84.94", "1.57 M"), ("FourierFT", "86.68", "480 K"),
                     ("FrameFT", "87.95", "240 K")], [3.35, 0.2, 0])
        self.at(tr, "llama", lead=0.2)
        self.play(FadeIn(llama, shift=UP * 0.15), run_time=0.7)
        self.at(tr, "vit", lead=0.2)
        self.play(FadeIn(vit, shift=UP * 0.15), run_time=0.7)
        src = T("numbers from the FrameFT paper (README tables)", 16, MUTED).move_to([0, -2.4, 0])
        self.play(FadeIn(src), run_time=0.4)
        self.finish(tr, pad=0.7)
        self.clear_all()


class S11Experiment(NScene):
    chapter = (11, "Our experiment: does the frame matter?")

    def construct(self):
        size = 2.2
        xs = [-4.3, 0, 4.3]
        y_img = 0.75

        # ---------------------------------------------------------------- experiment_0: the question
        tr = self.narrate("experiment_0")
        self.show_tag()
        self.at(tr, "orth", lead=0.2)
        imgs = {
            "frame": mat_image(D["B_frame"], size, border=FRAME_C, stroke=2.5),
            "random": mat_image(D["B_random"], size, border=RANDOM_C, stroke=2.5),
            "identity": mat_image(D["B_identity"], size, border=IDENT_C, stroke=2.5, dilate=2, gamma=1),
        }
        for (k, im), x in zip(imgs.items(), xs):
            im.move_to([x, y_img, 0])
        names = {k: T(k, 28, ARM_C[k], weight=BOLD).next_to(imgs[k], UP, buff=0.18) for k in imgs}
        orth_note = T("an orthogonal matrix, like every basis here", 20, INK_2).next_to(imgs["frame"], DOWN, buff=0.18)
        self.play(FadeIn(imgs["frame"]), FadeIn(names["frame"]), FadeIn(orth_note), run_time=0.7)
        self.at(tr, "any", lead=0.2)
        qs = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.1, width=size, height=size, stroke_color=MUTED,
                                              stroke_width=2).set_stroke(opacity=0.8),
                             T("?", 60, MUTED)).move_to([x, y_img, 0]) for x in xs[1:]])
        for q in qs:
            q[1].move_to(q[0])
        question = T("wave structure, or just any orthogonal basis?", 26, INK).move_to([0, -1.55, 0])
        self.play(FadeIn(qs), FadeIn(question), run_time=0.7)
        self.finish(tr)

        # ---------------------------------------------------------------- experiment_1: the two alternatives
        tr = self.narrate("experiment_1")
        self.at(tr, "random", lead=0.2)
        self.play(FadeOut(question), FadeOut(orth_note), FadeOut(qs[0]), FadeIn(imgs["random"]), FadeIn(names["random"]),
                  run_time=0.6)
        r_note = T("uniformly random rotation", 20, INK_2).next_to(imgs["random"], DOWN, buff=0.18)
        self.play(FadeIn(r_note), run_time=0.4)
        self.at(tr, "qr", lead=0.2)
        rng = np.random.default_rng(1)
        noise = mat_image(rng.standard_normal((64, 64)), 0.8, border=GRID).move_to([-1.2, -1.7, 0])
        qr_arrow = Arrow(noise.get_right(), noise.get_right() + RIGHT * 1.3, buff=0.1, stroke_width=3, color=MUTED)
        qr_t = T("QR", 20, INK, weight=BOLD).next_to(qr_arrow, UP, buff=0.04)
        q_small = mat_image(D["B_random"][:64, :64], 0.8, border=RANDOM_C).next_to(qr_arrow, RIGHT, buff=0.1)
        qr_note = T("random numbers → straightened into\nperpendicular, unit-length rows", 18, INK_2, line_spacing=0.85)
        qr_note.next_to(q_small, RIGHT, buff=0.25)
        self.play(FadeIn(noise), GrowArrow(qr_arrow), FadeIn(qr_t), FadeIn(q_small), FadeIn(qr_note), run_time=0.8)
        card = term_card("QR decomposition", "Splits any square matrix into Q, whose rows are perpendicular and of length "
                                             "one, times a triangular R. Q is the part we keep.", chars=46)
        card.to_corner(DL, buff=0.4).shift(UP * 1.05)
        self.show_term(card)

        self.at(tr, "identity", lead=0.2)
        self.play(FadeOut(card), FadeOut(qs[1]), FadeIn(imgs["identity"]), FadeIn(names["identity"]), run_time=0.6)
        i_note = T("the plainest basis", 20, INK_2).next_to(imgs["identity"], DOWN, buff=0.18)
        self.play(FadeIn(i_note), run_time=0.4)

        self.at(tr, "one", lead=0.2)
        self.play(FadeOut(noise), FadeOut(qr_arrow), FadeOut(qr_t), FadeOut(q_small), FadeOut(qr_note), FadeOut(r_note),
                  FadeOut(i_note), run_time=0.4)
        atoms = {}
        a_size = 1.5
        for k, x in zip(("frame", "random", "identity"), xs):
            Bk = D[f"B_{k}"]
            A = np.outer(Bk[4], Bk[7]) if k != "identity" else np.outer(Bk[300], Bk[520])
            atoms[k] = mat_image(A, a_size, border=ARM_C[k], stroke=2, dilate=6 if k == "identity" else 0,
                                 gamma=1 if k == "identity" else 0.75).move_to([x, -1.2, 0])
        a_labels = VGroup()
        a_desc = {"frame": "a wave, everywhere", "random": "static, everywhere", "identity": "a single weight"}
        a_notes = VGroup(*[T(a_desc[k], 18, ARM_C[k]).next_to(atoms[k], DOWN, buff=0.12) for k in atoms])
        a_labels.add(T("one\ncoefficient's\npattern:", 18, MUTED, line_spacing=0.85).next_to(atoms["frame"], LEFT, buff=0.3))
        self.play(*[FadeIn(a) for a in atoms.values()], FadeIn(a_labels), FadeIn(a_notes), run_time=0.8)
        self.finish(tr)

        # ---------------------------------------------------------------- experiment_2: fairness
        tr = self.narrate("experiment_2")
        self.at(tr, "same", lead=0.3)
        self.play(*[FadeOut(m) for m in (*atoms.values(), a_labels, a_notes)],
                  *[im.animate.scale(0.6).move_to([x, 1.7, 0]) for im, x in zip(imgs.values(), xs)],
                  *[FadeOut(n) for n in names.values()], run_time=0.6)
        small_names = VGroup(*[T(k, 20, ARM_C[k], weight=BOLD).next_to(imgs[k], LEFT, buff=0.15) for k in imgs])
        self.play(FadeIn(small_names), run_time=0.3)
        checks = VGroup(*[VGroup(T("✓", 22, GOOD, weight=BOLD), T(t_, 20, INK)).arrange(RIGHT, buff=0.15) for t_ in (
            "same 1,000 positions (entry_seed 2024)", "same seeds and data order",
            "same learning rates, scale, epochs", "same number of trainable numbers")])
        checks.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        checks.move_to([-6.4 + checks.width / 2, -0.6, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.1) for c in checks], lag_ratio=0.2), run_time=1.1)

        self.at(tr, "sv", lead=0.3)
        ax = Axes(x_range=[0, 520, 100], y_range=[0, 4, 1], x_length=4.2, y_length=2.1,
                  axis_config={"color": MUTED, "stroke_width": 1.5, "include_ticks": False}).move_to([3.7, -0.6, 0])
        curves = VGroup()
        for k, w, dash in (("frame", 7, False), ("random", 4.5, False), ("identity", 2.5, True)):
            sv = D[f"sv_{k}"][:500]
            pts = [ax.c2p(i, v) for i, v in enumerate(sv)]
            c = VMobject(stroke_color=ARM_C[k], stroke_width=w).set_points_as_corners(pts)
            curves.add(DashedVMobject(c, num_dashes=40) if dash else c)
        ax_l = T("singular values of ΔW, largest first", 18, INK_2).next_to(ax, UP, buff=0.1)
        same_l = T("three curves, exactly on top of each other", 18, GOOD).next_to(ax, DOWN, buff=0.12)
        self.play(Create(ax), FadeIn(ax_l), run_time=0.5)
        self.play(*[Create(c) for c in curves], run_time=1.2)
        self.play(FadeIn(same_l), run_time=0.4)
        card = term_card("Singular values", "How strongly a matrix stretches space along each of its main directions, "
                                            "sorted from largest to smallest.")
        card.to_corner(DL, buff=0.4).shift(UP * 1.05)
        self.play(FadeOut(checks), run_time=0.3)
        self.show_term(card)

        self.at(tr, "orient", lead=0.2)
        orient = VGroup(T("same size, rank and spectrum;", 24, INK),
                        T("only the orientation differs", 24, TRAIN_C, weight=BOLD)).arrange(DOWN, buff=0.1).move_to([-3.3, -0.6, 0])
        self.play(FadeOut(card), FadeIn(orient, shift=UP * 0.1), run_time=0.6)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---------------------------------------------------------------- experiment_3: results
        tr = self.narrate("experiment_3")
        self.at(tr, "rte", lead=0.3)
        p1 = dot_panel("rte_best", "RTE · best epoch", (68, 84), width=3.4, paper=79.8).move_to([-4.7, 0.25, 0])
        self.play(FadeIn(p1[0]), FadeIn(p1[2]), LaggedStart(*[FadeIn(a) for a in p1[1]], lag_ratio=0.2), run_time=1.2)
        card = term_card("Epoch", "One full pass through the training data. “Best epoch” is the best score of "
                                  "all passes; “final” is the last one.")
        card.to_corner(DR, buff=0.4).shift(UP * 1.05)
        self.show_term(card)

        self.at(tr, "neck", lead=0.2)
        ring = VGroup(SurroundingRectangle(VGroup(p1[1][0][1], p1[1][0][2]), color=INK, buff=0.1),
                      SurroundingRectangle(VGroup(p1[1][1][1], p1[1][1][2]), color=INK, buff=0.1))
        self.play(Create(ring), run_time=0.5)
        self.at(tr, "paperline", lead=0.2)
        self.play(FadeOut(ring), Create(p1[3]), run_time=0.6)
        self.at(tr, "idgap", lead=0.2)
        gap = DoubleArrow(p1[1][1][1].get_center() + RIGHT * 0.9, [p1[1][1][1].get_center()[0] + 0.9,
                                                                    p1[1][2][1].get_center()[1], 0],
                          buff=0, stroke_width=3, color=BAD, max_tip_length_to_length_ratio=0.08)
        gap_t = T("−7 pts", 20, BAD, weight=BOLD).next_to(gap, RIGHT, buff=0.08)
        self.play(GrowFromCenter(gap), FadeIn(gap_t), run_time=0.6)

        self.at(tr, "final", lead=0.2)
        p2 = dot_panel("rte_final", "RTE · final epoch", (68, 84), width=3.4).move_to([-0.1, 0.25, 0])
        self.play(FadeOut(card), FadeIn(p2[0]), FadeIn(p2[2]), LaggedStart(*[FadeIn(a) for a in p2[1]], lag_ratio=0.2),
                  run_time=1.0)
        self.at(tr, "mrpc", lead=0.2)
        p3 = dot_panel("mrpc_best", "MRPC · best epoch", (86, 91), width=3.4).move_to([4.5, 0.25, 0])
        self.play(FadeIn(p3[0]), FadeIn(p3[2]), LaggedStart(*[FadeIn(a) for a in p3[1]], lag_ratio=0.2), run_time=1.0)
        src = T("from GLUE/figures/fig_results.png · RoBERTa-base · 1,000 coefficients per matrix", 16, MUTED)
        src.move_to([0, -2.55, 0])
        self.play(FadeIn(src), run_time=0.3)

        self.at(tr, "loss", lead=0.3)
        crop = Image.open(HERE.parent / "GLUE" / "figures" / "fig_curves.png").convert("RGBA").crop((0, 780, 1000, 1440))
        crop_path = BUILD / "loss_crop.png"
        crop.save(crop_path)
        loss = image_from_file(crop_path, 3.5)
        loss_bg = panel(loss.width + 0.3, loss.height + 0.3, color=GRID, fill="#FCFCFB").move_to(loss)
        loss_g = Group(loss_bg, loss).move_to([1.3, 0.35, 0])
        legend = VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.4, color=ARM_C[k], stroke_width=4), T(k, 18, ARM_C[k]))
                          .arrange(RIGHT, buff=0.1) for k in ARM_C]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.next_to(loss_g, RIGHT, buff=0.25)
        loss_note = T("similar training loss: all three fit the training data", 20, INK_2)
        loss_note.next_to(loss_g, DOWN, buff=0.18)
        self.play(FadeOut(p2), FadeOut(p3), FadeOut(gap), FadeOut(gap_t), FadeOut(src),
                  p1.animate.scale(0.85).move_to([-4.9, 0.3, 0]), run_time=0.6)
        self.play(FadeIn(loss_g, shift=UP * 0.15), FadeIn(legend), run_time=0.7)
        self.play(FadeIn(loss_note), run_time=0.4)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---------------------------------------------------------------- experiment_4: why, and how sure
        tr = self.narrate("experiment_4")
        self.at(tr, "quarter", lead=0.3)
        S = D["S_768"]
        empty = np.abs(S).sum(axis=1) == 0
        from scipy.ndimage import maximum_filter
        pix = colorize(np.zeros_like(S), vmax=1, cmap="mono", gamma=1).astype(float)
        pix[empty] = pix[empty] * 0.4 + np.array(ManimColor(BAD).to_rgb()) * 255 * 0.45  # rows no coefficient touches
        dots = maximum_filter((np.abs(S) > 0).astype(float), size=3) > 0
        pix[dots] = np.array(ManimColor(TRAIN_C).to_rgb()) * 255
        img = Image.fromarray(pix.astype(np.uint8)).resize((round(3.2 * PX_PER_UNIT),) * 2, Image.BOX)
        s_img = ImageMobject(np.array(img)).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        s_img.stretch_to_fit_width(3.2).stretch_to_fit_height(3.2).move_to([-4.75, 0.25, 0])
        s_border = Rectangle(width=3.2, height=3.2, stroke_color=IDENT_C, stroke_width=2).move_to(s_img)
        s_lab = T("identity: ΔW = S", 22, IDENT_C, weight=BOLD).next_to(s_border, UP, buff=0.15)
        frac = empty.mean()
        q_lab = VGroup(T(f"{100 * frac:.0f}% of rows are empty", 22, BAD, weight=BOLD),
                       T("(inputs that never affect ΔW;", 18, INK_2), T("block size 768, as for MRPC)", 18, INK_2))
        q_lab.arrange(DOWN, buff=0.05)
        q_lab.next_to(s_border, DOWN, buff=0.15)
        self.play(FadeIn(s_img), Create(s_border), FadeIn(s_lab), run_time=0.7)
        self.play(FadeIn(q_lab), run_time=0.5)

        self.at(tr, "dense", lead=0.2)
        d_img = mat_image(D["dW_frame"], 3.2, border=FRAME_C, stroke=2).move_to([-1.0, 0.25, 0])
        d_lab = T("frame: ΔW = Bᵀ S B", 22, FRAME_C, weight=BOLD).next_to(d_img, UP, buff=0.15)
        d_note = VGroup(T("100% of entries touched", 22, GOOD, weight=BOLD),
                        T("(each coefficient reaches all)", 18, INK_2)).arrange(DOWN, buff=0.06)
        d_note.next_to(d_img, DOWN, buff=0.15)
        self.play(FadeIn(d_img), FadeIn(d_lab), FadeIn(d_note), run_time=0.7)

        self.at(tr, "beat", lead=0.3)
        verdict = VGroup(
            VGroup(T("✓", 26, GOOD, weight=BOLD), T("dense beats local: clear", 22, INK)).arrange(RIGHT, buff=0.15),
            VGroup(T("?", 26, TRAIN_C, weight=BOLD), T("frame vs. random: small,\nneeds more seeds", 22, INK, line_spacing=0.9))
            .arrange(RIGHT, buff=0.2, aligned_edge=UP)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        verdict.move_to([1.35 + verdict.width / 2, 0.55, 0])
        self.play(FadeIn(verdict[0], shift=LEFT * 0.1), run_time=0.5)
        self.play(FadeIn(verdict[1], shift=LEFT * 0.1), run_time=0.5)

        self.at(tr, "oneex", lead=0.2)
        one = VGroup(T("RTE validation set:", 18, MUTED), T("277 examples", 22, INK, weight=BOLD),
                     T("1 example = 0.36 points", 22, TRAIN_C, weight=BOLD)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        one.next_to(verdict, DOWN, buff=0.45, aligned_edge=LEFT)
        card = term_card("Validation set", "Held-out examples the model never trains on, used to score it.")
        self.play(FadeIn(one, shift=UP * 0.1), run_time=0.5)
        self.show_term(card)
        self.finish(tr, pad=0.9)
        self.clear_all()


class S12Recap(NScene):
    chapter = (12, "Recap")

    def construct(self):
        tr = self.narrate("recap_0")
        self.show_tag()

        def box(text, color, width=None, size=20, sub=None):
            t = T(text, size, INK, weight=BOLD)
            parts = [t] + ([T(sub, 15, INK_2)] if sub else [])
            content = VGroup(*parts).arrange(DOWN, buff=0.05)
            w = width or content.width + 0.4
            r = RoundedRectangle(corner_radius=0.12, width=w, height=content.height + 0.32, stroke_color=color,
                                 stroke_width=2, fill_color=PANEL, fill_opacity=1)
            return VGroup(r, content.move_to(r))

        def arrow(a, b, color=MUTED):
            return Arrow(a.get_right(), b.get_left(), buff=0.08, stroke_width=3, color=color, max_tip_length_to_length_ratio=0.2)

        row1 = [box("n, l", INK_2), box("Spectral Tetris", FRAME_C, sub="seed matrix"), box("Modulation", FRAME_C, sub="copies at k speeds"),
                box("Real 2×2 blocks", FRAME_C, sub="complex → real"), box("B", FRAME_C, sub="frozen · cached · shared")]
        r1 = VGroup(*row1).arrange(RIGHT, buff=0.45).move_to([0, 1.9, 0])
        row2 = [box("seed 2024", INK_2), box("1,000 positions", TRAIN_C, sub="randperm"), box("S", TRAIN_C, sub="trainable coefficients")]
        r2 = VGroup(*row2).arrange(RIGHT, buff=0.45)
        r2.shift([0.45, 0, 0] - r2.get_center() + [0, 0, 0])
        r2.set_y(0.45)
        r2.shift(RIGHT * (row1[4].get_left()[0] - 0.35 - row2[2].get_right()[0]))
        eqb = VGroup(RoundedRectangle(corner_radius=0.12, width=4.4, height=1.0, stroke_color=DW_C, stroke_width=2.5,
                                      fill_color=PANEL, fill_opacity=1),
                     M(r"\Delta W = \frac{\text{scale}}{n}\, B^{\top} S\, B", 34, INK))
        eqb[1].move_to(eqb[0])
        eqb.move_to([(row2[2].get_center()[0] + row1[4].get_center()[0]) / 2, -1.25, 0])
        addb = box("W + ΔW", INK_2, sub="frozen W + update").next_to(eqb, LEFT, buff=0.7)
        mergeb = box("merge", GOOD, sub="W' = W + ΔW, no extra cost").next_to(addb, LEFT, buff=0.7)
        motto = T("measure  →  edit  →  rebuild", 20, INK_2).next_to(eqb, DOWN, buff=0.15)

        self.at(tr, "build", lead=0.3)
        a1 = [arrow(row1[i], row1[i + 1]) for i in range(4)]
        self.play(FadeIn(row1[0]), run_time=0.3)
        for i in range(4):
            self.play(GrowArrow(a1[i]), FadeIn(row1[i + 1], shift=RIGHT * 0.1), run_time=0.45)
        self.at(tr, "S", lead=0.3)
        a2 = [arrow(row2[i], row2[i + 1]) for i in range(2)]
        self.play(FadeIn(row2[0]), run_time=0.3)
        for i in range(2):
            self.play(GrowArrow(a2[i]), FadeIn(row2[i + 1], shift=RIGHT * 0.1), run_time=0.45)
        self.at(tr, "eq", lead=0.3)
        fromB = Arrow(row1[4].get_bottom(), [row1[4].get_center()[0], eqb.get_top()[1], 0], buff=0.1, stroke_width=3,
                      color=FRAME_C)
        fromS = Arrow(row2[2].get_bottom(), [row2[2].get_center()[0], eqb.get_top()[1], 0], buff=0.1, stroke_width=3,
                      color=TRAIN_C)
        self.play(GrowArrow(fromB), GrowArrow(fromS), FadeIn(eqb), run_time=0.8)
        self.play(FadeIn(motto), run_time=0.4)
        self.at(tr, "add", lead=0.2)
        a3 = Arrow(eqb.get_left(), addb.get_right(), buff=0.08, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.2)
        self.play(GrowArrow(a3), FadeIn(addb), run_time=0.5)
        self.at(tr, "merge", lead=0.2)
        a4 = Arrow(addb.get_left(), mergeb.get_right(), buff=0.08, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.2)
        self.play(GrowArrow(a4), FadeIn(mergeb), run_time=0.5)
        self.finish(tr)
        self.clear_all()

        tr = self.narrate("recap_1")
        self.at(tr, "end", lead=0.1)
        backdrop = mat_image(D["B_frame"][:432], 8.0, width=14.3, border=None, gamma=0.6, dim=0.13)
        line1 = T("24,000 numbers  ·  a couple of seeds  ·  one formula", 34, INK, weight=BOLD)
        self.play(FadeIn(backdrop), FadeIn(line1, shift=UP * 0.1), run_time=0.9)
        self.at(tr, "thanks", lead=0.2)
        title = T("FrameFT", 72, INK, weight=BOLD)
        cite = T("Adepu, Zhang, Kumar, Singh · Fine-Tuning of Transformer models with Frames · ICML 2026", 20, INK_2)
        thanks = T("Thanks for watching", 30, TRAIN_C)
        end = VGroup(title, cite, thanks).arrange(DOWN, buff=0.3).shift(DOWN * 0.3)
        self.play(line1.animate.shift(UP * 1.5).set_color(INK_2), FadeIn(end, shift=UP * 0.15), run_time=0.9)
        self.finish(tr, pad=2.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.2)
