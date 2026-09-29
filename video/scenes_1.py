"""Chapters 1-3: the hook, the fine-tuning problem, and the PEFT landscape."""
import numpy as np
from scipy.fft import dctn, idctn

from common import *  # noqa: F401,F403
from data import load

D = load()


def model_stack(n_layers=12, width=3.0, bar_h=0.26, gap=0.085):
    """RoBERTa as a column of layer bars, each holding six little weight matrices."""
    layers = VGroup()
    for i in range(n_layers):
        bar = RoundedRectangle(corner_radius=0.06, width=width, height=bar_h, stroke_color=GRID,
                               stroke_width=1.2, fill_color=PANEL_2, fill_opacity=1)
        mats = VGroup(*[Square(bar_h * 0.62, stroke_width=0, fill_color=W_C, fill_opacity=0.9)
                        for _ in range(6)]).arrange(RIGHT, buff=0.16).move_to(bar)
        layers.add(VGroup(bar, mats))
    layers.arrange(UP, buff=gap)
    return layers


class S01Intro(NScene):
    chapter = (1, "FrameFT, end to end")

    def construct(self):
        # ---- silent title card over a faint picture of the real frame matrix
        backdrop = mat_image(D["B_frame"][:432], 8.0, width=14.3, border=None, gamma=0.6, dim=0.16)
        title = T("FrameFT", 120, INK, weight=BOLD)
        sub = T("Fine-tuning Transformers with Frames", 38, INK_2)
        note = T("the method, the math, and the code, step by step", 26, MUTED)
        card = VGroup(title, sub, note).arrange(DOWN, buff=0.32).shift(UP * 0.2)
        self.play(FadeIn(backdrop), run_time=1.2)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.9)
        self.play(FadeIn(sub, shift=UP * 0.1), FadeIn(note), run_time=0.8)
        self.wait(1.6)
        self.play(FadeOut(card), FadeOut(backdrop), run_time=0.9)

        # ---- intro_0: 125M numbers -> 24K
        tr = self.narrate("intro_0")
        self.show_tag()
        stack = model_stack().to_edge(LEFT, buff=1.3).shift(DOWN * 0.35)
        name = T("RoBERTa-base", 30, INK, weight=BOLD).next_to(stack, UP, buff=0.3)
        sub = T("12 layers", 22, MUTED).next_to(name, DOWN, buff=0.08)
        name.shift(UP * 0.18)
        sub.next_to(name, DOWN, buff=0.08)
        self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in stack], lag_ratio=0.06),
                  FadeIn(name), FadeIn(sub), run_time=self.upto(tr, "count", minimum=1.0))

        right_x = 2.8
        value = ValueTracker(0)
        counter_color = {"c": INK}
        counter = always_redraw(lambda: T(f"{value.get_value():,.0f}", 84, counter_color["c"], weight=BOLD).move_to([right_x, 2.05, 0]))
        counter_label = T("learned numbers", 28, INK_2).move_to([right_x, 1.2, 0])
        self.add(counter)
        self.play(value.animate.set_value(125_000_000), FadeIn(counter_label), run_time=2.0,
                  rate_func=rate_functions.ease_out_cubic)

        self.at(tr, "all")
        mats_all = VGroup(*[l[1] for l in stack])
        full_note = VGroup(flame(0.42), T("Full fine-tuning: train all of them", 30, TRAIN_C)).arrange(RIGHT, buff=0.2)
        full_note.move_to([right_x, -0.2, 0])
        self.play(mats_all.animate.set_fill(TRAIN_C), FadeIn(full_note, shift=UP * 0.1), run_time=0.9)

        self.at(tr, "few", lead=0.25)
        # FrameFT: everything frozen except a few coefficients in query and value (matrices 0 and 2)
        dots = VGroup()
        for l in stack:
            for k in (0, 2):
                dots.add(Dot(l[1][k].get_center(), radius=0.045, color=TRAIN_C))
        frame_note = VGroup(snowflake(0.4), T("FrameFT: freeze them, train", 30, INK_2),
                            T("24,000", 30, TRAIN_C, weight=BOLD)).arrange(RIGHT, buff=0.18)
        frame_note.move_to(full_note)
        counter_color["c"] = TRAIN_C
        new_label = T("trainable numbers", 28, TRAIN_C).move_to(counter_label)
        self.play(mats_all.animate.set_fill(W_C, opacity=0.35),
                  value.animate(rate_func=rate_functions.ease_in_out_cubic).set_value(24_000),
                  FadeTransform(counter_label, new_label),
                  FadeTransform(full_note, frame_note), run_time=1.5)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.02), run_time=0.8)

        self.at(tr, "pct", lead=0.2)
        side = 2.5
        px = 4  # each of the 100 x 100 cells is 3 px with a 1 px gutter
        bg_rgb = (np.array(ManimColor(BG).to_rgb()) * 255).astype(np.uint8)
        cell_rgb = (np.array(ManimColor(PANEL_2).to_rgb()) * 255 + 18).clip(0, 255).astype(np.uint8)
        hot_rgb = (np.array(ManimColor(TRAIN_C).to_rgb()) * 255).astype(np.uint8)
        pix = np.zeros((100 * px, 100 * px, 3), dtype=np.uint8)
        pix[:] = cell_rgb
        pix[px - 1::px, :] = bg_rgb  # gutters between rows
        pix[:, px - 1::px] = bg_rgb  # and between columns
        hot = ((37, 61), (72, 18))
        for r, c in hot:
            pix[r * px:r * px + px - 1, c * px:c * px + px - 1] = hot_rgb
        grid = ImageMobject(pix).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        grid.stretch_to_fit_width(side).stretch_to_fit_height(side).move_to([right_x - 1.75, -1.6, 0])
        lit = VGroup(*[Circle(0.15, color=TRAIN_C, stroke_width=3).move_to(
            grid.get_corner(UL) + np.array([side * (c + 0.4) / 100, -side * (r + 0.4) / 100, 0])) for r, c in hot])
        pct = T("0.02%", 64, TRAIN_C, weight=BOLD).move_to([right_x + 1.55, -1.05, 0])
        pct_note = T(wrap("If the model were 10,000 squares, FrameFT would train 2 of them.", 24), 22, INK_2)
        pct_note.next_to(pct, DOWN, buff=0.2)
        self.play(FadeIn(grid), FadeIn(pct, scale=1.1), run_time=0.8)
        self.play(Create(lit), FadeIn(pct_note), run_time=0.7)

        self.at(tr, "match", lead=0.3)
        bars_title = T("GLUE average, RoBERTa-base", 22, MUTED)
        rows = VGroup()
        for label, score, color in (("Full fine-tuning", 85.2, W_C), ("FrameFT", 86.1, TRAIN_C)):
            lab = T(label, 24, INK_2)
            bar = Rectangle(width=(score - 80) * 0.42, height=0.34, stroke_width=0, fill_color=color, fill_opacity=1)
            num = T(f"{score}", 24, INK, weight=BOLD)
            rows.add(VGroup(lab, bar, num))
        for r in rows:
            r[0].set_width(r[0].width)
        block = VGroup()
        for i, r in enumerate(rows):
            r[0].move_to([0, -i * 0.55, 0], aligned_edge=RIGHT)
            r[1].next_to(r[0], RIGHT, buff=0.2)
            r[2].next_to(r[1], RIGHT, buff=0.15)
            block.add(r)
        axis_note = T("bars start at 80", 18, MUTED)
        chart = VGroup(bars_title, block, axis_note).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        chart.move_to([right_x, -1.7, 0])
        self.play(FadeOut(grid), FadeOut(lit), FadeOut(pct), FadeOut(pct_note), run_time=0.5)
        self.play(FadeIn(bars_title), *[GrowFromEdge(r[1], LEFT) for r in block],
                  *[FadeIn(r[0]) for r in block], FadeIn(axis_note), run_time=0.9)
        self.play(*[FadeIn(r[2]) for r in block], run_time=0.4)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---- intro_1: the plan
        tr = self.narrate("intro_1")
        head = T("The plan", 40, INK, weight=BOLD).move_to([-3.2, 2.2, 0])
        items = [
            ("The problem: fine-tuning with few numbers", "c1"),
            ("What a frame is: basis → frame → fusion frame", "c2"),
            ("How FrameFT builds its frame", "c3"),
            ("Training, and the design choices in the code", "c4"),
            ("Our experiment: frame vs. random vs. identity", "c5"),
        ]
        rows = VGroup()
        for i, (text, _) in enumerate(items):
            circ = Circle(0.26, color=TRAIN_C, stroke_width=2.5)
            num = T(str(i + 1), 24, TRAIN_C, weight=BOLD).move_to(circ)
            rows.add(VGroup(VGroup(circ, num), T(text, 30, INK_2)).arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, buff=0.36, aligned_edge=LEFT)
        VGroup(head, rows).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to([0, 0.2, 0])
        self.play(FadeIn(head), run_time=0.5)
        for row, (_, mark) in zip(rows, items):
            self.at(tr, mark, lead=0.15)
            self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.45)
            row[1].set_color(INK)
        self.finish(tr, pad=0.6)
        self.clear_all()


class S02Problem(NScene):
    chapter = (2, "The problem: fine-tuning")

    def construct(self):
        W = D["W"]
        rng = np.random.default_rng(3)
        x = rng.standard_normal(768)
        y = x @ W

        tr = self.narrate("problem_0")
        self.show_tag()
        self.at(tr, "mats", lead=0.2)
        w_img = mat_image(W, 4.3, border=W_C, stroke=2)
        w_img.move_to([-1.2, 0.05, 0])
        w_cap = T("A real weight matrix: RoBERTa-base, layer 1, query", 20, MUTED).next_to(w_img, DOWN, buff=0.18)
        self.play(FadeIn(w_img, scale=0.96), FadeIn(w_cap), run_time=1.0)
        # zoom into the corner to show that it's just numbers
        corner = Square(0.26, color=TRAIN_C, stroke_width=2.5).move_to(w_img.get_corner(UL) + np.array([0.13, -0.13, 0]))
        nums = cell_matrix(W[:4, :4], cell=0.78, fmt="{:+.2f}", size=21,
                           fill_fn=lambda v, i, j: ManimColor(rgb_to_color(colorize(np.array([[v]]), vmax=0.25)[0, 0] / 255)))
        nums.move_to([3.9, 0.5, 0])
        nums_cap = T("just numbers", 22, INK_2).next_to(nums, DOWN, buff=0.2)
        links = VGroup(Line(corner.get_corner(UR), nums.get_corner(UL), stroke_width=1.2, color=TRAIN_C),
                       Line(corner.get_corner(DR), nums.get_corner(DL), stroke_width=1.2, color=TRAIN_C))
        self.play(Create(corner), run_time=0.4)
        self.play(Create(links), FadeIn(nums, scale=0.8), FadeIn(nums_cap), run_time=0.9)

        self.at(tr, "x", lead=0.35)
        self.play(FadeOut(nums), FadeOut(nums_cap), FadeOut(links), FadeOut(corner), FadeOut(w_cap),
                  w_img.animate.scale(0.8).move_to([0, 0.3, 0]), run_time=0.7)
        x_strip = strip(x, 3.0, 0.34).move_to([-4.6, 0.3, 0])
        x_lab = T("input x", 26, INK_2).next_to(x_strip, UP, buff=0.2)
        dot_op = M(r"\cdot", 110).move_to([-2.45, 0.3, 0])
        self.play(FadeIn(x_strip, shift=RIGHT * 0.3), FadeIn(x_lab), run_time=0.6)

        self.at(tr, "W", lead=0.1)
        self.play(FadeIn(dot_op), Indicate(w_img[1], color=TRAIN_C, scale_factor=1.0), run_time=0.7)

        self.at(tr, "y", lead=0.2)
        eq_op = M("=", 60).move_to([2.45, 0.3, 0])
        y_strip = strip(y, 3.0, 0.34).move_to([4.6, 0.3, 0])
        y_lab = T("output y", 26, INK_2).next_to(y_strip, UP, buff=0.2)
        pulse = Dot(x_strip.get_center(), radius=0.09, color=TRAIN_C)
        self.play(FadeIn(pulse), run_time=0.15)
        self.play(pulse.animate.move_to(w_img.get_center()), run_time=0.45)
        self.play(pulse.animate.move_to(y_strip.get_center()), FadeIn(eq_op), FadeIn(y_strip), FadeIn(y_lab),
                  run_time=0.55)
        self.play(FadeOut(pulse), run_time=0.2)
        formula = M("y = x\\,W", 54).move_to([0, -2.35, 0])
        self.play(Write(formula), run_time=0.6)

        self.at(tr, "size", lead=0.2)
        br_top = Brace(w_img, UP, buff=0.08, color=INK_2)
        br_left = Brace(w_img, LEFT, buff=0.08, color=INK_2)
        t_top = T("768", 24, INK_2).next_to(br_top, UP, buff=0.08)
        t_left = T("768", 24, INK_2).next_to(br_left, LEFT, buff=0.08)
        count = T("768 × 768 = 589,824 numbers", 34, INK).move_to(formula)
        self.play(FadeOut(x_lab), FadeOut(y_lab), GrowFromCenter(br_top), GrowFromCenter(br_left),
                  FadeIn(t_top), FadeIn(t_left), run_time=0.6)
        self.play(ReplacementTransform(formula, count), run_time=0.7)
        self.finish(tr)

        # ---- problem_1: delta W
        tr = self.narrate("problem_1")
        self.at(tr, "dw", lead=0.3)
        self.play(*[FadeOut(m) for m in (x_strip, dot_op, eq_op, y_strip, br_top, br_left, t_top, t_left, count)],
                  w_img.animate.scale_to_fit_height(2.7).move_to([-4.3, 0.2, 0]), run_time=0.7)
        dw_img = mat_image(D["dW_frame"], 2.7, border=DW_C, stroke=2.5).move_to([-0.3, 0.2, 0])
        dw_lab = M(r"\Delta W", 46, DW_C).next_to(dw_img, DOWN, buff=0.22)
        card = term_card("Δ (delta)", "The Greek letter people use to mean “change in”. ΔW is the change in W.")
        self.play(FadeIn(dw_img, scale=0.9), FadeIn(dw_lab), run_time=0.7)
        self.show_term(card)

        self.at(tr, "sum", lead=0.3)
        plus = M("+", 60).move_to([-2.3, 0.2, 0])
        eq = M("=", 60).move_to([1.7, 0.2, 0])
        new_img = mat_image(D["W"] + 0.35 * D["dW_frame"] / np.abs(D["dW_frame"]).max(), 2.7, border=INK_2, stroke=2)
        new_img.move_to([3.7, 0.2, 0])
        w_lab = M("W", 46, W_C).next_to(w_img, DOWN, buff=0.22)
        new_lab = M(r"W + \Delta W", 46, INK).next_to(new_img, DOWN, buff=0.22)
        subs = VGroup(T("pretrained", 20, MUTED).next_to(w_lab, DOWN, buff=0.1),
                      T("the change", 20, MUTED).next_to(dw_lab, DOWN, buff=0.1),
                      T("fine-tuned", 20, MUTED).next_to(new_lab, DOWN, buff=0.1))
        self.play(FadeOut(card), FadeIn(plus), FadeIn(w_lab), run_time=0.5)
        self.play(FadeIn(eq), FadeIn(new_img, shift=LEFT * 0.2), FadeIn(new_lab), FadeIn(subs), run_time=0.8)

        self.at(tr, "frozen", lead=0.15)
        b_frozen = badge("frozen").next_to(w_img, UP, buff=0.22)
        b_train = badge("trainable").next_to(dw_img, UP, buff=0.22)
        self.play(FadeIn(b_frozen, shift=DOWN * 0.1), run_time=0.5)
        self.play(FadeIn(b_train, shift=DOWN * 0.1), Indicate(dw_img[1], color=TRAIN_C, scale_factor=1.0), run_time=0.7)
        self.finish(tr)

        # ---- problem_2: the cost of an unrestricted delta W
        tr = self.narrate("problem_2")
        self.at(tr, "any", lead=0.2)
        big = T("589,824", 44, TRAIN_C, weight=BOLD).move_to([0, -2.45, 0])
        big_lab = T("trainable numbers per matrix", 24, INK_2).next_to(big, RIGHT, buff=0.25)
        VGroup(big, big_lab).move_to([0, -2.45, 0])
        self.play(FadeIn(big, shift=UP * 0.1), FadeIn(big_lab), Indicate(dw_img[1], color=TRAIN_C, scale_factor=1.0),
                  run_time=0.8)

        self.at(tr, "copies", lead=0.3)
        self.play(*[FadeOut(m) for m in (w_img, dw_img, new_img, plus, eq, w_lab, dw_lab, new_lab, subs, b_frozen,
                                          b_train, big, big_lab)], run_time=0.6)
        files = Group()
        for i, task in enumerate(("sentiment", "paraphrase", "entailment", "grammar")):
            box = panel(2.6, 3.0, color=GRID, fill=PANEL_2)
            icon = model_stack(n_layers=8, width=1.6, bar_h=0.14, gap=0.05).move_to(box).shift(UP * 0.2)
            for l in icon:
                l[1].set_fill(TRAIN_C, opacity=0.8)
            lab = T(f"{task} model", 22, INK).next_to(icon, DOWN, buff=0.2)
            size = T("≈ 500 MB", 20, MUTED).next_to(lab, DOWN, buff=0.08)
            files.add(Group(box, icon, lab, size))
        files.arrange(RIGHT, buff=0.35).move_to([0, 0.1, 0])
        self.play(LaggedStart(*[FadeIn(f, shift=UP * 0.3) for f in files], lag_ratio=0.25), run_time=1.4)
        note = T("one full copy per task", 26, INK_2).next_to(files, DOWN, buff=0.35)
        self.play(FadeIn(note), run_time=0.4)

        self.at(tr, "q", lead=0.3)
        self.play(FadeOut(files), FadeOut(note), run_time=0.5)
        q = T("How can we describe ΔW\nwith far fewer numbers?", 50, INK, weight=BOLD, line_spacing=1.1,
              t2c={"ΔW": DW_C})
        self.play(FadeIn(q, scale=0.95), run_time=0.8)
        self.finish(tr, pad=0.8)
        self.clear_all()


def dct_pattern(u, v, size=64):
    k = np.arange(size)
    cu = np.cos(np.pi * (2 * k + 1) * u / (2 * size))
    cv = np.cos(np.pi * (2 * k + 1) * v / (2 * size))
    return np.outer(cu, cv)


def toy_photo(size=64):
    """A tiny synthetic landscape: sky gradient, a sun, two mountains."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = 0.25 + 0.45 * yy
    img = np.where((xx - 0.72) ** 2 + (yy - 0.26) ** 2 < 0.012, 1.0, img)
    ridge1 = 0.55 + 0.25 * np.abs(xx - 0.3) * 2
    ridge2 = 0.62 + 0.3 * np.abs(xx - 0.78) * 2
    img = np.where(yy > np.minimum(ridge1, ridge2), 0.08 + 0.1 * yy, img)
    return img


class S03Landscape(NScene):
    chapter = (3, "Parameter-efficient fine-tuning")

    def construct(self):
        rng = np.random.default_rng(5)
        tr = self.narrate("landscape_0")
        self.show_tag()
        self.at(tr, "peft", lead=0.1)
        card = term_card("PEFT", "Parameter-efficient fine-tuning: learn a small number of new parameters that "
                                 "describe ΔW, instead of all of the model's weights.")
        self.show_term(card)

        self.at(tr, "lora", lead=0.2)
        lora_t = T("LoRA", 40, INK, weight=BOLD).move_to([-4.6, 2.3, 0])
        dw = mat_image(D["dW_frame"], 3.0, border=DW_C, stroke=2.5).move_to([-4.3, -0.1, 0])
        dw_lab = M(r"\Delta W", 42, DW_C).next_to(dw, DOWN, buff=0.2)
        self.play(FadeIn(lora_t), FadeIn(dw), FadeIn(dw_lab), run_time=0.7)

        U = rng.standard_normal((768, 8))
        V = rng.standard_normal((8, 768))
        eq = M("=", 56).move_to([-2.2, -0.1, 0])
        tall = mat_image(U, 3.0, width=0.42, border=TRAIN_C, stroke=2.5).move_to([-1.2, -0.1, 0])
        times = M(r"\times", 50).move_to([-0.45, -0.1, 0])
        wide = mat_image(V, 0.42, width=3.0, border=TRAIN_C, stroke=2.5).move_to([1.35, -0.1, 0])
        self.at(tr, "tall", lead=0.2)
        self.play(FadeIn(eq), FadeIn(tall, shift=RIGHT * 0.5), run_time=0.8)
        tall_lab = T("768 × r", 22, INK_2).next_to(tall, DOWN, buff=0.2)
        self.play(FadeIn(tall_lab), run_time=0.3)
        self.at(tr, "wide", lead=0.2)
        self.play(FadeIn(times), FadeIn(wide, shift=RIGHT * 0.5), run_time=0.8)
        wide_lab = T("r × 768", 22, INK_2).next_to(wide, DOWN, buff=0.2)
        self.play(FadeIn(wide_lab), run_time=0.3)

        self.at(tr, "rank", lead=0.2)
        br = Brace(tall, UP, buff=0.08, color=TRAIN_C)
        br_t = M("r", 38, TRAIN_C).next_to(br, UP, buff=0.06)
        br2 = Brace(wide, RIGHT, buff=0.08, color=TRAIN_C)
        br2_t = M("r", 38, TRAIN_C).next_to(br2, RIGHT, buff=0.06)
        self.play(FadeOut(card), GrowFromCenter(br), FadeIn(br_t), GrowFromCenter(br2), FadeIn(br2_t), run_time=0.6)
        card = term_card("Rank", "Roughly, how many independent patterns a matrix really contains. "
                                 "A rank-r matrix is a sum of r simple “row times column” pieces.")
        self.show_term(card)

        self.at(tr, "count", lead=0.3)
        calc = M(r"r\,(768 + 768) = 8 \times 1536 = 12{,}288", 40, TRAIN_C).move_to([0.2, -2.35, 0])
        vs = T("instead of 589,824", 24, MUTED).next_to(calc, RIGHT, buff=0.3)
        VGroup(calc, vs).move_to([0, -2.4, 0])
        self.play(Write(calc), run_time=0.9)
        self.play(FadeIn(vs), run_time=0.4)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---- landscape_1: a dictionary of patterns
        tr = self.narrate("landscape_1")
        self.at(tr, "dict", lead=0.2)
        pats = [(0, 1), (1, 0), (1, 1), (0, 3), (2, 1), (3, 2)]
        coefs = [0.9, -0.5, 0.7, 0.3, -0.6, 0.4]
        tiles = Group(*[mat_image(dct_pattern(u, v), 1.15, vmax=1, border=FROZEN_C, stroke=1.5) for u, v in pats])
        tiles.arrange(RIGHT, buff=0.3).move_to([-1.7, 1.0, 0])
        dict_lab = VGroup(snowflake(0.3), T("a fixed dictionary of patterns", 26, FROZEN_C)).arrange(RIGHT, buff=0.15)
        dict_lab.next_to(tiles, UP, buff=0.3)
        self.play(LaggedStart(*[FadeIn(t, shift=DOWN * 0.2) for t in tiles], lag_ratio=0.12), FadeIn(dict_lab),
                  run_time=1.2)

        self.at(tr, "coef", lead=0.2)
        c_labels = VGroup(*[T(f"× {c:+.1f}", 26, TRAIN_C, weight=BOLD).next_to(t, DOWN, buff=0.18)
                            for t, c in zip(tiles, coefs)])
        combo = sum(c * dct_pattern(u, v) for (u, v), c in zip(pats, coefs))
        result = mat_image(combo, 1.7, border=DW_C, stroke=2.5).move_to([4.6, 1.0, 0])
        res_eq = M("=", 50).next_to(result, LEFT, buff=0.25)
        coef_lab = VGroup(flame(0.32), T("learned coefficients", 24, TRAIN_C)).arrange(RIGHT, buff=0.12)
        coef_lab.next_to(c_labels, DOWN, buff=0.22).align_to(tiles, LEFT)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in c_labels], lag_ratio=0.1), FadeIn(coef_lab),
                  run_time=0.9)
        self.play(FadeIn(res_eq), FadeIn(result, shift=LEFT * 0.2), run_time=0.7)
        card = term_card("Coefficient", "A number that says how much of one pattern to use.")
        self.show_term(card)

        self.at(tr, "jpeg", lead=0.2)
        self.play(FadeOut(card), run_time=0.3)
        photo = toy_photo()
        coeffs = dctn(photo, norm="ortho")
        keep = np.abs(coeffs) >= np.sort(np.abs(coeffs).ravel())[-60]
        approx = idctn(coeffs * keep, norm="ortho")
        orig_img = mat_image(photo - 0.5, 1.9, vmax=0.55, border=GRID).move_to([-3.6, -1.65, 0])
        approx_img = mat_image(approx - 0.5, 1.9, vmax=0.55, border=GRID).move_to([1.0, -1.65, 0])
        o_lab = T("4,096 pixels", 22, INK_2).next_to(orig_img, RIGHT, buff=0.3)
        a_lab = VGroup(T("60 wave coefficients", 24, TRAIN_C), T("the idea behind JPEG", 20, MUTED))
        a_lab.arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(approx_img, RIGHT, buff=0.3)
        arrow = Arrow(o_lab.get_right() + RIGHT * 0.05, approx_img.get_left() + LEFT * 0.1, buff=0.1,
                      stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.2)
        self.play(FadeIn(orig_img), FadeIn(o_lab), run_time=0.6)
        self.play(GrowArrow(arrow), FadeIn(approx_img, shift=RIGHT * 0.2), FadeIn(a_lab), run_time=0.9)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---- landscape_2: FourierFT and FrameFT, then the roadmap
        tr = self.narrate("landscape_2")
        self.at(tr, "fourier", lead=0.2)

        def method_card(name, desc, color, wave):
            n = T(name, 36, color, weight=BOLD)
            d_ = T(desc, 22, INK_2)
            ax = VGroup(n, d_).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            w = FunctionGraph(wave, x_range=[0, 1.0], color=color, stroke_width=3)
            content = VGroup(ax, w).arrange(RIGHT, buff=0.35)
            box = panel(content.width + 0.6, content.height + 0.55, color=color, fill=PANEL).move_to(content)
            return VGroup(box, content)

        fourier = method_card("FourierFT", "dictionary: Fourier waves", INK_2,
                              lambda t: 0.3 * np.sin(2 * PI * 2 * t)).move_to([-3.35, 0.9, 0])
        frame = method_card("FrameFT", "dictionary: a tight fusion frame", FRAME_C,
                            lambda t: 0.18 * np.sin(2 * PI * 1.5 * t) + 0.12 * np.cos(2 * PI * 4 * t)).move_to([3.35, 0.9, 0])
        self.play(FadeIn(fourier, shift=UP * 0.2), run_time=0.7)
        self.at(tr, "frameft", lead=0.1)
        self.play(FadeIn(frame, shift=UP * 0.2), run_time=0.7)

        self.at(tr, "tff", lead=0.1)
        words = VGroup(T("tight", 44, INK, weight=BOLD), T("fusion", 44, INK, weight=BOLD),
                       T("frame", 44, FRAME_C, weight=BOLD)).arrange(RIGHT, buff=0.35).move_to([0, -1.05, 0])
        self.play(Indicate(frame[0], color=FRAME_C, scale_factor=1.02), FadeIn(words, shift=UP * 0.15), run_time=0.8)

        self.at(tr, "b", lead=0.15)
        chips = VGroup(pill("basis", INK, 28), pill("frame", INK, 28), pill("fusion frame", INK, 28))
        chips.arrange(RIGHT, buff=0.9).move_to([0, -2.35, 0])
        arrows = VGroup(*[Arrow(chips[i].get_right(), chips[i + 1].get_left(), buff=0.12, stroke_width=3,
                                color=MUTED, max_tip_length_to_length_ratio=0.25) for i in range(2)])
        for i, mark in enumerate(("b", "f", "ff")):
            if i:
                self.at(tr, mark, lead=0.15)
                self.play(GrowArrow(arrows[i - 1]), FadeIn(chips[i], shift=RIGHT * 0.15), run_time=0.45)
            else:
                self.play(FadeIn(chips[0], shift=RIGHT * 0.15), run_time=0.45)
            chips[i][0].set_stroke(TRAIN_C)
        self.finish(tr, pad=0.6)
        self.clear_all()
