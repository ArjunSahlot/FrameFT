"""Chapters 6-8: building the frame, the FrameFT update, and the two knobs (block size and scale)."""
import numpy as np
from PIL import Image

from common import *  # noqa: F401,F403
from data import load

D = load()
B = D["B_frame"]
N = 768


def stepper(active):
    """'1 · 2 · 3' progress marker for the three construction steps."""
    items = VGroup()
    for i, name in enumerate(("Spectral Tetris", "Modulation", "Real 2×2 blocks")):
        on = i + 1 == active
        circ = Circle(0.2, color=TRAIN_C if on else MUTED, stroke_width=2.5,
                      fill_color=TRAIN_C if on else BG, fill_opacity=1 if on else 0)
        num = T(str(i + 1), 20, BG if on else MUTED, weight=BOLD).move_to(circ)
        lab = T(name, 22, INK if on else MUTED, weight=BOLD if on else NORMAL)
        items.add(VGroup(VGroup(circ, num), lab).arrange(RIGHT, buff=0.15))
    items.arrange(RIGHT, buff=0.6).move_to([0.6, 2.75, 0])
    return items


def wave_plot(values, width, height, color):
    xs = np.linspace(-width / 2, width / 2, len(values))
    ys = values / (np.abs(values).max() or 1) * height / 2
    line = VMobject(stroke_color=color, stroke_width=3).set_points_smoothly([[x, y, 0] for x, y in zip(xs, ys)])
    axis = Line([-width / 2, 0, 0], [width / 2, 0, 0], color=GRID, stroke_width=1.2)
    return VGroup(axis, line)


def image_from_file(path, height):
    img = Image.open(path).convert("RGBA")
    mob = ImageMobject(np.array(img))
    mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    return mob.scale_to_fit_height(height)


class S06Build(NScene):
    chapter = (6, "Building the frame")

    def construct(self):
        # ---------------------------------------------------------------- build_0: generated, not stored
        tr = self.narrate("build_0")
        self.show_tag()
        self.at(tr, "never", lead=0.2)
        b_img = mat_image(B, 2.4, border=FRAME_C, stroke=2).move_to([3.9, 0.9, 0])
        b_lab = M("B", 46, FRAME_C).next_to(b_img, UP, buff=0.15)
        nots = VGroup(badge("frozen", "never trained"),
                      VGroup(T("×", 30, BAD, weight=BOLD), T("never saved to disk", 20, INK_2)).arrange(RIGHT, buff=0.12))
        nots.arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(b_img, DOWN, buff=0.3)
        self.play(FadeIn(b_img), FadeIn(b_lab), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(n, shift=UP * 0.1) for n in nots], lag_ratio=0.4), run_time=0.8)

        self.at(tr, "regen", lead=0.2)
        n_in = pill("n = 768", INK, 26)
        l_in = pill("l = 2", INK, 26)
        ins = VGroup(n_in, l_in).arrange(DOWN, buff=0.25).move_to([-5.9, 0.9, 0])
        fn_text = Mono('build_basis("frame", n, l)', 22, INK)
        fn = VGroup(panel(fn_text.width + 0.6, 1.1, color=FRAME_C, fill=PANEL), fn_text).move_to([-1.35, 0.9, 0])
        fn[1].move_to(fn[0])
        a1 = Arrow(ins.get_right(), fn.get_left(), buff=0.15, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.2)
        a2 = Arrow(fn.get_right(), b_img.get_left(), buff=0.15, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.2)
        same = T("same two numbers in → the same B out, every time", 24, INK_2).move_to([-1.7, -0.45, 0])
        self.play(FadeIn(ins, shift=RIGHT * 0.2), run_time=0.5)
        self.play(GrowArrow(a1), FadeIn(fn), run_time=0.6)
        self.play(GrowArrow(a2), Indicate(b_img[1], color=FRAME_C, scale_factor=1.0), run_time=0.6)
        self.play(FadeIn(same), run_time=0.5)

        self.at(tr, "share", lead=0.3)
        self.play(*[FadeOut(m) for m in (ins, fn, a1, a2, same, nots)], run_time=0.5)
        self.play(Group(b_img, b_lab).animate.move_to([0, -0.15, 0]), run_time=0.6)
        mats = VGroup()
        for i in range(24):
            ang = PI / 2 + 2 * PI * i / 24
            pos = np.array([4.6 * np.cos(ang), 2.05 * np.sin(ang) - 0.15, 0])
            sq = Square(0.36, stroke_color=W_C, stroke_width=1.5, fill_color=PANEL_2, fill_opacity=1).move_to(pos)
            lab = T(("Q" if i % 2 == 0 else "V") + str(i // 2 + 1), 13, INK_2).move_to(sq)
            mats.add(VGroup(sq, lab))
        links = VGroup(*[Line(m.get_center(), b_img.get_center(), stroke_width=1, color=FRAME_C, stroke_opacity=0.5)
                         for m in mats])
        links.set_z_index(-1)
        share_lab = T("24 wrapped matrices (query and value × 12 layers) → one shared B", 22, INK_2).move_to([0, -2.55, 0])
        self.play(LaggedStart(*[FadeIn(m, scale=0.7) for m in mats], lag_ratio=0.03), run_time=0.9)
        self.play(Create(links), FadeIn(share_lab), run_time=0.8)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---------------------------------------------------------------- build_1: Spectral Tetris
        tr = self.narrate("build_1")
        steps = stepper(0)
        self.play(FadeIn(steps, shift=DOWN * 0.1), run_time=0.6)
        self.at(tr, "tetris", lead=0.3)
        self.play(Transform(steps, stepper(1)), run_time=0.5)
        cell = 1.0
        rows, cols = 2, 5
        origin = np.array([-4.6, 1.0, 0])

        def cpos(r, c):
            return origin + np.array([c * cell, -r * cell, 0])

        grid = VGroup(*[Square(cell, stroke_color=GRID, stroke_width=1.5).move_to(cpos(r, c))
                        for r in range(rows) for c in range(cols)])
        zeros = VGroup(*[T("0", 26, MUTED).move_to(cpos(r, c)) for r in range(rows) for c in range(cols)])
        seed_lab = T("seed matrix: l = 2 rows, n = 5 columns", 24, INK_2).next_to(grid, UP, buff=0.25)
        self.play(Create(grid), FadeIn(zeros), FadeIn(seed_lab), run_time=0.8)

        self.at(tr, "cols", lead=0.2)
        col_note = VGroup(*[T("‖col‖ = 1", 17, FROZEN_C).next_to(grid[c + (rows - 1) * cols], DOWN, buff=0.12)
                            for c in range(cols)])
        self.play(LaggedStart(*[FadeIn(c) for c in col_note], lag_ratio=0.1), run_time=0.7)

        self.at(tr, "rows", lead=0.2)
        energy = [ValueTracker(0.0), ValueTracker(0.0)]
        meters = VGroup()
        for r in range(rows):
            meters.add(always_redraw(lambda r=r: VGroup(
                T("row energy", 20, MUTED),
                T(f"{energy[r].get_value():.2f}", 26, TRAIN_C if abs(energy[r].get_value() - 2.5) < 1e-3 else INK,
                  weight=BOLD),
                T("/ 2.5", 22, MUTED)).arrange(RIGHT, buff=0.15).next_to(cpos(r, cols - 1), RIGHT, buff=0.75)))
        target = VGroup(T("rows: equal energy, n / l = 5 / 2 = 2.5 each", 22, INK_2),
                        T("and perpendicular to each other", 22, INK_2)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        target.next_to(cpos(1, cols - 1), RIGHT, buff=0.75).shift(DOWN * 1.05)
        perp = VGroup()
        self.play(FadeIn(meters), FadeIn(target), FadeIn(perp), run_time=0.6)

        fills = VGroup()

        def put(r, c, text, color=TRAIN_C, e=1.0):
            sq = Square(cell, stroke_width=0, fill_color=color, fill_opacity=0.35).move_to(cpos(r, c))
            t = T(text, 26, INK, weight=BOLD).move_to(cpos(r, c))
            idx = r * cols + c
            fills.add(sq, t)
            return AnimationGroup(FadeIn(sq), FadeOut(zeros[idx]), FadeIn(t, scale=1.3),
                                  energy[r].animate.increment_value(e))

        self.play(put(0, 0, "1"), run_time=0.45)
        self.play(put(0, 1, "1"), run_time=0.45)

        self.at(tr, "piece", lead=0.2)
        vals = [["0.50", "0.50"], ["0.87", "−0.87"]]
        piece = VGroup()
        for r in range(2):
            for c in range(2):
                sq = Square(cell, stroke_color=DW_C, stroke_width=3, fill_color=DW_C, fill_opacity=0.4)
                t = T(vals[r][c], 24, INK, weight=BOLD).move_to(sq)
                piece.add(VGroup(sq, t).move_to(cpos(r, 2 + c)))
        piece.shift(UP * 3.2)
        leftover = T("row 1 needs 0.5 more: split it with a 2×2 piece", 22, DW_C).move_to([-2.6, -1.75, 0])
        formula = M(r"\tfrac{1}{\sqrt 2}\begin{pmatrix}\sqrt{x} & \sqrt{x}\\ \sqrt{2-x} & -\sqrt{2-x}\end{pmatrix},\ x = 0.5",
                    34, INK_2).next_to(leftover, DOWN, buff=0.25)
        self.play(FadeIn(leftover), run_time=0.4)
        self.play(piece.animate.shift(DOWN * 3.2), FadeOut(VGroup(*[zeros[r * cols + c] for r in range(2) for c in (2, 3)])),
                  energy[0].animate.increment_value(0.5), energy[1].animate.increment_value(1.5),
                  run_time=0.8, rate_func=rate_functions.ease_in_quad)
        self.play(FadeIn(formula), run_time=0.5)
        self.play(put(1, 4, "1"), run_time=0.5)

        self.at(tr, "even", lead=0.2)
        self.play(*[FadeOut(m) for m in (grid, zeros, fills, piece, col_note, meters, target, perp, leftover, formula,
                                          seed_lab)], run_time=0.6)
        ours = T("In FrameFT, n / l is always a whole number: no pieces needed", 26, INK).move_to([0, 1.85, 0])

        def ones_row(pattern, color=TRAIN_C):
            g = VGroup()
            for r, row in enumerate(pattern):
                for c, v in enumerate(row):
                    sq = Square(0.48, stroke_color=GRID, stroke_width=1.2,
                                fill_color=color if v else BG, fill_opacity=0.35 if v else 0)
                    g.add(VGroup(sq, T("1" if v else "0", 18, INK if v else MUTED)).move_to([c * 0.48, -r * 0.48, 0]))
            return g

        one = ones_row([[1] * 12])
        stair = ones_row([[1] * 6 + [0] * 6, [0] * 6 + [1] * 6])
        one_lab = VGroup(Mono("tff_l = 2", 22, INK), T("(RoBERTa runs): one row of 384 ones", 22, INK_2)).arrange(RIGHT, buff=0.2)
        stair_lab = VGroup(Mono("tff_l = 4", 22, INK), T(": a staircase, 192 ones per row", 22, INK_2)).arrange(RIGHT, buff=0.2)
        g1 = VGroup(one_lab, VGroup(one, T("…", 30, MUTED)).arrange(RIGHT, buff=0.15)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        g2 = VGroup(stair_lab, VGroup(stair, T("…", 30, MUTED)).arrange(RIGHT, buff=0.15)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        VGroup(g1, g2).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to([0, -0.2, 0])
        self.play(FadeIn(ours), run_time=0.5)
        self.play(FadeIn(g1, shift=UP * 0.1), run_time=0.6)
        self.play(FadeIn(g2, shift=UP * 0.1), run_time=0.6)
        self.finish(tr)
        self.play(FadeOut(ours), FadeOut(g1), FadeOut(g2), run_time=0.5)

        # ---------------------------------------------------------------- build_2: modulation
        tr = self.narrate("build_2")
        self.at(tr, "mod", lead=0.3)
        self.play(Transform(steps, stepper(2)), run_time=0.5)

        self.at(tr, "arrows", lead=0.3)
        cp_c = np.array([-5.4, 1.2, 0])
        cp = VGroup(Circle(1.0, color=GRID, stroke_width=1.5).move_to(cp_c),
                    Line(cp_c + LEFT * 1.15, cp_c + RIGHT * 1.15, color=MUTED, stroke_width=1.2),
                    Line(cp_c + DOWN * 1.15, cp_c + UP * 1.15, color=MUTED, stroke_width=1.2))
        phase = ValueTracker(0.5)
        ph = always_redraw(lambda: Arrow(cp_c, cp_c + np.array([np.cos(phase.get_value()), np.sin(phase.get_value()), 0]),
                                         buff=0, color=TRAIN_C, stroke_width=5, max_tip_length_to_length_ratio=0.2))
        ph_lab = M("a + ib", 30, TRAIN_C).next_to(cp, UP, buff=0.12)
        rot_lab = T("multiplying = rotating", 20, INK_2).next_to(cp, DOWN, buff=0.12)
        self.play(FadeIn(cp), FadeIn(ph), FadeIn(ph_lab), run_time=0.6)
        self.play(phase.animate.set_value(0.5 + 1.2), FadeIn(rot_lab), run_time=0.9)

        self.at(tr, "rotate", lead=0.3)
        K, rows_k = 12, 5
        dx, dy, r0 = 0.52, 0.62, 0.2
        g_origin = np.array([-2.7, 1.6, 0])
        t_sweep = ValueTracker(0.0)   # spins every hand at its copy's own speed
        shown = ValueTracker(1.0)     # how many copies are visible

        def clocks():
            g = VGroup()
            for k in range(rows_k):
                if k >= shown.get_value():
                    break
                for n in range(K):
                    c = g_origin + np.array([n * dx, -k * dy, 0])
                    ang = 2 * PI * k * (n + t_sweep.get_value()) / K
                    g.add(Circle(r0, color=GRID, stroke_width=1.2).move_to(c))
                    g.add(Line(c, c + r0 * 0.9 * np.array([np.cos(ang), np.sin(ang), 0]), color=BAR_C[k], stroke_width=3))
                    g.add(Dot(c, radius=0.03, color=BAR_C[k]))  # the hub, so 0° and 180° look different
            return g

        BAR_C = [INK_2, FRAME_C, RANDOM_C, IDENT_C, DW_C]
        clock_g = always_redraw(clocks)
        row_labs = VGroup(*[T(f"copy {k}", 18, BAR_C[k]).move_to(g_origin + np.array([-0.75, -k * dy, 0]))
                            for k in range(rows_k)])
        col_lab = T("column n = 0 … 11  (angle grows along the row)", 18, MUTED).move_to(
            g_origin + np.array([(K - 1) * dx / 2, 0.45, 0]))
        seed_note = T("copy 0 = the seed (all ones)", 18, INK_2).next_to(g_origin + np.array([(K - 1) * dx, 0, 0]), RIGHT, buff=0.35)
        self.add(clock_g)
        self.play(FadeIn(row_labs[0]), FadeIn(col_lab), FadeIn(seed_note), run_time=0.5)
        self.play(shown.animate.set_value(2), FadeIn(row_labs[1]), run_time=0.6)

        self.at(tr, "clocks", lead=0.2)
        self.play(shown.animate.set_value(rows_k), FadeIn(row_labs[2:]), FadeOut(seed_note), run_time=0.7)
        self.play(t_sweep.animate.set_value(3.0), run_time=self.upto(tr, "cancel", lead=0.3, minimum=1.0),
                  rate_func=linear)
        self.play(t_sweep.animate.set_value(0.0), run_time=0.3)

        # two copies with different speeds: their product's arrows chain into a closed loop (sum 0)
        self.at(tr, "cancel", lead=0.2)
        hi = VGroup(SurroundingRectangle(row_labs[1], color=FRAME_C, buff=0.06),
                    SurroundingRectangle(row_labs[3], color=IDENT_C, buff=0.06))
        chain_c = np.array([5.3, 0.9, 0])
        step = 0.5
        pts, cur = [chain_c], chain_c.copy()
        for n in range(K):
            ang = 2 * PI * (1 - 3) * n / K
            cur = cur + step * np.array([np.cos(ang), np.sin(ang), 0])
            pts.append(cur.copy())
        loop = VGroup(*[Arrow(pts[i], pts[i + 1], buff=0, stroke_width=3, color=TRAIN_C,
                              max_tip_length_to_length_ratio=0.3) for i in range(K)])
        loop.move_to(chain_c)
        loop_lab = T("copy 1 vs copy 3:\narrows chain into a loop,\nsum = 0 → perpendicular", 19, INK_2, line_spacing=0.9)
        loop_lab.next_to(loop, DOWN, buff=0.25)
        same = VGroup(*[Arrow(ORIGIN, RIGHT * step, buff=0, stroke_width=3, color=MUTED, max_tip_length_to_length_ratio=0.3)
                        for _ in range(6)]).arrange(RIGHT, buff=0).next_to(loop_lab, DOWN, buff=0.35)
        same_lab = T("same speed: arrows line up", 18, MUTED).next_to(same, DOWN, buff=0.1)
        self.play(Create(hi), run_time=0.4)
        self.play(LaggedStart(*[GrowArrow(a) for a in loop], lag_ratio=0.35), run_time=1.6)
        self.play(FadeIn(loop_lab), run_time=0.4)
        self.play(FadeIn(same), FadeIn(same_lab), run_time=0.5)
        dft = M(r"\sum_{n=0}^{K-1} e^{2\pi i (k-k')n/K} = \begin{cases} K & k = k' \\ 0 & k \ne k' \end{cases}", 34, INK)
        dft.move_to([-1.0, -2.15, 0])
        self.play(Write(dft), run_time=1.0)
        self.finish(tr)
        self.play(*[FadeOut(m) for m in (cp, ph, ph_lab, rot_lab, clock_g, row_labs, col_lab, hi, loop, loop_lab, same,
                                          same_lab, dft)], run_time=0.6)

        # ---------------------------------------------------------------- build_3: complex -> real 2x2 blocks
        tr = self.narrate("build_3")
        self.play(Transform(steps, stepper(3)), run_time=0.5)
        self.at(tr, "complex", lead=0.3)
        z = M(r"a + i\,b", 64, TRAIN_C).move_to([-3.0, 1.55, 0])
        self.play(Write(z), run_time=0.6)
        self.at(tr, "block", lead=0.2)
        arr = Arrow(z.get_right(), z.get_right() + RIGHT * 1.6, buff=0.25, color=MUTED, stroke_width=3)
        blk = M(r"\begin{pmatrix} a & -b \\ b & a \end{pmatrix}", 60, TRAIN_C).next_to(arr, RIGHT, buff=0.25)
        self.play(GrowArrow(arr), FadeIn(blk, shift=RIGHT * 0.2), run_time=0.7)
        check1 = M(r"(a+ib)(x+iy) = (ax - by) + i\,(bx + ay)", 34, INK_2)
        check2 = M(r"\begin{pmatrix} a & -b \\ b & a \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} = "
                   r"\begin{pmatrix} ax - by \\ bx + ay \end{pmatrix}", 34, INK_2)
        checks = VGroup(check1, check2).arrange(DOWN, buff=0.3).move_to([-0.5, -0.35, 0])
        same_t = T("same rotation-and-scaling, written with real numbers", 22, GOOD).next_to(checks, DOWN, buff=0.25)
        VGroup(checks, same_t).move_to([0, -0.15, 0])
        self.play(FadeIn(checks, shift=UP * 0.1), run_time=0.7)
        self.play(FadeIn(same_t), run_time=0.4)

        self.at(tr, "double", lead=0.2)
        cz = VGroup(*[Square(0.5, stroke_color=TRAIN_C, stroke_width=2, fill_color=TRAIN_C, fill_opacity=0.2)
                      for _ in range(8)]).arrange(RIGHT, buff=0.08)
        cz_lab = T("384 complex coordinates", 22, TRAIN_C)
        rz = VGroup(*[Square(0.5, stroke_color=FRAME_C, stroke_width=2, fill_color=FRAME_C, fill_opacity=0.2 + 0.25 * (i % 2))
                      for i in range(16)]).arrange(RIGHT, buff=0.04)
        rz_lab = T("768 real coordinates", 22, FRAME_C)
        top = VGroup(cz, T("…", 28, MUTED)).arrange(RIGHT, buff=0.15)
        bot = VGroup(rz, T("…", 28, MUTED)).arrange(RIGHT, buff=0.15)
        g = VGroup(VGroup(cz_lab, top).arrange(RIGHT, buff=0.35), VGroup(rz_lab, bot).arrange(RIGHT, buff=0.35))
        g.arrange(DOWN, aligned_edge=RIGHT, buff=0.3).move_to([0, -2.2, 0])
        self.play(FadeIn(g[0]), run_time=0.4)
        self.play(*[TransformFromCopy(cz[i], VGroup(rz[2 * i], rz[2 * i + 1])) for i in range(8)], FadeIn(bot[1]),
                  FadeIn(rz_lab), run_time=0.9)
        self.finish(tr)
        self.play(*[FadeOut(m) for m in (z, arr, blk, checks, same_t, g)], run_time=0.5)

        # ---------------------------------------------------------------- build_4: the result
        tr = self.narrate("build_4")
        self.at(tr, "result", lead=0.3)
        big = mat_image(B, 4.6, border=FRAME_C, stroke=2).move_to([-3.4, -0.05, 0])
        big_lab = VGroup(M("B", 44, FRAME_C), T("768 × 768, from build_basis(\"frame\", 768, 2)", 20, INK_2))
        big_lab.arrange(RIGHT, buff=0.25).next_to(big, UP, buff=0.15)
        self.play(FadeOut(steps), FadeIn(big), FadeIn(big_lab), run_time=0.8)

        self.at(tr, "waves", lead=0.3)
        plots = VGroup()
        for i, k in enumerate((1, 2, 3, 5)):
            row = B[2 * k, 0::2]  # even entries of row 2k hold cos(2*pi*k*n/384)
            w = wave_plot(row, 4.6, 0.75, [FRAME_C, RANDOM_C, IDENT_C, DW_C][i])
            lab = T(f"row {2 * k}: frequency {k}", 18, INK_2).next_to(w, LEFT, buff=0.2)
            plots.add(VGroup(lab, w))
        plots.arrange(DOWN, buff=0.28, aligned_edge=RIGHT).move_to([3.25, -0.05, 0])
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.2) for p in plots], lag_ratio=0.25), run_time=1.3)

        self.at(tr, "fourier", lead=0.2)
        concl = T("a real-valued Fourier basis", 30, FRAME_C, weight=BOLD).move_to([3.25, -2.3, 0])
        self.play(FadeIn(concl, shift=UP * 0.1), run_time=0.6)
        self.finish(tr, pad=0.6)
        self.clear_all()


class S07Update(NScene):
    chapter = (7, "The FrameFT update")

    def construct(self):
        S = D["S_768"]
        # ---------------------------------------------------------------- update_0: the formula
        tr = self.narrate("update_0")
        self.show_tag()
        eq = MathTex(r"\Delta W", r"=", r"\frac{\text{scale}}{n}", r"\cdot", r"B^{\top}", r"\,S\,", r"B", font_size=66, color=INK)
        eq[0].set_color(DW_C)
        eq[2].set_color(INK_2)
        eq[4].set_color(FRAME_C)
        eq[5].set_color(TRAIN_C)
        eq[6].set_color(FRAME_C)
        eq.move_to([0, 2.3, 0])
        self.play(FadeIn(eq[0:2]), run_time=0.5)
        size = 2.6
        bt_img = mat_image(B.T, size, border=FRAME_C, stroke=2).move_to([-4.0, -0.35, 0])
        s_img = mat_image(S, size, cmap="div", border=TRAIN_C, stroke=2.5, dilate=1, gamma=0.5).move_to([0, -0.35, 0])
        b_img = mat_image(B, size, border=FRAME_C, stroke=2).move_to([4.0, -0.35, 0])
        dots = VGroup(M(r"\cdot", 90).move_to([-2.0, -0.35, 0]), M(r"\cdot", 90).move_to([2.0, -0.35, 0]))
        labs = VGroup(badge("frozen", "frozen frame").next_to(bt_img, DOWN, buff=0.2),
                      badge("trainable", "sparse coefficients").next_to(s_img, DOWN, buff=0.2),
                      badge("frozen", "frozen frame").next_to(b_img, DOWN, buff=0.2))

        self.at(tr, "bt", lead=0.15)
        self.play(FadeIn(eq[4]), FadeIn(bt_img), FadeIn(labs[0]), run_time=0.5)
        self.at(tr, "S", lead=0.15)
        self.play(FadeIn(eq[5]), FadeIn(s_img), FadeIn(dots[0]), FadeIn(labs[1]), run_time=0.5)
        self.at(tr, "B", lead=0.15)
        self.play(FadeIn(eq[6]), FadeIn(b_img), FadeIn(dots[1]), FadeIn(labs[2]), run_time=0.5)
        self.at(tr, "scale", lead=0.15)
        self.play(FadeIn(eq[2:4]), run_time=0.5)

        self.at(tr, "Sgrid", lead=0.2)
        self.play(FadeOut(labs[1]), Indicate(eq[5], color=TRAIN_C), Indicate(s_img[1], color=TRAIN_C, scale_factor=1.0),
                  run_time=0.6)
        zero_note = T("768 × 768, almost all zeros", 22, INK_2).next_to(s_img, DOWN, buff=0.2)
        self.play(FadeIn(zero_note), run_time=0.4)

        self.at(tr, "thousand", lead=0.2)
        count = T("1,000 nonzero out of 589,824  (0.17%)", 24, TRAIN_C, weight=BOLD).move_to(zero_note)
        self.play(FadeOut(labs[0]), FadeOut(labs[2]), run_time=0.3)
        dot_note = T("dots enlarged to be visible", 16, MUTED).next_to(count, DOWN, buff=0.1)
        self.play(ReplacementTransform(zero_note, count), FadeIn(dot_note), run_time=0.6)

        self.at(tr, "only", lead=0.2)
        only = VGroup(flame(0.45), T("the only trainable numbers in this matrix", 24, TRAIN_C)).arrange(RIGHT, buff=0.15)
        only.next_to(s_img, UP, buff=0.2)
        self.play(FadeIn(only, shift=DOWN * 0.1), run_time=0.6)
        self.finish(tr)

        # ---------------------------------------------------------------- update_1: measure, edit, rebuild
        tr = self.narrate("update_1")
        self.play(*[FadeOut(m) for m in (bt_img, s_img, b_img, dots, labs[0], labs[2], count, dot_note, only)],
                  eq.animate.scale(0.75).move_to([0, 2.45, 0]), run_time=0.6)
        rng = np.random.default_rng(11)
        x = np.convolve(rng.standard_normal(N + 40), np.ones(40) / 40, mode="valid")[:N] + 0.3 * rng.standard_normal(N)
        coords = x @ B.T
        edited = coords @ S
        out = edited @ B
        sw, bw = 2.0, 0.95
        y0 = 0.1
        strips = [strip(v, sw, 0.36) for v in (x, coords, edited, out)]
        slots = Group(strips[0], Square(bw), strips[1], Square(bw), strips[2], Square(bw), strips[3])
        slots.arrange(RIGHT, buff=0.32).move_to([0, y0, 0])
        box_x = [slots[1].get_center()[0], slots[3].get_center()[0], slots[5].get_center()[0]]
        strip_labs = [T("input x", 20, INK_2), T("shadows on the\nframe directions", 20, INK_2, line_spacing=0.85),
                      T("edited shadows", 20, INK_2), T("output: x · ΔW", 20, INK_2)]
        for s_, l_ in zip(strips, strip_labs):
            l_.next_to(s_, DOWN, buff=0.22)
        ops = []
        for name, color, img, px in ((r"B^{\top}", FRAME_C, B.T, box_x[0]), (r"S", TRAIN_C, S, box_x[1]),
                                     (r"B", FRAME_C, B, box_x[2])):
            box = mat_image(img, bw, border=color, stroke=2, dilate=1 if name == "S" else 0,
                            gamma=0.5 if name == "S" else 0.75).move_to([px, y0, 0])
            lab = M(r"\cdot\," + name, 34, color).next_to(box, UP, buff=0.12)
            ops.append(Group(box, lab))
        stage = [T("analysis", 26, FRAME_C, weight=BOLD), T("edit", 26, TRAIN_C, weight=BOLD),
                 T("synthesis", 26, FRAME_C, weight=BOLD)]
        for st, op in zip(stage, ops):
            st.next_to(op, UP, buff=0.18)
        self.play(FadeIn(strips[0], shift=RIGHT * 0.2), FadeIn(strip_labs[0]), run_time=0.5)

        self.at(tr, "analysis", lead=0.2)
        self.play(FadeIn(ops[0]), run_time=0.4)
        self.play(FadeIn(strips[1], shift=RIGHT * 0.2), FadeIn(strip_labs[1]), FadeIn(stage[0]), run_time=0.6)
        self.at(tr, "edit", lead=0.2)
        self.play(FadeIn(ops[1]), run_time=0.4)
        self.play(FadeIn(strips[2], shift=RIGHT * 0.2), FadeIn(strip_labs[2]), FadeIn(stage[1]), run_time=0.6)
        self.at(tr, "synth", lead=0.2)
        self.play(FadeIn(ops[2]), run_time=0.4)
        self.play(FadeIn(strips[3], shift=RIGHT * 0.2), FadeIn(strip_labs[3]), FadeIn(stage[2]), run_time=0.6)
        self.at(tr, "mer", lead=0.2)
        motto = T("measure  →  edit  →  rebuild", 34, INK, weight=BOLD).move_to([0, -1.9, 0])
        self.play(FadeIn(motto, shift=UP * 0.1), run_time=0.6)
        self.finish(tr)
        self.play(*[FadeOut(m) for m in (*strips, *strip_labs, *ops, *stage, motto)], run_time=0.6)

        # ---------------------------------------------------------------- update_2: atoms
        tr = self.narrate("update_2")
        self.at(tr, "atom", lead=0.3)
        i_row, j_row = 2 * 2, 2 * 3 + 1
        bi, bj = B[i_row], B[j_row]
        atom = np.outer(bi, bj)
        sq = 3.0
        center = np.array([-2.2, -0.45, 0])
        col_strip = mat_image(bi[:, None], sq, width=0.3, border=FRAME_C).move_to(center + LEFT * (sq / 2 + 0.35))
        row_strip = mat_image(bj[None, :], 0.3, width=sq, border=FRAME_C).move_to(center + UP * (sq / 2 + 0.35))
        col_lab = M(rf"b_{{{i_row}}}^{{\top}}", 32, FRAME_C).next_to(col_strip, LEFT, buff=0.15)
        row_lab = M(rf"b_{{{j_row}}}", 32, FRAME_C).next_to(row_strip, UP, buff=0.1)
        coef = ValueTracker(1.0)
        vmax = np.abs(atom).max()
        atom_img = live(lambda: mat_image(coef.get_value() * atom, sq, vmax=vmax, border=DW_C, stroke=2).move_to(center))
        atom_eq = M(rf"c \cdot b_{{{i_row}}}^{{\top}} b_{{{j_row}}}", 40, INK).move_to([3.2, 1.3, 0])
        atom_note = T("one frame row stood on its end,\ntimes another lying flat", 22, INK_2, line_spacing=0.9)
        atom_note.next_to(atom_eq, DOWN, buff=0.25)
        self.play(FadeIn(col_strip, shift=RIGHT * 0.2), FadeIn(col_lab), FadeIn(row_strip, shift=DOWN * 0.2), FadeIn(row_lab),
                  run_time=0.7)
        self.add(atom_img)
        self.play(FadeIn(atom_eq), FadeIn(atom_note), run_time=0.6)

        self.at(tr, "covers", lead=0.2)
        cover = T("reaches (nearly) every entry", 24, TRAIN_C).next_to(atom_note, DOWN, buff=0.45)
        flash = SurroundingRectangle(Square(sq).move_to(center), color=TRAIN_C, buff=0.04, stroke_width=4)
        self.play(Create(flash), FadeIn(cover), run_time=0.6)
        self.play(FadeOut(flash), run_time=0.3)

        self.at(tr, "nudge", lead=0.3)
        c_read = always_redraw(lambda: T(f"c = {coef.get_value():+.2f}", 30, TRAIN_C, weight=BOLD).next_to(cover, DOWN, buff=0.4))
        self.add(c_read)
        self.play(coef.animate.set_value(-1.0), run_time=1.2, rate_func=rate_functions.ease_in_out_sine)
        self.play(coef.animate.set_value(0.6), run_time=1.0, rate_func=rate_functions.ease_in_out_sine)

        self.at(tr, "sumatoms", lead=0.3)
        self.remove(atom_img, c_read)
        self.play(*[FadeOut(m) for m in (col_strip, row_strip, col_lab, row_lab, atom_eq, atom_note, cover)], run_time=0.5)
        picks = [(4, 7), (10, 3), (1, 22)]
        terms = Group()
        for n_, (a, b_) in enumerate(picks):
            tile = mat_image(np.outer(B[a], B[b_]), 1.35, border=DW_C, stroke=1.5)
            c = T(["+0.8", "−1.1", "+0.4"][n_], 24, TRAIN_C, weight=BOLD)
            terms.add(Group(c, tile).arrange(RIGHT, buff=0.12))
        plus = [M("+", 44) for _ in range(3)]
        dots3 = T("…", 40, MUTED)
        eqs = M("=", 50)
        dw = mat_image(D["dW_frame"], 2.3, border=DW_C, stroke=2.5)
        row = Group(terms[0], plus[0], terms[1], plus[1], terms[2], plus[2], dots3, eqs, dw).arrange(RIGHT, buff=0.22)
        row.move_to([0, 0.1, 0])
        dw_lab = M(r"\Delta W", 40, DW_C).next_to(dw, DOWN, buff=0.15)
        formula = M(r"\Delta W = \frac{\text{scale}}{n}\sum_{(i,j)} c_{ij}\; b_i^{\top} b_j", 44, INK).move_to([0, -2.1, 0])
        n_terms = T("1,000 terms", 24, TRAIN_C).next_to(formula, RIGHT, buff=0.35)
        self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.1) for m in row], lag_ratio=0.12), run_time=1.3)
        self.play(FadeIn(dw_lab), Write(formula), FadeIn(n_terms), run_time=0.9)
        self.finish(tr)
        self.wait(1.5)  # a beat to take in the sum
        self.play(FadeOut(row), FadeOut(dw_lab), FadeOut(formula), FadeOut(n_terms), run_time=0.5)

        # ---------------------------------------------------------------- update_3: where the 1,000 go
        tr = self.narrate("update_3")
        self.at(tr, "random", lead=0.3)
        rows_, cols_ = D["rows_768"], D["cols_768"]
        shown = ValueTracker(0)

        def s_partial():
            k = int(shown.get_value())
            M_ = np.zeros((N, N))
            M_[rows_[:k], cols_[:k]] = 1.0
            return mat_image(M_, 3.3, vmax=1, cmap="mono", border=TRAIN_C, stroke=2, dilate=1, gamma=1).move_to([-3.3, -0.3, 0])

        s_live = live(s_partial)
        counter = always_redraw(lambda: T(f"{int(shown.get_value()):,} / 1,000 positions", 22, TRAIN_C)
                                .move_to([-3.3, -2.25, 0]))
        code = VGroup(Mono("torch.randperm(589_824,", 21, INK),
                      Mono("    generator=seed(2024))[:1000]", 21, INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        code_box = panel(code.width + 0.4, code.height + 0.35, color=GRID, fill=PANEL_2).move_to(code)
        code_g = VGroup(code_box, code).move_to([3.2, 0.85, 0])
        self.add(s_live, counter)
        self.play(shown.animate.set_value(1000), FadeIn(code_g), run_time=2.2, rate_func=rate_functions.ease_in_quad)
        card = term_card("Seed", "The starting number for a random number generator. Same seed, same “random” "
                                 "choices, every time.")
        self.show_term(card)

        self.at(tr, "nosave", lead=0.3)
        saved = VGroup(T("Saved to disk:", 24, INK, weight=BOLD),
                       VGroup(flame(0.3), T("the 1,000 coefficient values", 22, TRAIN_C)).arrange(RIGHT, buff=0.12),
                       T("Rebuilt from seeds instead:", 24, INK, weight=BOLD),
                       VGroup(snowflake(0.28), T("the frame B and the 1,000 positions", 22, FROZEN_C)).arrange(RIGHT, buff=0.12))
        saved.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.2, -0.85, 0])
        saved[2].shift(DOWN * 0.15)
        saved[3].shift(DOWN * 0.15)
        self.play(FadeOut(card), LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in saved], lag_ratio=0.2), run_time=1.0)

        self.at(tr, "share", lead=0.3)
        self.remove(s_live, counter)
        final = s_partial()
        self.add(final)
        self.play(FadeOut(code_g), FadeOut(saved), final.animate.scale(0.55).move_to([-4.6, 0.2, 0]), run_time=0.6)
        thumbs = Group()
        for layer in range(1, 6):
            t_ = s_partial().scale(0.33)
            lab = T(f"layer {layer + 1}", 16, INK_2).next_to(t_, DOWN, buff=0.08)
            thumbs.add(Group(t_, lab))
        thumbs.add(T("…", 30, MUTED))
        thumbs.arrange(RIGHT, buff=0.3).move_to([1.4, 0.15, 0])
        l1 = T("layer 1", 16, INK_2).next_to(final, DOWN, buff=0.08)
        share = VGroup(Mono("share_entry=True", 22, INK), T(": the same positions in every layer", 22, INK_2)).arrange(RIGHT, buff=0.1)
        share.move_to([0, -1.9, 0])
        self.play(FadeIn(l1), LaggedStart(*[FadeIn(t_, shift=RIGHT * 0.15) for t_ in thumbs], lag_ratio=0.12), run_time=1.0)
        self.play(FadeIn(share), run_time=0.5)
        self.finish(tr, pad=0.6)
        self.clear_all()


class S08Knobs(NScene):
    chapter = (8, "Two knobs: block size and scale")

    def construct(self):
        # ---------------------------------------------------------------- knobs_0: block size
        tr = self.narrate("knobs_0")
        self.show_tag()
        self.at(tr, "bs", lead=0.3)
        size = 2.6
        left = mat_image(np.abs(D["S_768"]) > 0, size, vmax=1, cmap="mono", border=TRAIN_C, stroke=2, dilate=1, gamma=1)
        right = mat_image(np.abs(D["S_2"]) > 0, size, vmax=1, cmap="mono", border=TRAIN_C, stroke=2, dilate=1, gamma=1)
        left.move_to([-5.0, 0.45, 0])
        right.move_to([-1.55, 0.45, 0])
        title = T("knob 1: block size", 28, INK, weight=BOLD).move_to([-2.5, 2.45, 0])
        self.play(FadeIn(title), run_time=0.4)
        self.at(tr, "mrpc", lead=0.3)
        l_lab = VGroup(Mono("tff_block_size = 768", 20, INK), T("MRPC: anywhere", 22, INK_2)).arrange(DOWN, buff=0.1)
        l_lab.next_to(left, DOWN, buff=0.2)
        self.play(FadeIn(left), FadeIn(l_lab), run_time=0.6)
        self.at(tr, "rte", lead=0.3)
        r_lab = VGroup(Mono("tff_block_size = 2", 20, INK), T("RTE: only on the diagonal", 22, INK_2)).arrange(DOWN, buff=0.1)
        r_lab.next_to(right, DOWN, buff=0.2)
        self.play(FadeIn(right), FadeIn(r_lab), run_time=0.6)

        self.at(tr, "diag", lead=0.3)
        S2 = D["S_2"][:12, :12]
        cells = VGroup()
        c = 0.26
        for i in range(12):
            for j in range(12):
                on = S2[i, j] != 0
                allowed = i // 2 == j // 2
                sq = Square(c, stroke_color=GRID, stroke_width=0.8,
                            fill_color=TRAIN_C if on else (PANEL_2 if allowed else BG), fill_opacity=0.9 if on else 1)
                cells.add(sq.move_to([j * c, -i * c, 0]))
        blocks = VGroup(*[Square(2 * c, stroke_color=INK_2, stroke_width=2).move_to([(2 * b + 0.5) * c, -(2 * b + 0.5) * c, 0])
                          for b in range(6)])
        zoom = VGroup(cells, blocks).move_to([3.0, 0.45, 0])
        corner = Square(size * 12 / N * 4, stroke_color=INK, stroke_width=2).move_to(right.get_corner(UL) + np.array([0.09, -0.09, 0]))
        zl = VGroup(Line(corner.get_corner(UR), zoom.get_corner(UL), stroke_width=1.2, color=INK_2),
                    Line(corner.get_corner(DR), zoom.get_corner(DL), stroke_width=1.2, color=INK_2))
        z_lab = T("zoomed corner: 2×2 diagonal blocks", 20, INK_2).next_to(zoom, DOWN, buff=0.2)
        self.play(Create(corner), Create(zl), FadeIn(zoom), FadeIn(z_lab), run_time=0.9)

        self.at(tr, "eq", lead=0.3)
        self.play(*[FadeOut(m) for m in (left, l_lab, corner, zl)], run_time=0.4)
        sliders = VGroup()
        levels = [ValueTracker(0.5) for _ in range(10)]
        rng = np.random.default_rng(4)
        targets = rng.uniform(0.15, 0.9, size=(3, 10))
        for k in range(10):
            track = Line(UP * 0.8, DOWN * 0.8, color=GRID, stroke_width=4)
            knob = always_redraw(lambda k=k, track=track: RoundedRectangle(
                corner_radius=0.05, width=0.34, height=0.16, stroke_width=0, fill_color=TRAIN_C, fill_opacity=1
            ).move_to(track.point_from_proportion(1 - levels[k].get_value())))
            lab = T(f"f{k}", 16, MUTED).next_to(track, DOWN, buff=0.1)
            sliders.add(VGroup(track, knob, lab))
        sliders.arrange(RIGHT, buff=0.22).move_to([-4.4, 0.4, 0])
        eq_lab = T("each frequency's plane\nis adjusted on its own", 22, INK_2, line_spacing=0.9).next_to(sliders, DOWN, buff=0.3)
        self.play(FadeIn(sliders), FadeIn(eq_lab), run_time=0.5)
        for tgt in targets[:2]:
            self.play(*[levels[k].animate.set_value(tgt[k]) for k in range(10)], run_time=0.8,
                      rate_func=rate_functions.ease_in_out_sine)

        self.at(tr, "paper", lead=0.3)
        fig = Image.open(HERE.parent / "figures" / "svd_vs_FF_FrameFT_merge.png").convert("RGBA").crop((0, 240, 780, 462))
        fig_path = BUILD / "paper_crop.png"
        fig.save(fig_path)
        paper = image_from_file(fig_path, 1.55)
        card_bg = panel(paper.width + 0.4, paper.height + 0.3, color=GRID, fill="#F7F7F4").move_to(paper)
        paper_g = Group(card_bg, paper).move_to([2.2, -1.85, 0])
        p_lab = T("the paper's figure: C is block-diagonal", 20, INK_2).next_to(paper_g, UP, buff=0.12)
        self.play(FadeOut(z_lab), zoom.animate.scale(0.8).move_to([4.9, 1.2, 0]), right.animate.scale(0.75).move_to([1.4, 1.25, 0]),
                  FadeOut(r_lab), run_time=0.5)
        self.play(FadeIn(paper_g, shift=UP * 0.2), FadeIn(p_lab), run_time=0.6)
        self.play(*[levels[k].animate.set_value(targets[2][k]) for k in range(10)], run_time=0.8)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---------------------------------------------------------------- knobs_1: scale
        tr = self.narrate("knobs_1")
        self.at(tr, "scale", lead=0.3)
        title = T("knob 2: scale", 28, INK, weight=BOLD).move_to([-4.3, 2.45, 0])
        eq = MathTex(r"\Delta W", r"=", r"\frac{\text{scale}}{n}", r"\, B^{\top} S\, B", font_size=54, color=INK)
        eq[0].set_color(DW_C)
        eq[2].set_color(TRAIN_C)
        eq.move_to([-2.6, 1.3, 0])
        box = SurroundingRectangle(eq[2], color=TRAIN_C, buff=0.1)
        self.play(FadeIn(title), Write(eq), run_time=0.8)
        self.play(Create(box), run_time=0.4)

        self.at(tr, "vals", lead=0.2)
        vals = VGroup(VGroup(T("RTE", 24, INK_2), T("scale = 10", 26, INK, weight=BOLD)).arrange(RIGHT, buff=0.3),
                      VGroup(T("MRPC", 24, INK_2), T("scale = 50", 26, INK, weight=BOLD)).arrange(RIGHT, buff=0.3))
        vals.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(eq, DOWN, buff=0.45).align_to(eq, LEFT)
        n_note = T("n = 768, the matrix size", 20, MUTED).next_to(vals, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(FadeIn(vals, shift=UP * 0.1), FadeIn(n_note), run_time=0.6)

        self.at(tr, "norm", lead=0.3)
        # an orthogonal matrix turns a circle into the same circle: it rotates, never stretches
        cc = np.array([4.3, 1.15, 0])
        circ = Circle(0.95, color=INK_2, stroke_width=2).move_to(cc)
        rot = ValueTracker(0)
        spokes = always_redraw(lambda: VGroup(*[Line(cc, cc + 0.95 * np.array([np.cos(a + rot.get_value()), np.sin(a + rot.get_value()), 0]),
                                                     color=FRAME_C, stroke_width=3) for a in (0, 2.1, 4.2)]))
        c_lab = T("B rotates, never stretches", 20, INK_2).next_to(circ, DOWN, buff=0.15)
        norm = M(r"\|\Delta W\|_F = \frac{\text{scale}}{n}\;\|c\|", 50, INK).move_to([3.6, -1.0, 0])
        self.play(FadeIn(circ), FadeIn(spokes), FadeIn(c_lab), run_time=0.5)
        self.play(rot.animate.set_value(1.6), Write(norm), run_time=1.2)
        card = term_card("Norm  ‖·‖", "The size of a vector or matrix: the square root of the sum of its squared entries.")
        card.to_corner(DL, buff=0.4).shift(UP * 1.05)
        self.show_term(card)

        self.at(tr, "prod", lead=0.3)
        self.play(FadeOut(card), FadeOut(circ), FadeOut(spokes), FadeOut(c_lab), norm.animate.scale(0.8).move_to([3.6, 1.35, 0]),
                  run_time=0.5)
        table = VGroup()
        head = [T(h, 20, MUTED) for h in ("task", "learning rate", "× scale", "=")]
        rows = [["RTE", "0.321", "× 10", "3.2"], ["MRPC", "0.078", "× 50", "3.9"]]
        grid_rows = [head] + [[T(v, 26, INK if j < 3 else TRAIN_C, weight=BOLD if j == 3 else NORMAL) for j, v in enumerate(r)]
                              for r in rows]
        for i, r in enumerate(grid_rows):
            for j, cell_ in enumerate(r):
                cell_.move_to([1.3 + j * 1.55, -0.1 - i * 0.55, 0])
                table.add(cell_)
        lr_card = term_card("Learning rate", "The step size: how far each training update moves the numbers being trained.")
        lr_card.to_corner(DL, buff=0.4).shift(UP * 1.05)
        self.play(FadeIn(table), run_time=0.6)
        self.show_term(lr_card)
        self.finish(tr)
        self.clear_all(keep_tag=True)

        # ---------------------------------------------------------------- knobs_2: initialization
        tr = self.narrate("knobs_2")
        self.at(tr, "init", lead=0.5)
        lora0 = mat_image(np.zeros((N, N)), 2.6, border=INK_2, stroke=2).move_to([-3.2, 0.2, 0])
        lora_lab = VGroup(T("LoRA, before training", 24, INK_2), M(r"\Delta W = 0", 36, INK)).arrange(DOWN, buff=0.12)
        lora_lab.next_to(lora0, DOWN, buff=0.2)
        frame0 = mat_image(D["dW_frame"], 2.6, border=DW_C, stroke=2).move_to([3.2, 0.2, 0])
        f_lab = VGroup(T("FrameFT, before training", 24, INK_2),
                       M(r"c \sim \mathcal{N}(0, 1) \;\Rightarrow\; \|\Delta W\|_F \approx 0.41", 32, INK)).arrange(DOWN, buff=0.12)
        f_lab.next_to(frame0, DOWN, buff=0.2)
        note = T("GLUE runs (init_std = 1); RTE settings: scale 10, 1,000 coefficients", 18, MUTED).next_to(f_lab, DOWN, buff=0.1)
        self.play(FadeIn(lora0), FadeIn(lora_lab), run_time=0.6)
        self.play(FadeIn(frame0), FadeIn(f_lab), FadeIn(note), run_time=0.6)
        self.finish(tr, pad=0.6)
        self.clear_all()
