"""Chapters 4-5: bases, frames, tight frames, and fusion frames."""
import numpy as np

from common import *  # noqa: F401,F403

UNIT = 1.55                       # plane units per math unit
ORIGIN = np.array([-3.3, -0.25, 0])
BAR_COLORS = [FRAME_C, RANDOM_C, IDENT_C]


def P(v):
    """Math coordinates -> scene point on the left-hand plane."""
    return ORIGIN + UNIT * np.array([v[0], v[1], 0])


def make_plane():
    plane = NumberPlane(x_range=[-2.1, 2.1, 1], y_range=[-1.6, 1.6, 1], x_length=4.2 * UNIT, y_length=3.2 * UNIT,
                        background_line_style={"stroke_color": GRID, "stroke_width": 1.2, "stroke_opacity": 0.8},
                        axis_config={"stroke_color": MUTED, "stroke_width": 1.5})
    plane.move_to(ORIGIN)
    return plane


def arrow(v, color, width=5):
    return Arrow(P((0, 0)), P(v), buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.16,
                 max_stroke_width_to_length_ratio=12)


def unit(angle_deg):
    a = np.deg2rad(angle_deg)
    return np.array([np.cos(a), np.sin(a)])


class ShadowBars(VGroup):
    """Live bars of the squared shadows <x, f_i>^2 and their stacked total, against a dashed target."""

    def __init__(self, frame_vecs, x_of, target, height_scale=0.9, origin=np.array([2.0, -2.0, 0])):
        super().__init__()
        self.frame_vecs, self.x_of, self.target, self.k, self.o = frame_vecs, x_of, target, height_scale, origin
        self.X = [0.5, 1.6, 2.7, 4.35]  # bar positions: three shadows, then the stacked sum
        self.static = VGroup()
        base = Line(origin + LEFT * 0.2, origin + RIGHT * 5.1, color=MUTED, stroke_width=1.5)
        labels = VGroup(*[M(rf"\langle x, f_{i + 1}\rangle^2", 26, BAR_COLORS[i]).move_to(origin + RIGHT * self.X[i] + DOWN * 0.36)
                          for i in range(len(frame_vecs))])
        total = T("sum", 24, INK).move_to(origin + RIGHT * self.X[3] + DOWN * 0.36)
        self.static.add(base, labels, total)
        self.add(self.static)
        self.dynamic = always_redraw(self._draw)
        self.add(self.dynamic)

    def _draw(self):
        x = self.x_of()
        sq = [float(np.dot(x, f)) ** 2 for f in self.frame_vecs]
        g = VGroup()
        for i, s in enumerate(sq):
            h = max(s * self.k, 0.001)
            g.add(Rectangle(width=0.62, height=h, stroke_width=0, fill_color=BAR_COLORS[i], fill_opacity=0.9)
                  .move_to(self.o + RIGHT * self.X[i] + UP * h / 2))
        y = 0
        for i, s in enumerate(sq):
            h = max(s * self.k, 0.001)
            g.add(Rectangle(width=0.8, height=h, stroke_width=0, fill_color=BAR_COLORS[i], fill_opacity=0.9)
                  .move_to(self.o + RIGHT * self.X[3] + UP * (y + h / 2)))
            y += h
        t = self.target(x) * self.k
        g.add(DashedLine(self.o + RIGHT * (self.X[3] - 0.55) + UP * t, self.o + RIGHT * (self.X[3] + 0.55) + UP * t,
                         color=INK, stroke_width=2, dash_length=0.08))
        g.add(M(r"\tfrac{3}{2}\|x\|^2", 26, INK_2).next_to(self.o + RIGHT * (self.X[3] + 0.55) + UP * t, RIGHT, buff=0.08))
        return g


class S04Frames(NScene):
    chapter = (4, "Bases and frames")

    def construct(self):
        # ---------------------------------------------------------------- frames_0: an orthonormal basis
        tr = self.narrate("frames_0")
        self.show_tag()
        plane = make_plane()
        self.at(tr, "basis", lead=0.2)
        e1, e2 = arrow((1, 0), FRAME_C), arrow((0, 1), FRAME_C)
        l1 = M("e_1", 36, FRAME_C).next_to(e1.get_end(), DOWN, buff=0.12)
        l2 = M("e_2", 36, FRAME_C).next_to(e2.get_end(), LEFT, buff=0.12)
        self.play(FadeIn(plane), run_time=0.6)
        self.play(GrowArrow(e1), GrowArrow(e2), FadeIn(l1), FadeIn(l2), run_time=0.8)

        self.at(tr, "ortho", lead=0.1)
        corner = RightAngle(Line(P((0, 0)), P((1, 0))), Line(P((0, 0)), P((0, 1))), length=0.28, color=INK_2)
        len_lab = T("length 1 each, at a right angle", 22, INK_2).next_to(plane, UP, buff=0.12)
        self.play(Create(corner), FadeIn(len_lab), run_time=0.6)
        card = term_card("Orthonormal basis", "Arrows that are perpendicular (ortho-) and of length one (-normal).")
        self.show_term(card)

        self.at(tr, "shadows", lead=0.2)
        xv = np.array([1.3, 0.9])
        x = arrow(xv, INK, 6)
        x_lab = M("x", 40, INK).next_to(x.get_end(), UR, buff=0.06)
        d1 = DashedLine(P(xv), P((xv[0], 0)), color=MUTED, stroke_width=2)
        d2 = DashedLine(P(xv), P((0, xv[1])), color=MUTED, stroke_width=2)
        s1 = Line(P((0, 0)), P((xv[0], 0)), color=TRAIN_C, stroke_width=9)
        s2 = Line(P((0, 0)), P((0, xv[1])), color=TRAIN_C, stroke_width=9)
        self.play(GrowArrow(x), FadeIn(x_lab), run_time=0.6)
        self.play(Create(d1), Create(d2), run_time=0.5)
        self.play(Create(s1), Create(s2), run_time=0.6)
        c1 = M(r"\langle x, e_1\rangle = 1.3", 34, TRAIN_C)
        c2 = M(r"\langle x, e_2\rangle = 0.9", 34, TRAIN_C)
        coords = VGroup(c1, c2).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.2, 0.7, 0])
        shadow_note = T("shadow = dot product", 24, INK_2).next_to(coords, UP, buff=0.35, aligned_edge=LEFT)
        self.play(FadeOut(card), FadeIn(shadow_note), Write(coords), run_time=0.9)

        self.at(tr, "pyth", lead=0.2)
        pyth = M(r"1.3^2 + 0.9^2 = 2.5 = \|x\|^2", 40, INK).next_to(coords, DOWN, buff=0.55, aligned_edge=LEFT)
        general = M(r"\sum_i \langle x, e_i\rangle^2 = \|x\|^2", 40, INK_2).next_to(pyth, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(Write(pyth), run_time=0.9)
        self.play(FadeIn(general, shift=UP * 0.1), run_time=0.6)
        self.finish(tr)

        # ---------------------------------------------------------------- frames_1: three arrows in 2D
        tr = self.narrate("frames_1")
        self.at(tr, "frame", lead=0.2)
        self.play(*[FadeOut(m) for m in (e1, e2, l1, l2, corner, len_lab, d1, d2, s1, s2, coords, shadow_note, pyth,
                                          general)], run_time=0.6)
        title = T("Frame: more arrows than dimensions", 25, INK, weight=BOLD)
        title.move_to([0.45 + title.width / 2, 2.3, 0])
        self.play(FadeIn(title), run_time=0.5)

        self.at(tr, "three", lead=0.2)
        angles = [90, 210, 330]
        F = [unit(a) for a in angles]
        f_arrows = VGroup(*[arrow(f, BAR_COLORS[i]) for i, f in enumerate(F)])
        f_labels = VGroup(*[M(f"f_{i + 1}", 36, BAR_COLORS[i]).move_to(P(F[i] * 1.28)) for i in range(3)])
        self.play(LaggedStart(*[GrowArrow(a) for a in f_arrows], lag_ratio=0.25), FadeIn(f_labels), run_time=1.0)
        note = T("3 arrows in a 2-D plane, 120° apart", 24, INK_2).next_to(title, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(note), run_time=0.4)

        self.at(tr, "redund", lead=0.2)
        # two different recipes for the same x: the tight-frame one, and that plus (t,t,t) since f1+f2+f3 = 0
        base = (2 / 3) * np.array([np.dot(xv, f) for f in F])

        def path(coeffs, color):
            pts, cur = [P((0, 0))], np.zeros(2)
            for c, f in zip(coeffs, F):
                cur = cur + c * f
                pts.append(P(cur))
            return VGroup(*[Arrow(pts[i], pts[i + 1], buff=0, color=color, stroke_width=4,
                                  max_tip_length_to_length_ratio=0.12) for i in range(3)])

        p1 = path(base, TRAIN_C)
        p2 = path(base + 0.55, GOOD)
        self.play(Create(p1), run_time=0.9)
        self.play(Create(p2), run_time=0.9)
        r1 = M(rf"x = {base[0]:.2f}f_1 {base[1]:+.2f}f_2 {base[2]:+.2f}f_3", 32, TRAIN_C)
        r2 = M(rf"x = {base[0] + .55:.2f}f_1 {base[1] + .55:+.2f}f_2 {base[2] + .55:+.2f}f_3", 32, GOOD)
        recipes = VGroup(r1, r2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(note, DOWN, buff=0.55, aligned_edge=LEFT)
        red = T("redundancy: 3 arrows ÷ 2 dimensions = 1.5", 24, INK_2).next_to(recipes, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(FadeIn(recipes, shift=UP * 0.1), run_time=0.6)
        self.play(FadeIn(red), run_time=0.5)
        self.finish(tr)

        # ---------------------------------------------------------------- frames_2: tightness, live
        tr = self.narrate("frames_2")
        self.at(tr, "tight", lead=0.25)
        self.play(FadeOut(p1), FadeOut(p2), FadeOut(recipes), FadeOut(red), FadeOut(note), FadeOut(x), FadeOut(x_lab),
                  run_time=0.5)
        new_title = T("Tight: no direction is favoured", 26, INK, weight=BOLD)
        new_title.move_to([0.45 + new_title.width / 2, 2.3, 0])
        self.play(ReplacementTransform(title, new_title), run_time=0.5)
        theta = ValueTracker(np.deg2rad(34))
        r = 1.35
        frame_state = {"F": F}
        xvec = lambda: r * np.array([np.cos(theta.get_value()), np.sin(theta.get_value())])
        live_x = always_redraw(lambda: arrow(xvec(), INK, 6))
        live_lab = always_redraw(lambda: M("x", 40, INK).move_to(P(xvec() * 1.2)))
        shadows = always_redraw(lambda: VGroup(*[
            Line(P((0, 0)), P(np.dot(xvec(), f) * f), color=BAR_COLORS[i], stroke_width=9, stroke_opacity=0.9)
            for i, f in enumerate(frame_state["F"])]))
        bars = ShadowBars(F, xvec, lambda x: 1.5 * np.dot(x, x), height_scale=0.62, origin=np.array([0.95, -1.85, 0]))
        bars.frame_vecs = F
        eq = M(r"\sum_i \langle x, f_i\rangle^2 = \tfrac{3}{2}\,\|x\|^2", 40, INK)
        eq.move_to([0.45 + eq.width / 2, 1.4, 0])
        self.add(shadows, live_x, live_lab)
        self.play(FadeIn(bars.static), run_time=0.4)
        self.add(bars.dynamic)

        self.at(tr, "spin", lead=0.1)
        omega = 2 * PI / 6.0  # one turn every six seconds
        t1 = self.upto(tr, "sum", lead=0.1)
        self.play(theta.animate.increment_value(omega * t1), run_time=t1, rate_func=linear)
        t2 = self.upto(tr, "lopsided", lead=0.3)
        self.play(AnimationGroup(theta.animate(run_time=t2, rate_func=linear).increment_value(omega * t2),
                                 FadeIn(eq, shift=DOWN * 0.1, run_time=0.6)))

        # lopsided frame: arrows bunched together; the total wobbles
        self.at(tr, "lopsided", lead=0.1)
        G = [unit(a) for a in (70, 100, 130)]
        g_arrows = VGroup(*[arrow(g, BAR_COLORS[i]) for i, g in enumerate(G)])
        g_labels = VGroup(*[M(f"f_{i + 1}", 36, BAR_COLORS[i]).move_to(P(G[i] * 1.3)) for i in range(3)])
        lop_title = T("Not tight: arrows bunched up", 26, BAD, weight=BOLD)
        lop_title.move_to([0.45 + lop_title.width / 2, 2.3, 0])
        self.play(ReplacementTransform(f_arrows, g_arrows), ReplacementTransform(f_labels, g_labels),
                  ReplacementTransform(new_title, lop_title), FadeOut(eq), run_time=0.8)
        frame_state["F"] = G
        bars.frame_vecs = G
        bars.target = lambda x: 1.5 * np.dot(x, x)
        wobble = T("now the sum changes with direction", 24, BAD)
        wobble.move_to([0.45 + wobble.width / 2, 1.5, 0])
        self.play(FadeIn(wobble), run_time=0.4)
        self.play(theta.animate.set_value(theta.get_value() + 2 * PI), run_time=self.until_end(tr, minimum=3.0),
                  rate_func=linear)
        self.finish(tr, pad=0.1)

        # ---------------------------------------------------------------- frames_3: analysis & synthesis
        tr = self.narrate("frames_3")
        frame_state["F"] = F
        self.remove(bars.dynamic)
        self.play(FadeOut(bars.static), FadeOut(wobble), ReplacementTransform(g_arrows, f_arrows),
                  ReplacementTransform(g_labels, f_labels), FadeOut(lop_title), run_time=0.7)
        theta.set_value(np.deg2rad(34))
        self.at(tr, "measure", lead=0.2)
        x_now = xvec()
        meas = [float(np.dot(x_now, f)) for f in F]
        m_title = T("1. Measure the shadows", 26, INK, weight=BOLD)
        m_title.move_to([0.35 + m_title.width / 2, 2.3, 0])
        m_vals = VGroup(*[M(rf"\langle x, f_{i + 1}\rangle = {m:+.2f}", 34, BAR_COLORS[i]) for i, m in enumerate(meas)])
        m_vals.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(m_title, DOWN, buff=0.35).align_to(m_title, LEFT)
        self.play(FadeIn(m_title), LaggedStart(*[FadeIn(v, shift=RIGHT * 0.1) for v in m_vals], lag_ratio=0.2),
                  run_time=0.9)

        self.at(tr, "rebuild", lead=0.2)
        self.remove(shadows)
        s_title = T("2. Add the arrows back", 26, INK, weight=BOLD).next_to(m_vals, DOWN, buff=0.5)
        s_title.align_to(m_title, LEFT)
        pts, cur = [P((0, 0))], np.zeros(2)
        for m, f in zip(meas, F):
            cur = cur + m * f
            pts.append(P(cur))
        chain = VGroup(*[Arrow(pts[i], pts[i + 1], buff=0, color=BAR_COLORS[i], stroke_width=5,
                               max_tip_length_to_length_ratio=0.14) for i in range(3)])
        self.play(FadeIn(s_title), run_time=0.4)
        self.play(LaggedStart(*[GrowArrow(a) for a in chain], lag_ratio=0.6), run_time=1.5)
        big = Arrow(P((0, 0)), P(cur), buff=0, color=MUTED, stroke_width=4, max_tip_length_to_length_ratio=0.1)
        formula = M(r"x = \frac{1}{A}\sum_i \langle x, f_i\rangle\, f_i", 40, INK).next_to(s_title, DOWN, buff=0.35)
        formula.align_to(m_title, LEFT)
        a_note = T("A = 3/2: the tightness constant", 22, INK_2).next_to(formula, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(FadeIn(big), Write(formula), run_time=0.9)
        self.play(FadeOut(chain), FadeOut(big), FadeIn(a_note), Indicate(live_x, color=TRAIN_C), run_time=0.8)

        self.at(tr, "names", lead=0.2)
        n1 = pill("analysis", FRAME_C, 20).next_to(m_title, RIGHT, buff=0.2)
        n2 = pill("synthesis", FRAME_C, 20).next_to(s_title, RIGHT, buff=0.2)
        self.play(FadeIn(n1, scale=0.9), FadeIn(n2, scale=0.9), run_time=0.6)
        self.finish(tr, pad=0.6)
        self.clear_all()


class S05aFusion(NScene):
    """3-D subspaces drawn with a hand-rolled orthographic projection, so live overlays stay simple."""
    chapter = (5, "Fusion frames")

    def construct(self):
        az = ValueTracker(-0.95)
        EL, C, S = 0.36, np.array([-2.4, -0.45, 0]), 1.3
        X, Y, Z = np.eye(3)

        def pr(p):
            x, y, z = p
            a = az.get_value()
            xr, yr = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
            return C + S * np.array([xr, z * np.cos(EL) + yr * np.sin(EL), 0])

        def arrow3(p, color, width=5):
            return Arrow(pr((0, 0, 0)), pr(p), buff=0, color=color, stroke_width=width,
                         max_tip_length_to_length_ratio=0.14, max_stroke_width_to_length_ratio=12)

        tr = self.narrate("fusion_0")
        self.show_tag()
        az.add_updater(lambda m, dt: m.increment_value(0.07 * dt))
        self.add(az)
        axes = always_redraw(lambda: VGroup(*[
            Line(pr(-L * e), pr(L * e), color=MUTED, stroke_width=1.5) for e, L in ((X, 2.3), (Y, 2.3), (Z, 1.9))]))
        self.play(FadeIn(axes), run_time=0.8)

        self.at(tr, "subspaces", lead=0.2)
        plane_op = ValueTracker(0.2)
        plane = always_redraw(lambda: Polygon(*[pr(q) for q in ((-1.9, -1.9, 0), (1.9, -1.9, 0), (1.9, 1.9, 0),
                                                                  (-1.9, 1.9, 0))],
                                              stroke_color=FRAME_C, stroke_width=2, fill_color=FRAME_C,
                                              fill_opacity=plane_op.get_value()))
        line = always_redraw(lambda: Line(pr((0, 0, -1.8)), pr((0, 0, 1.95)), color=RANDOM_C, stroke_width=3,
                                          stroke_opacity=0.55))
        lab_plane = T("subspace 1: a plane", 26, FRAME_C)
        lab_line = T("subspace 2: a line", 26, RANDOM_C)
        labels = VGroup(lab_plane, lab_line).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        labels.move_to([1.6 + labels.width / 2, 2.0, 0])
        self.play(FadeIn(plane), FadeIn(lab_plane), run_time=0.9)
        self.play(Create(line), FadeIn(lab_line), run_time=0.7)
        sub_note = T("Subspace: a flat slice through the\norigin (a line, a plane, ...)", 22, INK_2, line_spacing=0.9)
        sub_note.next_to(labels, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(FadeIn(sub_note), run_time=0.5)

        self.at(tr, "proj", lead=0.2)
        vec = np.array([1.2, 0.8, 1.3])
        v_state = {"v": vec}
        x = always_redraw(lambda: arrow3(v_state["v"], INK, 6))
        p_plane = always_redraw(lambda: arrow3((v_state["v"][0], v_state["v"][1], 0), "#7DB8FF", 9))
        p_line = always_redraw(lambda: arrow3((0, 0, v_state["v"][2]), "#FF9A6B", 9))
        drops = always_redraw(lambda: VGroup(
            DashedLine(pr(v_state["v"]), pr((v_state["v"][0], v_state["v"][1], 0)), color=INK_2, stroke_width=2),
            DashedLine(pr(v_state["v"]), pr((0, 0, v_state["v"][2])), color=INK_2, stroke_width=2)))
        x_lab = always_redraw(lambda: M("x", 38, INK).move_to(pr(v_state["v"] * 1.13)))
        self.play(FadeIn(x), FadeIn(x_lab), run_time=0.6)
        self.play(FadeIn(drops), run_time=0.5)
        self.play(FadeIn(p_plane), FadeIn(p_line), run_time=0.7)
        card = term_card("Projection", "The shadow of a vector on a subspace: the closest point to it inside that subspace.")
        self.play(FadeOut(sub_note), run_time=0.3)
        self.show_term(card)
        self.finish(tr)

        # ---------------------------------------------------------------- fusion_1: tight fusion frame
        tr = self.narrate("fusion_1")
        self.at(tr, "tightf", lead=0.2)
        eq = M(r"\sum_i \|P_i\,x\|^2 = A\,\|x\|^2", 44, INK)
        eq.move_to([1.6 + eq.width / 2, 0.35, 0])
        self.play(FadeOut(card), Write(eq), run_time=0.8)

        self.at(tr, "plane", lead=0.1)
        self.play(plane_op.animate.set_value(0.45), Indicate(lab_plane, color=FRAME_C, scale_factor=1.06), run_time=0.6)
        self.at(tr, "line", lead=0.1)
        self.play(plane_op.animate.set_value(0.2), Indicate(lab_line, color=RANDOM_C, scale_factor=1.06), run_time=0.6)

        self.at(tr, "pyth", lead=0.2)
        ang = ValueTracker(0.0)
        R = float(np.linalg.norm(vec))
        a0, b0 = np.arctan2(vec[1], vec[0]), np.arcsin(vec[2] / R)

        def v_now():  # x swings around on a sphere, so its length never changes
            a = a0 + ang.get_value()
            b = b0 * np.cos(0.75 * ang.get_value())
            return R * np.array([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)])

        def numbers():
            v = v_now()
            a, b = v[0] ** 2 + v[1] ** 2, v[2] ** 2
            l1 = M(r"\|P_1 x\|^2 + \|P_2 x\|^2", 36, INK_2)
            l2 = VGroup(T("=", 28, INK_2), T(f"{a:.2f}", 28, FRAME_C, weight=BOLD), T("+", 28, INK_2),
                        T(f"{b:.2f}", 28, RANDOM_C, weight=BOLD), T("=", 28, INK_2),
                        T(f"{a + b:.2f}", 28, INK, weight=BOLD)).arrange(RIGHT, buff=0.15)
            l3 = VGroup(M(r"= \|x\|^2", 36, INK), T("(here A = 1)", 22, MUTED)).arrange(RIGHT, buff=0.2)
            g = VGroup(l1, l2, l3).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            return g.next_to(eq, DOWN, buff=0.5).align_to(eq, LEFT)

        v_state["v"] = v_now()
        nums = always_redraw(numbers)
        self.add(nums)

        def follow(m):
            v_state["v"] = v_now()
        ang.add_updater(lambda m, dt: follow(m))
        self.add(ang)
        self.play(ang.animate.set_value(2 * PI), run_time=self.until_end(tr, minimum=4.0), rate_func=linear)
        self.finish(tr, pad=0.4)
        self.play(*[FadeOut(m) for m in (x, x_lab, p_plane, p_line, drops, plane, line, axes, labels, eq, nums)],
                  run_time=0.8)
        az.clear_updaters()


class S05bFusion(NScene):
    chapter = (5, "Fusion frames")

    def construct(self):
        from data import load
        D = load()
        self.add(self.tag)
        tr = self.narrate("fusion_2")
        head = T("The fusion frame FrameFT's code builds", 30, INK, weight=BOLD).move_to([0, 2.45, 0])
        self.play(FadeIn(head, shift=DOWN * 0.1), run_time=0.6)
        self.at(tr, "red1", lead=0.2)
        red = VGroup(T("In FrameFT's code:", 28, INK_2),
                     T("redundancy = 1", 34, TRAIN_C, weight=BOLD),
                     T("subspaces perpendicular, no overlap, A = 1", 26, INK)).arrange(DOWN, buff=0.18)
        red.move_to([0, 1.2, 0])
        self.play(FadeIn(red, shift=DOWN * 0.1), run_time=0.8)

        self.at(tr, "carve", lead=0.2)
        n_show = 24
        cells = VGroup(*[Square(0.4, stroke_color=GRID, stroke_width=1.5, fill_color=PANEL_2, fill_opacity=1)
                         for _ in range(n_show)]).arrange(RIGHT, buff=0)
        dots = T("…", 36, MUTED)
        strip_g = VGroup(cells, dots).arrange(RIGHT, buff=0.2).move_to([0, 0.35, 0])
        strip_lab = T("768 coordinates of a hidden vector", 24, INK_2).next_to(strip_g, UP, buff=0.2)
        self.play(FadeOut(red), FadeOut(head), FadeIn(strip_g), FadeIn(strip_lab), run_time=0.7)

        self.at(tr, "planes", lead=0.25)
        colors = [FRAME_C, RANDOM_C, IDENT_C, TRAIN_C, DW_C, "#FF7AB6"]
        braces = VGroup()
        for p in range(n_show // 2):
            col = colors[p % len(colors)]
            cells[2 * p].set_fill(col, opacity=0.55)
            cells[2 * p + 1].set_fill(col, opacity=0.55)
            b = Brace(VGroup(cells[2 * p], cells[2 * p + 1]), DOWN, buff=0.05, color=col)
            braces.add(b)
        plane_labels = VGroup(*[T(f"{p + 1}", 18, colors[p % len(colors)]).next_to(braces[p], DOWN, buff=0.05)
                                for p in range(n_show // 2)])
        plane_labels.add(T("each coloured pair = one plane", 22, INK_2).next_to(braces, DOWN, buff=0.45))
        count = T("384 planes × 2 dimensions = 768", 30, INK).move_to([0, -1.25, 0])
        self.play(LaggedStart(*[GrowFromCenter(b) for b in braces], lag_ratio=0.05),
                  cells.animate.set_stroke(GRID), run_time=0.9)
        self.play(FadeIn(plane_labels), FadeIn(count), run_time=0.5)

        self.at(tr, "rows", lead=0.2)
        self.play(*[FadeOut(m) for m in (strip_g, strip_lab, braces, plane_labels, count)], run_time=0.5)
        B = D["B_frame"]
        # the first 16 rows (8 planes), enlarged so the row pairs are visible
        top = mat_image(B[:16, :192], 3.2, width=6.4, border=FRAME_C, stroke=2).move_to([-1.6, 0.35, 0])
        pair_braces = VGroup()
        for p in range(8):
            y0 = top.get_top()[1] - 3.2 * (2 * p) / 16
            seg = Line([0, y0, 0], [0, y0 - 3.2 * 2 / 16, 0])
            seg.next_to(top, LEFT, buff=0.05).set_y(y0 - 3.2 / 16)
            pair_braces.add(Brace(seg, LEFT, buff=0.02, color=colors[p % len(colors)], sharpness=1.2))
        pl = VGroup(*[T(f"plane {p + 1}", 16, colors[p % len(colors)]).next_to(pair_braces[p], LEFT, buff=0.06)
                      for p in range(8)])
        crop_note = T("first 16 rows × 192 columns", 20, MUTED).next_to(top, DOWN, buff=0.15)
        self.play(FadeIn(top), run_time=0.7)
        self.play(LaggedStart(*[GrowFromCenter(b) for b in pair_braces], lag_ratio=0.08), FadeIn(pl), FadeIn(crop_note),
                  run_time=0.9)

        self.at(tr, "B", lead=0.2)
        full = mat_image(B, 3.6, border=FRAME_C, stroke=2).move_to([4.2, 0.3, 0])
        full_lab = VGroup(M("B", 52, FRAME_C), T("768 × 768", 24, INK_2)).arrange(RIGHT, buff=0.3)
        full_lab.next_to(full, UP, buff=0.2)
        zoom_box = Rectangle(width=3.6 * 192 / 768, height=3.6 * 16 / 768, stroke_color=TRAIN_C, stroke_width=2)
        zoom_box.move_to(full.get_corner(UL) + np.array([3.6 * 96 / 768, -3.6 * 8 / 768, 0]))
        self.play(FadeIn(full), FadeIn(full_lab), Create(zoom_box), run_time=0.8)

        self.at(tr, "orth", lead=0.3)
        self.play(FadeOut(top), FadeOut(pair_braces), FadeOut(pl), FadeOut(crop_note), FadeOut(zoom_box),
                  Group(full, full_lab).animate.move_to([-2.6, 0.15, 0]), run_time=0.7)
        orth = M(r"B\,B^{\top} = I", 48, INK).next_to(full, RIGHT, buff=0.8)
        card = term_card("Orthogonal matrix", "Its rows are perpendicular and of length one, so it can rotate space "
                                              "but never stretches it.")
        self.play(Write(orth), run_time=0.6)
        self.show_term(card)
        self.finish(tr, pad=0.6)
        self.clear_all()
