from manim import *
import math

# ==============================================================================
# CONFIGURATION: Magic Numbers, Colors, Timing, Camera Settings
# ==============================================================================

CONFIG = {
    # Colors
    "color_original": WHITE,
    "color_orthogonal": GREEN,
    "color_orthonormal": YELLOW,
    "color_projection": "#00D4FF",  # Cyan
    "color_highlight": "#20B2AA",   # Teal
    "color_eigenvalue_1": RED,
    "color_eigenvalue_2": BLUE,
    "color_eigenvalue_3": "#FF00FF", # Magenta for the 3rd pair
    "color_matrix_Q": YELLOW,
    "color_matrix_R": "#00FFFF",    # Cyan
    "color_matrix_D": ORANGE,
    "color_active_vector": YELLOW,
    "color_dim_vector": GRAY_B,
    
    # Timing (seconds)
    "TRANSITION_TIME": 2.4,
    "pause_intro": 8.0,
    "pause_step": 5.0,
    "pause_result": 3.0,
    "pause_key": 3.0,
    "animation_vector_base": 0.8,
    "animation_camera": 2.0,
    "animation_camera_long": 2.8,
    "subtitle_duration": 3.0,
    "chapter_gap": 0.5,
    "pause_formula_short": 2.4,
    "pause_formula_long": 4.2,
    "pause_scene_detail": 3.6,
    
    # Camera
    "camera_phi_main": 70 * DEGREES,
    "camera_theta_main": -30 * DEGREES,
    "camera_phi_topdown": 15 * DEGREES,
    "camera_theta_topdown": -90 * DEGREES,
    "camera_phi_2d": 0,
    "camera_theta_2d": -90 * DEGREES,
    "camera_breathing_amplitude": 2 * DEGREES,
    "CAMERA_DISTANCE": 12,
    
    # Layout (Zoning)
    "ZONE_TITLE": UP * 3.5,
    "ZONE_SUBTITLE": DOWN * 3.5,
    "ZONE_VISUAL": ORIGIN,
    "ZONE_MATH": UL * 2.5 + LEFT * 1.5,
    
    # Scales, Strokes & Fonts
    "DEFAULT_FONT": "Arial",
    "subtitle_font_size": 24,
    "title_font_size": 40,
    "label_3d_font_size": 42,
    "formula_font_size": 42,
    "MATH_SCALE": 0.6,
    "VEC_STROKE": 6,
    "MATRIX_STROKE": 2.5,
    "INTRO_AXIS_SCALE": 0.6,
    "INTRO_DRAW_TIME": 2.8,
    "VECTOR_DRAW_TIME": 2.2,
    "VECTOR_TRANSFORM_TIME": 2.8,
    "VECTOR_GROW_BASE": 1.2,
    "VECTOR_GROW_SCALE": 0.75,
    "AXIS_GUIDE_STROKE": 2.0,
    "BUFF_MED": 0.5,
    "axes_range": 4,
    "axes_length": 8,
    "axes_padding_factor": 1.35,
}

# ==============================================================================
# PURE PYTHON LINEAR ALGEBRA HELPERS
# ==============================================================================

def dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def norm(v):
    """Euclidean norm of a vector."""
    return math.sqrt(dot(v, v))

def scalar_mul(c, v):
    """Multiply vector by scalar."""
    return [c * x for x in v]

def add_vec(a, b):
    """Add two vectors."""
    return [x + y for x, y in zip(a, b)]

def sub_vec(a, b):
    """Subtract two vectors: a - b."""
    return [x - y for x, y in zip(a, b)]

def proj(u, v):
    """Project vector u onto vector v: proj_v(u)."""
    d_vv = dot(v, v)
    if abs(d_vv) < 1e-9:
        return [0, 0, 0]
    return scalar_mul(dot(u, v) / d_vv, v)

def cross_product(a, b):
    """Cross product for 3D vectors."""
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

def normalize(v):
    """Normalize vector to unit length."""
    n = norm(v)
    if abs(n) < 1e-9:
        return [0, 0, 0]
    return scalar_mul(1/n, v)

def matrix_mult_3x3(A, B):
    """Multiply two 3x3 matrices: result = A @ B."""
    result = [[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    return result

def determinant_3x3(A):
    """Compute determinant of 3x3 matrix."""
    return (A[0][0] * (A[1][1]*A[2][2] - A[1][2]*A[2][1])
          - A[0][1] * (A[1][0]*A[2][2] - A[1][2]*A[2][0])
          + A[0][2] * (A[1][0]*A[2][1] - A[1][1]*A[2][0]))


# ==============================================================================
# ANIMATION UTILITIES & HELPERS
# ==============================================================================

class SubtitleManager:
    """Manages subtitle display to avoid flickering and redundant recreations."""
    def __init__(self, scene):
        self.scene = scene
        self.current_sub = VGroup()
        self.text_obj = None
        self.bg_obj = None
    
    def update(self, text_str, duration=None):
        """Update subtitle text."""
        if self.current_sub in self.scene.mobjects:
            self.scene.remove(self.current_sub)
        
        self.text_obj = Text(text_str, font_size=CONFIG["subtitle_font_size"], color=WHITE, font=CONFIG["DEFAULT_FONT"])
        self.bg_obj = BackgroundRectangle(self.text_obj, color=BLACK, fill_opacity=0.8, buff=0.2)
        self.current_sub = VGroup(self.bg_obj, self.text_obj).to_edge(DOWN, buff=0.2)
        self.scene.add_fixed_in_frame_mobjects(self.current_sub)
    
    def remove(self):
        """Remove current subtitle."""
        if self.current_sub in self.scene.mobjects:
            self.scene.remove(self.current_sub)
        self.current_sub = VGroup()

def get_3d_label(scene, tex, pos, color=WHITE, font_size=None):
    """Create a 3D label that always faces the camera (billboarding)."""
    if font_size is None:
        font_size = CONFIG["label_3d_font_size"]
    
    lbl = MathTex(tex, color=color, font_size=font_size, background_stroke_width=3, background_stroke_color=BLACK)
    
    def face_camera(mob):
        new_mob = MathTex(tex, color=color, font_size=font_size, background_stroke_width=3, background_stroke_color=BLACK)
        new_mob.rotate(scene.camera.get_phi(), axis=RIGHT)
        new_mob.rotate(scene.camera.get_theta() + PI/2, axis=OUT)
        new_mob.move_to(pos)
        mob.become(new_mob)
    
    lbl.add_updater(face_camera)
    return lbl

def get_grow_arrow_anim(start, end, color=WHITE):
    """Create arrow and grow animation from start to end point."""
    arrow = Arrow3D(start=start, end=end, color=color)
    anim = GrowFromPoint(arrow, point=start)
    return anim, arrow

def subtle_pause(scene, duration=3.0, focus=None):
    """Pause with optional focus highlight."""
    if focus is not None:
        effect_time = min(1.2, duration * 0.4)
        scene.play(
            focus.animate.set_color(YELLOW).scale(1.03),
            run_time=effect_time,
            rate_func=there_and_back
        )
        scene.wait(duration - effect_time)
    else:
        scene.wait(duration)

def camera_breathing_effect(scene, center_phi, center_theta, amplitude, duration):
    """Gentle camera oscillation for better pacing (breathing effect)."""
    scene.move_camera(
        phi=center_phi + amplitude,
        theta=center_theta,
        run_time=duration/2,
        rate_func=there_and_back
    )
    scene.move_camera(
        phi=center_phi - amplitude,
        theta=center_theta,
        run_time=duration/2,
        rate_func=there_and_back
    )

class MatrixProjectScene(ThreeDScene):
    def setup(self):
        super().setup()
        self.sub_mgr = SubtitleManager(self)
        self.zones = {
            "title": CONFIG["ZONE_TITLE"],
            "subtitle": CONFIG["ZONE_SUBTITLE"],
            "visual": CONFIG["ZONE_VISUAL"],
            "math": CONFIG["ZONE_MATH"]
        }
        self.setup_layout()

    def setup_layout(self):
        """Hidden boundaries to help visual consistency."""
        # Optional: Add debug boxes if needed by setting opacity > 0
        self.title_boundary = Line(LEFT*7, RIGHT*7).move_to(self.zones["title"]).set_stroke(opacity=0)
        self.subtitle_boundary = Line(LEFT*7, RIGHT*7).move_to(self.zones["subtitle"]).set_stroke(opacity=0)
        self.add(self.title_boundary, self.subtitle_boundary)

    def smart_add(self, mobject, zone="visual", is_fixed=True):
        """Move mobject to specified zone and add to scene."""
        target_pos = self.zones.get(zone, ORIGIN)
        if zone == "math":
            mobject.scale(CONFIG["MATH_SCALE"]).move_to(target_pos)
        else:
            mobject.move_to(target_pos)
            
        if is_fixed:
            self.add_fixed_in_frame_mobjects(mobject)
        else:
            self.add(mobject)
        return mobject

    def smart_play(self, *anims, zone="visual", **kwargs):
        """Wrapper for play to ensure consistent timing and zoning if added during play."""
        if "run_time" not in kwargs:
            kwargs["run_time"] = CONFIG["TRANSITION_TIME"]
        self.play(*anims, **kwargs)

    def clear_visual_zone(self):
        """Clears center stage but keeps Title/Subtitle/Math reference."""
        mobs_to_remove = []
        for mob in self.mobjects:
            # Simple heuristic: remove if near visual center or if not in Title/Subtitle zone
            pos_y = mob.get_center()[1]
            if -3 < pos_y < 3:
                mobs_to_remove.append(mob)
        if mobs_to_remove:
            self.play(*[FadeOut(m) for m in mobs_to_remove], run_time=1)

    def get_billboard_label(self, tex, pos, color=WHITE, font_size=None):
        """Create a 3D label that always faces the camera."""
        if font_size is None:
            font_size = CONFIG["label_3d_font_size"]
        
        lbl = MathTex(tex, color=color, font_size=font_size, background_stroke_width=3, background_stroke_color=BLACK)
        lbl_base = lbl.copy()
        
        def face_camera_updater(mob):
            phi = self.camera.get_phi()
            theta = self.camera.get_theta()
            # Use a copy of the base to avoid cumulative rotations
            new_mob = lbl_base.copy()
            new_mob.rotate(phi, axis=RIGHT)
            new_mob.rotate(theta + PI/2, axis=OUT)
            new_mob.move_to(pos)
            mob.become(new_mob)
            
        lbl.add_updater(face_camera_updater)
        return lbl

    def shrink_and_move(self, mobject, target_zone="math"):
        """Transition logic to prevent overlap when moving to reference corner."""
        self.play(
            mobject.animate.scale(CONFIG["MATH_SCALE"]).move_to(self.zones[target_zone]),
            run_time=CONFIG["TRANSITION_TIME"]
        )

    def draw_detailed_3d_vector(self, axes, coords, color=WHITE, label=None):
        """Step-by-step 3D vector construction."""
        x, y, z = coords
        origin = axes.get_origin()
        
        # 1. Axis Guides + Marks on Ox, Oy, Oz
        guide_x = DashedLine(axes.c2p(0, 0, 0), axes.c2p(x, 0, 0), color=GRAY, stroke_width=CONFIG["AXIS_GUIDE_STROKE"])
        guide_y = DashedLine(axes.c2p(0, 0, 0), axes.c2p(0, y, 0), color=GRAY, stroke_width=CONFIG["AXIS_GUIDE_STROKE"])
        guide_z = DashedLine(axes.c2p(0, 0, 0), axes.c2p(0, 0, z), color=GRAY, stroke_width=CONFIG["AXIS_GUIDE_STROKE"])

        dot_x = Dot3D(axes.c2p(x, 0, 0), color=color, radius=0.08)
        dot_y = Dot3D(axes.c2p(0, y, 0), color=color, radius=0.08)
        dot_z = Dot3D(axes.c2p(0, 0, z), color=color, radius=0.08)
        
        self.play(Create(guide_x), Create(dot_x), run_time=1.3)
        self.play(Create(guide_y), Create(dot_y), run_time=1.3)
        self.play(Create(guide_z), Create(dot_z), run_time=1.3)
        self.wait(1.0)

        # 2. Base Plane Projection and frame construction
        line_x = DashedLine(axes.c2p(x, 0, 0), axes.c2p(x, y, 0), color=GRAY)
        line_y = DashedLine(axes.c2p(0, y, 0), axes.c2p(x, y, 0), color=GRAY)
        self.play(Create(line_x), Create(line_y), run_time=1.8)
        
        # 3. Vertical Projection
        line_z = DashedLine(axes.c2p(x, y, 0), axes.c2p(x, y, z), color=GRAY)
        line_ztop = DashedLine(axes.c2p(0, 0, z), axes.c2p(x, y, z), color=GRAY)
        self.play(Create(line_z), run_time=1.4)
        self.play(Create(line_ztop), run_time=1.4)
        self.wait(1.2)

        # 4. Final Arrow
        arrow = Arrow3D(start=origin, end=axes.c2p(x, y, z), color=color)
        vec_length = norm([x, y, z])
        dynamic_run_time = max(CONFIG["VECTOR_GROW_BASE"], vec_length * CONFIG["VECTOR_GROW_SCALE"])
        grow_anim = GrowFromPoint(arrow, point=origin)
        if label:
            lbl = self.get_billboard_label(label, axes.c2p(x, y, z) + OUT*0.3, color=color)
            self.play(grow_anim, FadeIn(lbl), run_time=dynamic_run_time)
            return arrow, lbl, VGroup(guide_x, guide_y, guide_z, dot_x, dot_y, dot_z, line_x, line_y, line_z, line_ztop)
        else:
            self.play(grow_anim, run_time=dynamic_run_time)
            return arrow, VGroup(guide_x, guide_y, guide_z, dot_x, dot_y, dot_z, line_x, line_y, line_z, line_ztop)

    def set_vector_focus(self, active_arrow, other_arrows):
        """Highlight active vector and dim the remaining vectors."""
        anims = [
            active_arrow.animate.set_opacity(1.0).set_color(CONFIG["color_active_vector"]) 
        ]
        for arr in other_arrows:
            if arr is not None:
                anims.append(arr.animate.set_opacity(0.25).set_color(CONFIG["color_dim_vector"]))
        self.play(*anims, run_time=0.7)

    def transform_formula(self, source, target):
        """Create a smoother formula morph with arc motion."""
        self.play(
            TransformMatchingTex(source, target, path_arc=PI / 4),
            run_time=2.5,
            rate_func=smooth
        )

# ==============================================================================
# MAIN SCENE CLASS
# ==============================================================================

class QRAndDiagonalization(MatrixProjectScene):
    def get_dynamic_axes_spec(self, vectors, min_range=None):
        """Compute symmetric axis ranges from data vectors to avoid clipping."""
        if min_range is None:
            min_range = CONFIG["axes_range"]
        max_abs = max(abs(c) for vec in vectors for c in vec)
        bound = max(min_range, int(math.ceil(max_abs * CONFIG["axes_padding_factor"])))
        axis_step = 1
        return {
            "bound": bound,
            "x_range": [-bound, bound, axis_step],
            "y_range": [-bound, bound, axis_step],
            "z_range": [-bound, bound, axis_step],
            "axis_length": max(CONFIG["axes_length"], 2 * bound),
        }

    def get_tracking_3d_label(self, tex, arrow, color=WHITE, offset=OUT * 0.25, font_size=None):
        """Always-redraw billboard label that tracks the current arrow endpoint."""
        if font_size is None:
            font_size = CONFIG["label_3d_font_size"]

        def build_label():
            lbl = MathTex(
                tex,
                color=color,
                font_size=font_size,
                background_stroke_width=3,
                background_stroke_color=BLACK,
            )
            lbl.rotate(self.camera.get_phi(), axis=RIGHT)
            lbl.rotate(self.camera.get_theta() + PI / 2, axis=OUT)
            lbl.move_to(arrow.get_end() + offset)
            return lbl

        return always_redraw(build_label)

    def show_projection_formula_panel(self):
        """Display and return the generic projection formula panel."""
        formula = MathTex(
            r"\operatorname{proj}_{\mathbf{v}}(\mathbf{u})",
            r"=",
            r"\frac{\langle \mathbf{u},\mathbf{v}\rangle}{\|\mathbf{v}\|^2}\,\mathbf{v}",
            font_size=36,
            color=CONFIG["color_projection"],
        )
        formula.to_corner(UR, buff=0.45)
        self.add_fixed_in_frame_mobjects(formula)
        self.play(Write(formula), run_time=1.7)
        return formula

    def show_orthogonal_formula(self, formula_tex):
        """Show an orthogonalization formula near math zone with pacing-aware wait."""
        formula = MathTex(formula_tex, font_size=34, color=YELLOW)
        formula.to_edge(DOWN, buff=0.95)
        self.add_fixed_in_frame_mobjects(formula)
        self.play(FadeIn(formula, shift=UP * 0.2), run_time=1.4)
        return formula

    def explain_r_upper_triangular(self, a_cols, q_cols, anchor_title):
        """Add a focused explanation of the upper-triangular structure of R."""
        r11 = dot(a_cols[0], q_cols[0])
        r12 = dot(a_cols[1], q_cols[0])
        r13 = dot(a_cols[2], q_cols[0])
        r22 = dot(a_cols[1], q_cols[1])
        r23 = dot(a_cols[2], q_cols[1])
        r33 = dot(a_cols[2], q_cols[2])

        r_rule = MathTex(
            r"R_{j,i}=\langle a_i,q_j\rangle\;(j\le i),\quad R_{j,i}=0\;(j>i)",
            font_size=34,
            color=CONFIG["color_matrix_R"],
        )
        r_rule.next_to(anchor_title, DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(r_rule)

        r_symbolic = MathTex(
            r"R=\begin{pmatrix}",
            r"\langle a_1,q_1\rangle & \langle a_2,q_1\rangle & \langle a_3,q_1\rangle \\",
            r"0 & \langle a_2,q_2\rangle & \langle a_3,q_2\rangle \\",
            r"0 & 0 & \langle a_3,q_3\rangle",
            r"\end{pmatrix}",
            font_size=30,
        )
        r_symbolic.to_corner(DR, buff=0.45)
        self.add_fixed_in_frame_mobjects(r_symbolic)

        r_numeric = Matrix([
            [f"{r11:.2f}", f"{r12:.2f}", f"{r13:.2f}"],
            ["0", f"{r22:.2f}", f"{r23:.2f}"],
            ["0", "0", f"{r33:.2f}"],
        ], left_bracket="(", right_bracket=")").scale(0.66)
        r_group = VGroup(MathTex("R =", color=CONFIG["color_matrix_R"]), r_numeric).arrange(RIGHT, buff=0.18)
        r_group.to_corner(DL, buff=0.45)
        self.add_fixed_in_frame_mobjects(r_group)

        self.play(FadeIn(r_rule, shift=DOWN * 0.2), run_time=1.2)
        self.play(FadeIn(r_symbolic, shift=LEFT * 0.2), run_time=1.5)
        self.wait(CONFIG["pause_formula_long"])
        self.play(FadeIn(r_group, shift=UP * 0.2), run_time=1.4)
        self.wait(CONFIG["pause_formula_long"])

        upper_triangle = VGroup(*[
            r_numeric.get_entries()[0], r_numeric.get_entries()[1], r_numeric.get_entries()[2],
            r_numeric.get_entries()[4], r_numeric.get_entries()[5], r_numeric.get_entries()[8],
        ])
        self.play(Indicate(upper_triangle, color=CONFIG["color_matrix_R"], scale_factor=1.08), run_time=1.6)
        self.wait(CONFIG["pause_formula_short"])
        return VGroup(r_rule, r_symbolic, r_group)

    def show_row_echelon_steps(self, start_tex, ref_tex, conclusion_tex, color):
        """Pedagogical chain for (A-lambda I)v=0 -> REF -> eigenspace conclusion."""
        eq_start = MathTex(start_tex, font_size=38).move_to(UP * 0.15)
        eq_ref = MathTex(ref_tex, font_size=34).move_to(UP * 0.15)
        eq_conclusion = MathTex(conclusion_tex, font_size=36, color=color).move_to(UP * 0.15)

        self.play(Write(eq_start), run_time=1.2)
        self.wait(CONFIG["pause_formula_short"])
        self.play(ReplacementTransform(eq_start, eq_ref), run_time=1.5)
        self.wait(CONFIG["pause_formula_long"])
        self.play(ReplacementTransform(eq_ref, eq_conclusion), run_time=1.5)
        self.wait(CONFIG["pause_formula_short"])
        return eq_conclusion

    def construct(self):
        """Main scene construction - structured by Chapters."""
        self.intro_chapter()
        self.wait(CONFIG["chapter_gap"])
        
        self.qr_chapter()
        self.wait(CONFIG["chapter_gap"])
        
        self.diagonal_chapter()
        self.wait(CONFIG["chapter_gap"])
        
        self.verification_chapter()
        self.wait(CONFIG["chapter_gap"])
        
        self.closing()

    def intro_chapter(self):
        """Chapter 1: Geometric Nature of Linear Algebra."""
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        
        # 1. Setup Title
        title = Text("Ý nghĩa hình học của Ma trận", font_size=CONFIG["title_font_size"], font=CONFIG["DEFAULT_FONT"])
        title.move_to(self.zones["title"])
        self.smart_play(Write(title))
        
        self.sub_mgr.update("Một vector trong 2D đơn giản là một cặp số (x, y).")
        
        # 2. Axes and Grid
        axes_2d = Axes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=7, y_length=7,
            axis_config={"color": GRAY},
            tips=False
        ).scale(CONFIG["INTRO_AXIS_SCALE"])
        grid = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=7, y_length=7,
            background_line_style={"stroke_color": GRAY, "stroke_width": 1, "stroke_opacity": 0.2}
        ).scale(CONFIG["INTRO_AXIS_SCALE"])
        axes_2d.move_to(self.zones["visual"])
        grid.move_to(self.zones["visual"])
        self.play(Create(grid), Create(axes_2d), run_time=CONFIG["INTRO_DRAW_TIME"])

        label_o = Text("O", font_size=24, font=CONFIG["DEFAULT_FONT"]).next_to(axes_2d.get_origin(), DL, buff=0.08)
        label_x = Text("x", font_size=24, font=CONFIG["DEFAULT_FONT"]).next_to(axes_2d.c2p(4, 0), DR, buff=0.1)
        label_y = Text("y", font_size=24, font=CONFIG["DEFAULT_FONT"]).next_to(axes_2d.c2p(0, 4), UL, buff=0.1)
        self.play(FadeIn(label_o), FadeIn(label_x), FadeIn(label_y), run_time=1.5)
        
        # 3. Vector v
        v_end = [1, 1, 0]
        arrow_v = Arrow(axes_2d.get_origin(), axes_2d.c2p(1, 1), buff=0, color=CONFIG["color_original"], stroke_width=CONFIG["VEC_STROKE"])
        label_v = MathTex("v = (1, 1)", color=CONFIG["color_original"], font_size=30, background_stroke_width=2, background_stroke_color=BLACK).next_to(arrow_v, UR, buff=0.1)
        guide_vx = DashedLine(axes_2d.c2p(1, 1), axes_2d.c2p(1, 0), color=GRAY_B)
        guide_vy = DashedLine(axes_2d.c2p(1, 1), axes_2d.c2p(0, 1), color=GRAY_B)
        
        self.play(GrowArrow(arrow_v), FadeIn(label_v), run_time=CONFIG["VECTOR_DRAW_TIME"])
        self.play(Create(guide_vx), Create(guide_vy), run_time=1.6)
        self.wait(CONFIG["pause_key"])
        
        # 4. Matrix A Reference
        self.sub_mgr.update("Ma trận là một bảng số thực hiện phép biến đổi không gian.")
        matrix_A = Matrix([[2, 1], [1, 2]], left_bracket="(", right_bracket=")")
        matrix_A.set_stroke(width=CONFIG["MATRIX_STROKE"])
        matrix_A.get_entries().set_stroke(width=CONFIG["MATRIX_STROKE"])
        matrix_group = VGroup(MathTex("A = "), matrix_A).arrange(RIGHT).scale(CONFIG["MATH_SCALE"])
        matrix_group[0].set_stroke(width=CONFIG["MATRIX_STROKE"])
        matrix_group.move_to(self.zones["math"])
        self.smart_play(FadeIn(matrix_group))
        self.wait(CONFIG["pause_key"])
        
        # 5. Transformation
        self.sub_mgr.update("Khi nhân ma trận A với v, diện mạo không gian và vector bị thay đổi.")
        matrix_transform = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 1]])
        guide_Avx = DashedLine(axes_2d.c2p(3, 3), axes_2d.c2p(3, 0), color=GRAY_B)
        guide_Avy = DashedLine(axes_2d.c2p(3, 3), axes_2d.c2p(0, 3), color=GRAY_B)
        label_Av = MathTex("Av = (3, 3)", color=CONFIG["color_highlight"], font_size=30, background_stroke_width=2, background_stroke_color=BLACK).move_to(axes_2d.c2p(3, 3) + UR*0.25)
        
        self.play(
            ApplyMatrix(matrix_transform, grid),
            ApplyMatrix(matrix_transform, arrow_v),
            AnimationGroup(
                ReplacementTransform(guide_vx, guide_Avx),
                ReplacementTransform(guide_vy, guide_Avy),
                lag_ratio=0.0
            ),
            run_time=CONFIG["VECTOR_TRANSFORM_TIME"]
        )
        self.play(TransformMatchingTex(label_v, label_Av), run_time=1.6)
        self.wait(CONFIG["pause_key"])
        
        # 6. Linear Combination Explanation
        self.sub_mgr.update("Nhân với ma trận là tổ hợp tuyến tính của các cột: Av = x·c1 + y·c2.")
        theory_text = MathTex(
            r"A \mathbf{v} = 1 \cdot \begin{pmatrix} 2 \\ 1 \end{pmatrix} + 1 \cdot \begin{pmatrix} 1 \\ 2 \end{pmatrix}",
            " = \\begin{pmatrix} 3 \\\\ 3 \\end{pmatrix}",
            font_size=34,
            background_stroke_width=2,
            background_stroke_color=BLACK
        ).next_to(matrix_group, DOWN, buff=0.5)
        self.play(Write(theory_text))
        self.wait(CONFIG["pause_key"])
        
        # 7. Cleanup
        self.play(
            AnimationGroup(
                FadeOut(arrow_v), FadeOut(label_Av), FadeOut(guide_Avx), FadeOut(guide_Avy),
                FadeOut(matrix_group), FadeOut(theory_text), FadeOut(grid), FadeOut(axes_2d),
                FadeOut(label_o), FadeOut(label_x), FadeOut(label_y),
                FadeOut(title),
                lag_ratio=0.04
            ),
            run_time=1.8
        )
    
    # =========================================================================
    # SCENE 2: QR DECOMPOSITION OVERVIEW (~3-4 min)
    # =========================================================================
    
    def qr_chapter(self):
        """Chapter 2: QR Decomposition via Gram-Schmidt."""
        # 1. Title
        title = Text("Phân rã QR: Trực quan hóa Gram-Schmidt", font_size=CONFIG["title_font_size"], font=CONFIG["DEFAULT_FONT"])
        title.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(title)
        self.smart_play(Write(title))
        
        # 2. QR Theory Intro
        self.sub_mgr.update("Mục tiêu: Tách A thành Q (Xoay/Trực chuẩn) và R (Nén/Tam giác).")
        qr_logic = VGroup(
            Text("Q: Ma trận trực giao (Rotation/Orthonormal)", font=CONFIG["DEFAULT_FONT"], font_size=24, color=CONFIG["color_matrix_Q"]),
            Text("R: Ma trận tam giác trên (Compression/Scaling)", font=CONFIG["DEFAULT_FONT"], font_size=24, color=CONFIG["color_matrix_R"])
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).shift(UP*0.5)
        self.add_fixed_in_frame_mobjects(qr_logic)
        self.play(FadeIn(qr_logic, shift=UP))
        self.wait(2)
        
        self.sub_mgr.update("Ý nghĩa: 'Bẻ thẳng' các vector ban đầu cho vuông góc với nhau.")
        self.wait(CONFIG["pause_key"])
        self.play(FadeOut(qr_logic))

        # Hide chapter title before 3D stage to avoid covering axis labels.
        self.play(FadeOut(title), run_time=0.8)

        # 3. Keep only matrix A reference in the corner.
        a_matrix_ref = Matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]], left_bracket="(", right_bracket=")").scale(0.42)
        a_matrix_group = VGroup(MathTex("A = "), a_matrix_ref).arrange(RIGHT)
        a_matrix_group.to_corner(UL, buff=0.45)
        self.add_fixed_in_frame_mobjects(a_matrix_group)
        self.play(FadeIn(a_matrix_group), run_time=1.2)
        current_col_rect = None

        def focus_column(col_idx):
            nonlocal current_col_rect
            target_rect = SurroundingRectangle(a_matrix_ref.get_columns()[col_idx], color=YELLOW, buff=0.06)
            if current_col_rect is None:
                current_col_rect = target_rect
                self.add_fixed_in_frame_mobjects(current_col_rect)
                self.play(Create(current_col_rect), run_time=0.5)
            else:
                self.play(Transform(current_col_rect, target_rect), run_time=0.5)
        
        # 4. 3D Gram-Schmidt
        self.sub_mgr.update("Sử dụng Gram-Schmidt để tìm các vector trực chuẩn q1, q2, q3.")
        
        # Setup 3D
        self.set_camera_orientation(phi=CONFIG["camera_phi_main"], theta=CONFIG["camera_theta_main"])
        a1_val = [2, 1, 1]
        a2_val = [1, 2, 1]
        a3_val = [1, 1, 2]
        axis_spec = self.get_dynamic_axes_spec([a1_val, a2_val, a3_val])
        axes = ThreeDAxes(
            x_range=axis_spec["x_range"],
            y_range=axis_spec["y_range"],
            z_range=axis_spec["z_range"],
            x_length=axis_spec["axis_length"],
            y_length=axis_spec["axis_length"],
            z_length=axis_spec["axis_length"],
            axis_config={"color": GRAY},
        )
        self.play(Create(axes), run_time=2.6)

        axis_bound = axis_spec["bound"]
        axis_label_x = self.get_billboard_label("x", axes.c2p(axis_bound + 0.3, 0, 0), color=WHITE, font_size=30)
        axis_label_y = self.get_billboard_label("y", axes.c2p(0, axis_bound + 0.3, 0), color=WHITE, font_size=30)
        axis_label_z = self.get_billboard_label("z", axes.c2p(0, 0, axis_bound + 0.3), color=WHITE, font_size=30)
        axis_label_o = self.get_billboard_label("O", axes.c2p(0, 0, 0) + LEFT * 0.2 + DOWN * 0.2, color=WHITE, font_size=28)
        axis_labels = VGroup(axis_label_x, axis_label_y, axis_label_z, axis_label_o)
        self.play(*[FadeIn(lbl) for lbl in axis_labels], run_time=1.4)

        proj_formula = self.show_projection_formula_panel()
        self.wait(CONFIG["pause_formula_long"])

        # Data from A = [[2,1,1],[1,2,1],[1,1,2]]
        q1_val = normalize(a1_val)
        
        # Step 1: Detailed construction of a1
        self.sub_mgr.update("Bước 1: Vẽ vector a1 và chuẩn hóa để được q1.")
        focus_column(0)
        a1_arrow, a1_label, a1_construction = self.draw_detailed_3d_vector(axes, a1_val, color=CONFIG["color_original"], label="a_1")
        self.set_vector_focus(a1_arrow, [])
        self.wait(CONFIG["pause_key"])
        
        q1_arrow = Arrow3D(start=ORIGIN, end=axes.c2p(*q1_val), color=CONFIG["color_matrix_Q"])
        q1_label = self.get_tracking_3d_label("q_1", q1_arrow, color=CONFIG["color_matrix_Q"], offset=DOWN * 0.55)
        
        self.play(
            ReplacementTransform(a1_arrow.copy(), q1_arrow),
            FadeIn(q1_label),
            a1_arrow.animate.set_opacity(0.3),
            FadeOut(a1_construction)
        )
        self.set_vector_focus(q1_arrow, [a1_arrow])
        self.wait(CONFIG["pause_key"])
        
        # Step 2: Orthogonalize a2
        self.sub_mgr.update("Bước 2: Tìm phần trực giao của a2 so với q1, sau đó chuẩn hóa.")
        focus_column(1)
        a2_arrow, a2_label, a2_construction = self.draw_detailed_3d_vector(axes, a2_val, color=CONFIG["color_original"], label="a_2")
        self.set_vector_focus(a2_arrow, [a1_arrow, q1_arrow])

        e2_formula = self.show_orthogonal_formula(r"e_2 = a_2 - \operatorname{proj}_{q_1}(a_2),\quad q_2=\frac{e_2}{\|e_2\|}")
        self.wait(CONFIG["pause_formula_long"])
        
        # Projection lines
        p_val = proj(a2_val, q1_val)
        proj_line = DashedLine(axes.c2p(*p_val), axes.c2p(*a2_val), color=GRAY)
        self.play(Create(proj_line), run_time=1.8)
        
        u2_val = sub_vec(a2_val, p_val)
        q2_val = normalize(u2_val)
        q2_arrow = Arrow3D(start=ORIGIN, end=axes.c2p(*q2_val), color=CONFIG["color_matrix_Q"])
        q2_label = self.get_tracking_3d_label("q_2", q2_arrow, color=CONFIG["color_matrix_Q"], offset=RIGHT * 0.3)
        self.play(Create(q2_arrow), FadeIn(q2_label), FadeOut(a2_construction), run_time=2.0)
        self.set_vector_focus(q2_arrow, [a1_arrow, a2_arrow, q1_arrow])
        self.play(FadeOut(e2_formula), run_time=0.9)
        self.wait(CONFIG["pause_key"])

        self.sub_mgr.update("Bước 3: Tách a3 thành phần nằm trên mặt phẳng q1-q2 và phần trực giao để dựng q3.")
        focus_column(2)
        a3_arrow, a3_label, a3_construction = self.draw_detailed_3d_vector(axes, a3_val, color=CONFIG["color_original"], label="a_3")
        self.set_vector_focus(a3_arrow, [a1_arrow, a2_arrow, q1_arrow, q2_arrow])
        e3_formula = self.show_orthogonal_formula(r"e_3 = a_3 - \operatorname{proj}_{q_1}(a_3) - \operatorname{proj}_{q_2}(a_3)")
        self.wait(CONFIG["pause_formula_long"])

        p31 = proj(a3_val, q1_val)
        p32 = proj(a3_val, q2_val)
        proj_plane_val = add_vec(p31, p32)
        u3_val = sub_vec(sub_vec(a3_val, p31), p32)

        plane_scale = 1.45
        span_plane = Polygon(
            axes.c2p(*add_vec(scalar_mul(plane_scale, q1_val), scalar_mul(plane_scale, q2_val))),
            axes.c2p(*add_vec(scalar_mul(-plane_scale, q1_val), scalar_mul(plane_scale, q2_val))),
            axes.c2p(*add_vec(scalar_mul(-plane_scale, q1_val), scalar_mul(-plane_scale, q2_val))),
            axes.c2p(*add_vec(scalar_mul(plane_scale, q1_val), scalar_mul(-plane_scale, q2_val))),
            color=CONFIG["color_projection"],
            fill_color=CONFIG["color_projection"],
            fill_opacity=0.18,
            stroke_opacity=0.45,
            stroke_width=2,
        )

        plane_arrow = Arrow3D(start=ORIGIN, end=axes.c2p(*proj_plane_val), color=GRAY_B)
        plane_label = self.get_billboard_label("proj_{q_1,q_2}(a_3)", axes.c2p(*proj_plane_val) + LEFT * 0.25, color=GRAY_B, font_size=30)
        residual_line = DashedLine(axes.c2p(*proj_plane_val), axes.c2p(*a3_val), color=CONFIG["color_projection"])
        u3_arrow = Arrow3D(start=ORIGIN, end=axes.c2p(*u3_val), color=CONFIG["color_projection"])
        u3_label = self.get_tracking_3d_label("u_3", u3_arrow, color=CONFIG["color_projection"], offset=RIGHT * 0.2)
        self.play(FadeIn(span_plane), Create(plane_arrow), FadeIn(plane_label), Create(residual_line), run_time=1.8)
        self.play(Create(u3_arrow), FadeIn(u3_label), run_time=1.8)

        q3_val = normalize(u3_val)
        q3_arrow = Arrow3D(start=ORIGIN, end=axes.c2p(*q3_val), color=CONFIG["color_matrix_Q"])
        q3_label = self.get_tracking_3d_label("q_3", q3_arrow, color=CONFIG["color_matrix_Q"], offset=LEFT * 0.25)
        self.play(
            ReplacementTransform(u3_arrow, q3_arrow),
            FadeIn(q3_label),
            a3_arrow.animate.set_opacity(0.3),
            FadeOut(u3_label),
            FadeOut(e3_formula),
            FadeOut(plane_arrow),
            FadeOut(plane_label),
            FadeOut(span_plane),
            FadeOut(residual_line),
            FadeOut(a3_construction),
            run_time=2.2
        )
        self.set_vector_focus(q3_arrow, [a1_arrow, a2_arrow, a3_arrow, q1_arrow, q2_arrow])
        self.wait(CONFIG["pause_scene_detail"])

        self.sub_mgr.update("Ma trận R lưu các tích vô hướng ở tam giác trên, còn phía dưới đường chéo là 0.")
        self.move_camera(
            phi=CONFIG["camera_phi_topdown"],
            theta=CONFIG["camera_theta_topdown"],
            run_time=CONFIG["animation_camera_long"],
        )
        r_overlay = self.explain_r_upper_triangular(
            [a1_val, a2_val, a3_val],
            [q1_val, q2_val, q3_val],
            a_matrix_group,
        )
        self.move_camera(
            phi=CONFIG["camera_phi_main"],
            theta=CONFIG["camera_theta_main"],
            run_time=CONFIG["animation_camera_long"],
        )
        self.play(FadeOut(r_overlay), FadeOut(proj_formula), run_time=1.1)
        
        # Cleanup
        self.play(
            AnimationGroup(
                FadeOut(axes), FadeOut(q1_arrow), FadeOut(q2_arrow), FadeOut(q3_arrow),
                FadeOut(a1_arrow), FadeOut(a2_arrow), FadeOut(a3_arrow),
                FadeOut(q1_label), FadeOut(q2_label), FadeOut(q3_label),
                FadeOut(a1_label), FadeOut(a2_label), FadeOut(a3_label), FadeOut(proj_line),
                FadeOut(axis_labels), FadeOut(a_matrix_group), FadeOut(current_col_rect),
                lag_ratio=0.03
            ),
            run_time=2.0
        )
    
    def diagonal_chapter(self):
        """Chapter 3: Diagonalization - A = PDP⁻¹."""
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        title = Text("Quy trình chéo hóa", font_size=CONFIG["title_font_size"], font=CONFIG["DEFAULT_FONT"])
        title.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(title)
        self.smart_play(Write(title))

        fast_wait = 1.6
        fast_rt = 1.5

        self.sub_mgr.update("Đầu tiên ta giải phương trình đặc trưng để tìm các trị riêng.")
        eq1 = MathTex(r"\det(A - \lambda I) = 0", font_size=46).shift(UP * 1.4)
        eq2 = MathTex(r"\begin{vmatrix} 2-\lambda & 1 & 1 \\ 1 & 2-\lambda & 1 \\ 1 & 1 & 2-\lambda \end{vmatrix} = 0", font_size=40).shift(UP * 1.4)
        eq3 = MathTex(r"(2-\lambda)^3 - 3(2-\lambda) + 2 = 0", font_size=40).shift(UP * 1.4)
        eq4 = MathTex(r"-(\lambda - 4)(\lambda - 1)^2 = 0", font_size=46).shift(UP * 1.4)
        eq_lambda = MathTex(
            r"\lambda_1 = ", "4", r",\; \lambda_2 = ", "1", r",\; \lambda_3 = ", "1",
            font_size=40,
            color=YELLOW
        ).shift(UP * 1.4)

        self.play(Write(eq1))
        self.wait(fast_wait)
        self.play(ReplacementTransform(eq1, eq2), run_time=fast_rt)
        self.wait(fast_wait)
        self.play(ReplacementTransform(eq2, eq3), run_time=fast_rt)
        self.wait(fast_wait)
        self.play(ReplacementTransform(eq3, eq4), run_time=fast_rt)
        self.wait(fast_wait)
        self.play(ReplacementTransform(eq4, eq_lambda), run_time=fast_rt)
        self.wait(0.8)
        self.play(eq_lambda.animate.next_to(title, DOWN, buff=0.35), run_time=1.2)

        self.sub_mgr.update("Với λ1 = 4, ta đi qua dạng bậc thang để thấy rõ ràng điều kiện x=y=z.")
        eq_l1_solution = self.show_row_echelon_steps(
            r"(A - 4I)\mathbf{v}=0",
            r"\left[\begin{array}{ccc|c}-2&1&1&0\\1&-2&1&0\\1&1&-2&0\end{array}\right]\sim"
            r"\left[\begin{array}{ccc|c}1&0&-1&0\\0&1&-1&0\\0&0&0&0\end{array}\right]",
            r"x=z,\;y=z\Rightarrow x=y=z\Rightarrow \mathbf{v}_1=(1,1,1)",
            CONFIG["color_eigenvalue_1"],
        )

        self.sub_mgr.update("Với λ2 = λ3 = 1, dạng bậc thang cho thấy không gian nghiệm có 2 bậc tự do.")
        self.play(eq_l1_solution.animate.shift(UP * 0.05), run_time=0.6)
        eq_l23_solution = self.show_row_echelon_steps(
            r"(A-I)\mathbf{v}=0",
            r"\left[\begin{array}{ccc|c}1&1&1&0\\1&1&1&0\\1&1&1&0\end{array}\right]\sim"
            r"\left[\begin{array}{ccc|c}1&1&1&0\\0&0&0&0\\0&0&0&0\end{array}\right]",
            r"x+y+z=0\Rightarrow \dim E_{\lambda=1}=2,\;\mathbf{v}_2=(-1,1,0),\;\mathbf{v}_3=(-1,0,1)",
            CONFIG["color_eigenvalue_2"],
        )

        self.sub_mgr.update("Từ đó thu được ba vector riêng để dựng ma trận P.")
        v1_label = MathTex(r"v_1 = ", color=CONFIG["color_eigenvalue_1"])
        v1_vec = Matrix([[1], [1], [1]], left_bracket="(", right_bracket=")")
        v1_vec.get_entries().set_color(CONFIG["color_eigenvalue_1"])
        v1_group = VGroup(v1_label, v1_vec).arrange(RIGHT)

        v2_label = MathTex(r"v_2 = ", color=CONFIG["color_eigenvalue_2"])
        v2_vec = Matrix([[-1], [1], [0]], left_bracket="(", right_bracket=")")
        v2_vec.get_entries().set_color(CONFIG["color_eigenvalue_2"])
        v2_group = VGroup(v2_label, v2_vec).arrange(RIGHT)

        v3_label = MathTex(r"v_3 = ", color=CONFIG["color_eigenvalue_3"])
        v3_vec = Matrix([[-1], [0], [1]], left_bracket="(", right_bracket=")")
        v3_vec.get_entries().set_color(CONFIG["color_eigenvalue_3"])
        v3_group = VGroup(v3_label, v3_vec).arrange(RIGHT)

        vecs_group = VGroup(v1_group, v2_group, v3_group).arrange(RIGHT, buff=0.8).scale(0.9).move_to(ORIGIN + UP * 0.1)
        self.play(FadeOut(eq_l23_solution), FadeIn(vecs_group, shift=UP), run_time=1.4)
        self.wait(fast_wait)

        self.sub_mgr.update("Hiện khung rỗng của P và D, sau đó để vector và trị riêng bay vào vị trí.")
        p_mat = Matrix([[1, -1, -1], [1, 1, 0], [1, 0, 1]], left_bracket="(", right_bracket=")")
        d_mat = Matrix([[4, 0, 0], [0, 1, 0], [0, 0, 1]], left_bracket="(", right_bracket=")")
        p_mat.get_entries().set_opacity(0)
        d_mat.get_entries().set_opacity(0)

        p_group = VGroup(MathTex("P = "), p_mat).arrange(RIGHT)
        d_group = VGroup(MathTex("D = "), d_mat).arrange(RIGHT)
        matrices_group = VGroup(p_group, d_group).arrange(RIGHT, buff=1.0).shift(DOWN * 2.0)

        self.play(
            FadeIn(p_group[0]), FadeIn(p_mat.get_brackets()),
            FadeIn(d_group[0]), FadeIn(d_mat.get_brackets()),
            run_time=1.4
        )
        self.wait(1.0)

        p_cols = p_mat.get_columns()
        p_cols[0].set_color(CONFIG["color_eigenvalue_1"]).set_opacity(0)
        p_cols[1].set_color(CONFIG["color_eigenvalue_2"]).set_opacity(0)
        p_cols[2].set_color(CONFIG["color_eigenvalue_3"]).set_opacity(0)

        self.play(
            TransformFromCopy(v1_vec.get_entries(), p_cols[0].set_opacity(1)),
            run_time=1.1
        )
        self.play(
            TransformFromCopy(v2_vec.get_entries(), p_cols[1].set_opacity(1)),
            run_time=1.1
        )
        self.play(
            TransformFromCopy(v3_vec.get_entries(), p_cols[2].set_opacity(1)),
            run_time=1.1
        )
        self.play(FadeOut(vecs_group), run_time=0.8)
        self.wait(0.6)

        d_diagonals = VGroup(d_mat.get_entries()[0], d_mat.get_entries()[4], d_mat.get_entries()[8])
        d_diagonals[0].set_color(CONFIG["color_eigenvalue_1"]).set_opacity(1)
        d_diagonals[1].set_color(CONFIG["color_eigenvalue_2"]).set_opacity(1)
        d_diagonals[2].set_color(CONFIG["color_eigenvalue_3"]).set_opacity(1)
        d_zeros = VGroup(*[d_mat.get_entries()[i] for i in [1, 2, 3, 5, 6, 7]])
        d_zeros.set_opacity(1)

        self.play(
            TransformFromCopy(eq_lambda[1], d_diagonals[0]),
            TransformFromCopy(eq_lambda[3], d_diagonals[1]),
            TransformFromCopy(eq_lambda[5], d_diagonals[2]),
            FadeIn(d_zeros),
            run_time=1.8
        )
        self.wait(fast_wait)

        self.play(FadeOut(eq_lambda), matrices_group.animate.move_to(UP * 1.2), run_time=1.2)

        final_formula = MathTex(r"A = P D P^{-1}", color=YELLOW, font_size=48).next_to(matrices_group, DOWN, buff=0.5)
        self.play(Write(final_formula), run_time=1.2)
        self.wait(1.0)

        bridge_title = Text("Cầu nối học thuật: Q và P đều là đổi cơ sở", font=CONFIG["DEFAULT_FONT"], font_size=24)
        bridge_title.next_to(title, DOWN, buff=0.9)
        bridge_q = Text("Q: đổi sang cơ sở trực chuẩn (đẹp về hình học)", font=CONFIG["DEFAULT_FONT"], font_size=22, color=CONFIG["color_matrix_Q"])
        bridge_p = Text("P: đổi sang cơ sở vector riêng (đẹp về đại số)", font=CONFIG["DEFAULT_FONT"], font_size=22, color=CONFIG["color_highlight"])
        bridge_group = VGroup(bridge_title, bridge_q, bridge_p).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        bridge_group.move_to(DOWN * 0.2 + LEFT * 1.9)
        self.add_fixed_in_frame_mobjects(bridge_group)
        self.play(FadeIn(bridge_group, shift=RIGHT * 0.2), run_time=1.4)
        self.wait(CONFIG["pause_formula_long"])

        p_copy = p_mat.copy().scale(0.6)
        d_copy = d_mat.copy().scale(0.6)
        p_inv = p_mat.copy().scale(0.6)
        a_eq_label = MathTex("A = ", font_size=40)
        expanded_A_group = VGroup(a_eq_label, p_copy, d_copy, p_inv).arrange(RIGHT, buff=0.1)
        expanded_A_group.next_to(final_formula, DOWN, buff=0.4)
        inv_label = MathTex("^{-1}", font_size=36)
        inv_label.next_to(p_inv.get_brackets()[1], UR, buff=0.05).shift(DOWN * 0.3 + LEFT * 0.1)
        self.play(FadeIn(expanded_A_group, shift=UP), FadeIn(inv_label), run_time=1.4)
        self.wait(fast_wait)

        chain_expr = MathTex(
            r"P^{-1}\mathbf{v}\;\xrightarrow{\;D\;}\;D(P^{-1}\mathbf{v})\;\xrightarrow{\;P\;}\;PDP^{-1}\mathbf{v}",
            font_size=34,
            color=CONFIG["color_matrix_D"],
        ).next_to(expanded_A_group, DOWN, buff=0.3)
        confirm_expr = MathTex(r"PDP^{-1}=A", font_size=44, color=GREEN).move_to(chain_expr)
        self.play(Write(chain_expr), run_time=1.6)
        self.wait(CONFIG["pause_formula_short"])
        self.play(TransformMatchingTex(chain_expr, confirm_expr), run_time=1.6)
        self.wait(CONFIG["pause_formula_short"])

        self.play(
            AnimationGroup(
                FadeOut(title), FadeOut(matrices_group),
                FadeOut(final_formula), FadeOut(expanded_A_group), FadeOut(inv_label),
                FadeOut(confirm_expr), FadeOut(bridge_group),
                lag_ratio=0.03
            ),
            run_time=1.8
        )

    def verification_chapter(self):
        """Chapter 4: Numerical Verification."""
        title = Text("Kiểm chứng kết quả", font_size=CONFIG["title_font_size"], font=CONFIG["DEFAULT_FONT"])
        title.move_to(self.zones["title"])
        self.smart_play(Write(title))
        
        self.sub_mgr.update("Rút gọn kiểm chứng: trước hết tính PD.")
        line_pd = MathTex(
            r"PD = \begin{pmatrix} 4 & -1 & -1 \\ 4 & 1 & 0 \\ 4 & 0 & 1 \end{pmatrix}",
            font_size=34
        ).move_to(UP * 0.8)
        self.play(Write(line_pd), run_time=1.3)
        self.wait(1.3)

        self.sub_mgr.update("Tiếp theo nhân với P^{-1} để quay lại ma trận gốc.")
        line_pdp = MathTex(
            r"(PD)P^{-1} = \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix}",
            font_size=34,
            color=GREEN
        ).move_to(ORIGIN)
        self.play(ReplacementTransform(line_pd.copy(), line_pdp), run_time=1.4)
        self.wait(1.2)

        self.sub_mgr.update("So sánh với A ban đầu: kết quả trùng khớp hoàn toàn.")
        line_a = MathTex(
            r"A = \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix}",
            font_size=34,
            color=YELLOW
        ).move_to(DOWN * 0.9)
        self.play(Write(line_a), run_time=1.3)
        self.wait(CONFIG["pause_key"])

        self.play(FadeOut(title, line_pd, line_pdp, line_a), run_time=1.2)

    def closing(self):
        """Final message."""
        self.sub_mgr.update("Cảm ơn các bạn đã theo dõi bài học!")
        thanks = Text("Cảm ơn các bạn đã theo dõi!", font="Arial", font_size=40)
        thanks.move_to(self.zones["visual"])
        self.smart_play(Write(thanks))
        self.wait(CONFIG["pause_key"])
        self.play(FadeOut(thanks))
        self.sub_mgr.remove()