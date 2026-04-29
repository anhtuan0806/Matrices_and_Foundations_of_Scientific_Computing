from manim import *
import math
import numpy as np
from typing import Union, List, Dict, Optional, Any, Tuple
import re

# ==============================================================================
# CẤU HÌNH: Các hằng số, màu sắc, thời gian và thiết lập camera
# ==============================================================================

CONFIG = {
    # Màu sắc
    "color_original": WHITE,
    "color_orthogonal": GREEN,
    "color_orthonormal": YELLOW,
    "color_projection": "#00D4FF",  # Màu xanh lơ
    "color_highlight": "#20B2AA",   # Màu mòng két
    "color_eigenvalue_1": RED,
    "color_eigenvalue_2": BLUE,
    "color_eigenvalue_3": "#FF00FF", # Màu đỏ cánh sen cho cặp thứ 3
    "color_matrix_Q": YELLOW,
    "color_matrix_R": "#00FFFF",    # Màu xanh lơ
    "color_matrix_D": ORANGE,
    "color_active_vector": YELLOW,
    "color_dim_vector": GRAY_B,
    "color_distorted_space": "#FF6B6B",
    "color_pipeline_arrow": "#FFD700",
    "color_shadow_line": "#808080",
    
    # Thời gian - Đơn vị tính: Giây
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
    "pause_eigen_filter": 4.0,
    "pause_pipeline_step": 3.0,
    "RANDOM_ARROW_COUNT": 24,
    "PIPELINE_STEP_TIME": 2.0,
    
    # Camera
    "camera_phi_main": 70 * DEGREES,
    "camera_theta_main": -30 * DEGREES,
    "camera_phi_topdown": 15 * DEGREES,
    "camera_theta_topdown": -90 * DEGREES,
    "camera_phi_2d": 0,
    "camera_theta_2d": -90 * DEGREES,
    "camera_breathing_amplitude": 2 * DEGREES,
    "CAMERA_DISTANCE": 12,
    
    # Thiết lập các phân vùng hiển thị
    "ZONE_TITLE": UP * 3.5,
    "ZONE_SUBTITLE": DOWN * 3.5,
    "ZONE_VISUAL": ORIGIN,
    "ZONE_MATH": UL * 2.5 + LEFT * 1.5,
    
    # Thông số đồ họa và định dạng văn bản
    "DEFAULT_FONT": "Arial",
    "MATH_FONT": "Cambria Math",
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
# CÁC HÀM TRỢ GIÚP ĐẠI SỐ TUYẾN TÍNH SỬ DỤNG PYTHON THUẦN TÚY
# ==============================================================================

def calculate_dot_product(vector_a: list, vector_b: list) -> float:
    """
    Cung cấp giá trị định lượng cho các phép toán hình chiếu và kiểm tra 
    mức độ trực giao giữa hai thành phần không gian.
    """
    return sum(val_a * val_b for val_a, val_b in zip(vector_a, vector_b))

def calculate_euclidean_norm(vector_v: list) -> float:
    """
    Xác định độ dài hình học của vector để phục vụ việc chuẩn hóa 
    các hướng về kích thước đơn vị.
    """
    return math.sqrt(calculate_dot_product(vector_v, vector_v))

def multiply_vector_by_scalar(scalar_c: float, vector_v: list) -> list:
    """
    Phóng đại hoặc thu nhỏ vector nhằm điều chỉnh tỷ lệ 
    trong các phép toán hình chiếu và trực chuẩn hóa.
    """
    return [scalar_c * val_x for val_x in vector_v]

def add_two_vectors(vector_a: list, vector_b: list) -> list:
    """
    Tổng hợp các thành phần không gian riêng biệt để tạo ra 
    vector kết quả trong các dựng hình hình học 3D.
    """
    return [val_a + val_b for val_a, val_b in zip(vector_a, vector_b)]

def subtract_two_vectors(vector_a: list, vector_b: list) -> list:
    """
    Loại bỏ thành phần hình chiếu của vector_b ra khỏi vector_a, 
    là bước cốt lõi trong thuật toán trực giao hóa Gram-Schmidt.
    """
    return [val_a - val_b for val_a, val_b in zip(vector_a, vector_b)]

def project_vector_onto(vector_u: list, vector_v: list) -> list:
    """
    Tìm thành phần của vector_u nằm theo hướng của vector_v. 
    Hàm tự động xử lý trường hợp vector không để đảm bảo tính ổn định cho hoạt ảnh.
    """
    dot_vv = calculate_dot_product(vector_v, vector_v)
    if abs(dot_vv) < 1e-9:
        return [0.0, 0.0, 0.0]
    return multiply_vector_by_scalar(calculate_dot_product(vector_u, vector_v) / dot_vv, vector_v)

def calculate_cross_product_3d(vector_a: list, vector_b: list) -> list:
    """
    Xác định vector pháp tuyến chung của hai vector trong không gian 3D, 
    giúp xác định các mặt phẳng trực quan hóa.
    """
    return [
        vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1],
        vector_a[2] * vector_b[0] - vector_a[0] * vector_b[2],
        vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
    ]

def normalize_to_unit_vector(vector_v: list) -> list:
    """
    Chuyển đổi một vector trực giao bất kỳ thành vector có độ dài đơn vị, 
    giúp xây dựng các hệ cơ sở trực chuẩn cho ma trận Q.
    """
    magnitude = calculate_euclidean_norm(vector_v)
    if abs(magnitude) < 1e-9:
        return [0.0, 0.0, 0.0]
    return multiply_vector_by_scalar(1.0 / magnitude, vector_v)

def multiply_matrices_3x3(matrix_a: list, matrix_b: list) -> list:
    """
    Thực hiện phép nhân ma trận để kiểm chứng tính đúng đắn 
    của các kết quả phân rã QR hoặc chéo hóa PDP^-1.
    """
    result_matrix = [[0.0] * 3 for _ in range(3)]
    for idx_row in range(3):
        for idx_col in range(3):
            for idx_inner in range(3):
                result_matrix[idx_row][idx_col] += matrix_a[idx_row][idx_inner] * matrix_b[idx_inner][idx_col]
    return result_matrix

def calculate_determinant_3x3(matrix_a: list) -> float:
    """
    Tính giá trị định thức để giải phương trình đặc trưng, 
    từ đó tìm ra các trị riêng phục vụ quá trình chéo hóa ma trận.
    """
    return (matrix_a[0][0] * (matrix_a[1][1] * matrix_a[2][2] - matrix_a[1][2] * matrix_a[2][1])
          - matrix_a[0][1] * (matrix_a[1][0] * matrix_a[2][2] - matrix_a[1][2] * matrix_a[2][0])
          + matrix_a[0][2] * (matrix_a[1][0] * matrix_a[2][1] - matrix_a[1][1] * matrix_a[2][0]))


# ==============================================================================
# CÁC TIỆN ÍCH & HÀM TRỢ GIÚP HOẠT ẢNH
# ==============================================================================

class SubtitleManager:
    """Quản lý việc hiển thị phụ đề để tránh hiện tượng nhấp nháy và khởi tạo lại dư thừa."""
    def __init__(self, scene: Scene):
        self.scene = scene
        self.current_subtitle_group = VGroup()
        self.text_object = None
        self.background_object = None
    
    def update_subtitle_text(self, text_string: str, math_substrings: list = None, duration: float = None) -> None:
        """
        Cập nhật nội dung phụ đề bằng MarkupText để hỗ trợ định dạng toán học.
        Hàm xử lý các lỗi hiển thị ký tự Unicode và chuyển đổi các tag Markup 
        cho chỉ số trên/dưới.
        """
        if self.current_subtitle_group in self.scene.mobjects:
            self.scene.remove(self.current_subtitle_group)
        
        processed_text = text_string
        font_math = CONFIG.get("MATH_FONT", "Cambria Math")
        color_math = "#FFFF00" # YELLOW

        # 1. Xử lý chuẩn LaTeX TRƯỚC để tránh xung đột với các thuộc tính thẻ HTML sau này
        # a_{1} -> a<sub>1</sub>
        processed_text = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", processed_text)
        processed_text = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", processed_text)
        # a_1 -> a<sub>1</sub> (chỉ hỗ trợ 1 ký tự sau dấu _)
        processed_text = re.sub(r"_([0-9a-zA-Z])", r"<sub>\1</sub>", processed_text)
        processed_text = re.sub(r"\^([0-9a-zA-Z\-])", r"<sup>\1</sup>", processed_text)

        # 2. Xử lý các ký tự Unicode chỉ số bị hiển thị lỗi thành tag Markup
        unicode_subs = {"₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9"}
        unicode_sups = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9","⁻":"-"}
        for char, val in unicode_subs.items(): processed_text = processed_text.replace(char, f"<sub>{val}</sub>")
        for char, val in unicode_sups.items(): processed_text = processed_text.replace(char, f"<sup>{val}</sup>")

        # 3. Xử lý math_substrings (Tô màu và đổi font cho các biến đơn lẻ)
        if math_substrings:
            for sub in sorted(list(set(math_substrings)), key=len, reverse=True):
                # Chỉ thay thế nếu sub chưa nằm trong một tag nào đó (đơn giản hóa bằng cách check >)
                if sub in processed_text:
                    replacement = f"<span foreground='{color_math}' font_family='{font_math}'>{sub}</span>"
                    processed_text = processed_text.replace(sub, replacement)

        self.text_object = MarkupText(
            processed_text,
            font_size=CONFIG["subtitle_font_size"],
            font=CONFIG["DEFAULT_FONT"],
            color=WHITE,
            justify=True
        )
        self.background_object = BackgroundRectangle(
            self.text_object, 
            color=BLACK, 
            fill_opacity=0.8, 
            buff=0.2
        )
        self.current_subtitle_group = VGroup(self.background_object, self.text_object).to_edge(DOWN, buff=0.2)
        self.scene.add_fixed_in_frame_mobjects(self.current_subtitle_group)
    
    def remove_subtitle_display(self) -> None:
        """
        Loại bỏ phụ đề hiện tại khỏi cảnh.
        Được sử dụng ở cuối các chương hoặc trong các quá trình chuyển cảnh không cần văn bản.
        """
        if self.current_subtitle_group in self.scene.mobjects:
            self.scene.remove(self.current_subtitle_group)
        self.current_subtitle_group = VGroup()

def create_billboard_3d_label(
    scene: ThreeDScene, 
    tex_string: str, 
    position: np.ndarray, 
    color: Union[str, Any] = WHITE, 
    font_size: int = None
) -> MathTex:
    """
    Tạo nhãn toán học 3D có khả năng tự xoay để luôn hướng về camera, 
    đảm bảo các công thức luôn dễ đọc trong suốt quá trình xoay camera.
    """
    if font_size is None:
        font_size = CONFIG["label_3d_font_size"]
    
    label_object = MathTex(
        tex_string, 
        color=color, 
        font_size=font_size, 
        background_stroke_width=3, 
        background_stroke_color=BLACK
    )
    
    def update_face_camera(mobject: Mobject) -> None:
        """Hàm cập nhật để đồng bộ hướng của nhãn với các góc quay của camera."""
        new_mobject = MathTex(
            tex_string, 
            color=color, 
            font_size=font_size, 
            background_stroke_width=3, 
            background_stroke_color=BLACK
        )
        new_mobject.rotate(scene.camera.get_phi(), axis=RIGHT)
        new_mobject.rotate(scene.camera.get_theta() + PI/2, axis=OUT)
        new_mobject.move_to(position)
        mobject.become(new_mobject)
    
    label_object.add_updater(update_face_camera)
    return label_object

def create_grow_arrow_animation(
    start_point: np.ndarray, 
    end_point: np.ndarray, 
    color: Union[str, Any] = WHITE
) -> tuple:
    """
    Mô phỏng sự hình thành của vector từ một điểm gốc, 
    giúp người xem dễ dàng theo dõi sự xuất hiện của các thành phần mới.
    """
    arrow_object = Arrow3D(start=start_point, end=end_point, color=color)
    grow_animation = GrowFromPoint(arrow_object, point=start_point)
    return grow_animation, arrow_object

def pause_with_visual_focus(
    scene: Scene, 
    duration: float = 3.0, 
    focus_mobject: Mobject = None
) -> None:
    """
    Tạm dừng cảnh với tùy chọn làm nổi bật một đối tượng cụ thể.
    Giúp hướng sự chú ý của người xem vào các kết quả chính.
    """
    if focus_mobject is not None:
        effect_time = min(1.2, duration * 0.4)
        scene.play(
            focus_mobject.animate.set_color(YELLOW).scale(1.03),
            run_time=effect_time,
            rate_func=there_and_back
        )
        scene.wait(duration - effect_time)
    else:
        scene.wait(duration)

def apply_camera_breathing_effect(
    scene: ThreeDScene, 
    center_phi: float, 
    center_theta: float, 
    amplitude: float, 
    duration: float
) -> None:
    """
    Áp dụng hiệu ứng rung nhẹ giúp cảnh phim sinh động và tự nhiên hơn 
    trong các đoạn giải thích lý thuyết kéo dài.
    """
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
    def setup(self) -> None:
        """
        Thiết lập môi trường làm việc ban đầu, cấu hình các phân vùng 
        và khởi tạo bộ quản lý phụ đề cho toàn bộ hoạt cảnh.
        """
        super().setup()
        self.subtitle_manager = SubtitleManager(self)
        self.list_dynamic_r_entries = [] # Theo dõi các số thập phân động của ma trận R
        self.layout_zones = {
            "title": CONFIG["ZONE_TITLE"],
            "subtitle": CONFIG["ZONE_SUBTITLE"],
            "visual": CONFIG["ZONE_VISUAL"],
            "math": CONFIG["ZONE_MATH"]
        }
        self.initialize_layout_boundaries()

    def initialize_layout_boundaries(self) -> None:
        """
        Dựng các ranh giới vô hình giúp duy trì vị trí nhất quán 
        của tiêu đề và phụ đề trên các khung hình khác nhau.
        """
        self.title_boundary = Line(LEFT * 7, RIGHT * 7).move_to(self.layout_zones["title"]).set_stroke(opacity=0)
        self.subtitle_boundary = Line(LEFT * 7, RIGHT * 7).move_to(self.layout_zones["subtitle"]).set_stroke(opacity=0)
        self.add(self.title_boundary, self.subtitle_boundary)

    def add_to_specified_zone(self, mobject: Mobject, zone_name: str = "visual", is_fixed_in_frame: bool = True) -> Mobject:
        """
        Đặt một đối tượng vào một phân vùng cụ thể và thêm nó vào cảnh.
        Xử lý tỷ lệ cho phân vùng 'math' để đảm bảo các công thức không quá lớn.
        """
        target_position = self.layout_zones.get(zone_name, ORIGIN)
        if zone_name == "math":
            mobject.scale(CONFIG["MATH_SCALE"]).move_to(target_position)
        else:
            mobject.move_to(target_position)
            
        if is_fixed_in_frame:
            self.add_fixed_in_frame_mobjects(mobject)
        else:
            self.add(mobject)
        return mobject

    def play_with_consistent_timing(self, *animations, zone_name: str = "visual", **kwargs) -> None:
        """
        Thực hiện hoạt ảnh với thời gian mặc định từ CONFIG.
        Đơn giản hóa việc gọi các hiệu ứng chuyển cảnh thông thường.
        """
        if "run_time" not in kwargs:
            kwargs["run_time"] = CONFIG["TRANSITION_TIME"]
        self.play(*animations, **kwargs)

    def clear_visual_stage(self) -> None:
        """
        Làm sạch khu vực quan sát trung tâm để chuẩn bị cho các bước minh họa mới, 
        trong khi vẫn giữ lại các thông tin tham chiếu quan trọng ở các góc.
        """
        mobjects_to_remove = []
        for mobject in self.mobjects:
            pos_y = mobject.get_center()[1]
            if -3 < pos_y < 3:
                mobjects_to_remove.append(mobject)
        if mobjects_to_remove:
            self.play(*[FadeOut(mobject) for mobject in mobjects_to_remove], run_time=1)

    def create_billboard_label(
        self, 
        tex_string: str, 
        position: np.ndarray, 
        color: Union[str, Any] = WHITE, 
        font_size: int = None
    ) -> VGroup:
        """
        Tạo nhãn 3D tích hợp chặt chẽ với trạng thái camera của cảnh phim, 
        đảm bảo hiển thị chính xác trong các hoạt ảnh phức tạp.
        """
        if font_size is None:
            font_size = CONFIG["label_3d_font_size"]
        
        label_object = MathTex(
            tex_string, 
            color=color, 
            font_size=font_size, 
            background_stroke_width=3, 
            background_stroke_color=BLACK
        )
        bg = BackgroundRectangle(label_object, color=BLACK, fill_opacity=0.6, buff=0.1)
        group = VGroup(bg, label_object)
        base_label = label_object.copy()
        
        def update_face_camera_logic(mobject: Mobject) -> None:
            camera_phi = self.camera.get_phi()
            camera_theta = self.camera.get_theta()
            new_label = base_label.copy()
            new_label.rotate(camera_phi, axis=RIGHT)
            new_label.rotate(camera_theta + PI/2, axis=OUT)
            new_label.move_to(position)
            
            new_bg = BackgroundRectangle(new_label, color=BLACK, fill_opacity=0.6, buff=0.1)
            new_group = VGroup(new_bg, new_label)
            mobject.become(new_group)
            
        group.add_updater(update_face_camera_logic)
        return group

    def resize_and_reposition(self, mobject: Mobject, target_zone: str = "math") -> None:
        """
        Chuyển đổi một đối tượng từ trung tâm hình ảnh sang một góc tham chiếu.
        Thường được sử dụng để lưu trữ một phép tính đã hoàn thành để tham khảo sau này.
        """
        self.play(
            mobject.animate.scale(CONFIG["MATH_SCALE"]).move_to(self.layout_zones.get(target_zone, ORIGIN)),
            run_time=CONFIG["TRANSITION_TIME"]
        )

    def render_detailed_3d_vector(
        self, 
        axes: ThreeDAxes, 
        coordinates: list, 
        color: Union[str, Any] = WHITE, 
        label_string: str = None
    ) -> tuple:
        """
        Minh họa chi tiết quá trình dựng vector 3D từ các thành phần tọa độ, 
        giúp xây dựng mối liên kết trực quan giữa số học và hình học không gian.
        """
        coord_x, coord_y, coord_z = coordinates
        axes_origin = axes.get_origin()
        
        # 1. Các đường dẫn hướng + Điểm đánh dấu trên Ox, Oy, Oz
        guide_ox = DashedLine(axes.c2p(0, 0, 0), axes.c2p(coord_x, 0, 0), color=GRAY, stroke_width=CONFIG["AXIS_GUIDE_STROKE"])
        guide_oy = DashedLine(axes.c2p(0, 0, 0), axes.c2p(0, coord_y, 0), color=GRAY, stroke_width=CONFIG["AXIS_GUIDE_STROKE"])
        guide_oz = DashedLine(axes.c2p(0, 0, 0), axes.c2p(0, 0, coord_z), color=GRAY, stroke_width=CONFIG["AXIS_GUIDE_STROKE"])

        dot_ox = Dot3D(axes.c2p(coord_x, 0, 0), color=color, radius=0.08)
        dot_oy = Dot3D(axes.c2p(0, coord_y, 0), color=color, radius=0.08)
        dot_oz = Dot3D(axes.c2p(0, 0, coord_z), color=color, radius=0.08)
        
        self.play(Create(guide_ox), Create(dot_ox), run_time=1.3)
        self.play(Create(guide_oy), Create(dot_oy), run_time=1.3)
        self.play(Create(guide_oz), Create(dot_oz), run_time=1.3)
        self.wait(1.0)

        # 2. Chiếu lên mặt phẳng nền và dựng khung hình
        projection_line_x = DashedLine(axes.c2p(coord_x, 0, 0), axes.c2p(coord_x, coord_y, 0), color=GRAY)
        projection_line_y = DashedLine(axes.c2p(0, coord_y, 0), axes.c2p(coord_x, coord_y, 0), color=GRAY)
        self.play(Create(projection_line_x), Create(projection_line_y), run_time=1.8)
        
        # 3. Chiếu thẳng đứng
        vertical_projection_line = DashedLine(axes.c2p(coord_x, coord_y, 0), axes.c2p(coord_x, coord_y, coord_z), color=GRAY)
        top_connector_line = DashedLine(axes.c2p(0, 0, coord_z), axes.c2p(coord_x, coord_y, coord_z), color=GRAY)
        self.play(Create(vertical_projection_line), run_time=1.4)
        self.play(Create(top_connector_line), run_time=1.4)
        self.wait(1.2)

        # 4. Mũi tên vector cuối cùng
        arrow_object = Arrow3D(start=axes_origin, end=axes.c2p(coord_x, coord_y, coord_z), color=color)
        vector_magnitude = calculate_euclidean_norm([coord_x, coord_y, coord_z])
        dynamic_run_time = max(CONFIG["VECTOR_GROW_BASE"], vector_magnitude * CONFIG["VECTOR_GROW_SCALE"])
        grow_animation = GrowFromPoint(arrow_object, point=axes_origin)
        
        construction_group = VGroup(
            guide_ox, guide_oy, guide_oz, dot_ox, dot_oy, dot_oz, 
            projection_line_x, projection_line_y, vertical_projection_line, top_connector_line
        )
        
        if label_string:
            label_object = self.create_billboard_label(label_string, axes.c2p(coord_x, coord_y, coord_z) + OUT * 0.3, color=color)
            self.play(grow_animation, FadeIn(label_object), run_time=dynamic_run_time)
            return arrow_object, label_object, construction_group
        else:
            self.play(grow_animation, run_time=dynamic_run_time)
            return arrow_object, construction_group

    def highlight_active_vector(self, active_arrow: Arrow3D, other_arrows: list) -> None:
        """
        Hướng sự chú ý của người xem vào vector đang được xử lý bằng cách 
        tăng độ tương phản màu sắc và làm mờ các thành phần phụ.
        """
        animations = [
            active_arrow.animate.set_opacity(1.0).set_color(CONFIG["color_active_vector"]) 
        ]
        for arrow_object in other_arrows:
            if arrow_object is not None:
                animations.append(arrow_object.animate.set_opacity(0.25).set_color(CONFIG["color_dim_vector"]))
        self.play(*animations, run_time=0.7)

    def animate_formula_transformation(self, source_tex: MathTex, target_tex: MathTex) -> None:
        """
        Chuyển đổi một công thức toán học này sang một công thức khác với chuyển động hình cung mượt mà.
        Minh họa các bước biến đổi đại số một cách hiệu quả.
        """
        self.play(
            TransformMatchingTex(source_tex, target_tex, path_arc=PI / 4),
            run_time=2.5,
            rate_func=smooth
        )

    def play_subtract_projection_animation(
        self,
        axes: ThreeDAxes,
        coord_target_vector: list,
        list_coord_basis_vectors: list,
        color_target: Union[str, Any] = WHITE,
        color_orthonormal: Union[str, Any] = YELLOW,
    ) -> Tuple[Arrow3D, list, list]:
        """
        Trực quan hóa một bước Gram-Schmidt: chiếu → trừ → chuẩn hóa.

        Quy trình trực quan hóa bao gồm:
          - Giai đoạn A: Dựng đường chiếu bóng từ vector gốc lên không gian cơ sở hiện tại.
          - Giai đoạn B: Xác định phần dư trực giao sau khi loại bỏ hình chiếu.
          - Giai đoạn C: Chuẩn hóa phần dư để thu được vector trực chuẩn q mới.

        Returns:
            (arrow_orthonormal, coord_orthonormal, list_shadow_lines)
        """
        origin = axes.get_origin()

        # Tính tổng tất cả các hình chiếu lên các vector cơ sở đã có
        coord_total_projection = [0.0, 0.0, 0.0]
        list_shadow_lines = []
        for coord_basis in list_coord_basis_vectors:
            coord_proj_component = project_vector_onto(coord_target_vector, coord_basis)
            coord_total_projection = add_two_vectors(coord_total_projection, coord_proj_component)

        # Phase A — Vẽ shadow line (đường chiếu)
        shadow_line = DashedLine(
            axes.c2p(*coord_total_projection),
            axes.c2p(*coord_target_vector),
            color=CONFIG["color_shadow_line"],
            stroke_width=2.5,
        )
        arrow_projection = Arrow3D(
            start=origin,
            end=axes.c2p(*coord_total_projection),
            color=GRAY_B,
        )
        self.play(Create(arrow_projection), Create(shadow_line), run_time=1.8)
        list_shadow_lines.extend([shadow_line, arrow_projection])
        self.wait(1.0)

        # Phase B — Tính phần dư và dựng đứng
        coord_residual = subtract_two_vectors(coord_target_vector, coord_total_projection)
        arrow_residual = Arrow3D(
            start=origin,
            end=axes.c2p(*coord_residual),
            color=CONFIG["color_projection"],
        )
        label_residual = self.create_billboard_label(
            "e", axes.c2p(*coord_residual) + OUT * 0.25,
            color=CONFIG["color_projection"], font_size=30,
        )
        self.play(
            Create(arrow_residual), FadeIn(label_residual),
            arrow_projection.animate.set_opacity(0.3),
            run_time=2.0,
        )
        self.wait(0.8)

        # Phase C — Chuẩn hóa phần dư thành vector trực chuẩn
        coord_orthonormal = normalize_to_unit_vector(coord_residual)
        arrow_orthonormal = Arrow3D(
            start=origin,
            end=axes.c2p(*coord_orthonormal),
            color=color_orthonormal,
        )
        self.play(
            ReplacementTransform(arrow_residual, arrow_orthonormal),
            FadeOut(label_residual),
            run_time=1.6,
        )

        return arrow_orthonormal, coord_orthonormal, list_shadow_lines

# ==============================================================================
# LỚP CẢNH CHÍNH (MAIN SCENE)
# ==============================================================================

class QRAndDiagonalization(MatrixProjectScene):
    """
    Cảnh phim chính điều phối quá trình minh họa giải phẫu ma trận 
    bao gồm phân rã QR và chéo hóa PDP^-1.
    """

    # ---- Dữ liệu ma trận mặc định ----
    MATRIX_A_ROWS = [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
    COORD_ORIGINAL_COL_1 = [2, 1, 1]
    COORD_ORIGINAL_COL_2 = [1, 2, 1]
    COORD_ORIGINAL_COL_3 = [1, 1, 2]
    EIGENVALUE_LAMBDA_1 = 4
    EIGENVALUE_LAMBDA_23 = 1
    COORD_EIGENVEC_1 = [1, 1, 1]
    COORD_EIGENVEC_2 = [-1, 1, 0]
    COORD_EIGENVEC_3 = [-1, 0, 1]
    def calculate_dynamic_axes_spec(self, list_vectors: list, min_range: float = None) -> dict:
        """
        Tính toán phạm vi đối xứng của các trục tọa độ từ các vector dữ liệu để tránh bị cắt.
        Điều chỉnh khung nhìn một cách linh hoạt dựa trên quy mô của các vector đầu vào.
        """
        if min_range is None:
            min_range = CONFIG["axes_range"]
        
        max_absolute_value = max(abs(coord) for vector_val in list_vectors for coord in vector_val)
        axis_boundary = max(min_range, int(math.ceil(max_absolute_value * CONFIG["axes_padding_factor"])))
        axis_step = 1
        
        return {
            "bound": axis_boundary,
            "x_range": [-axis_boundary, axis_boundary, axis_step],
            "y_range": [-axis_boundary, axis_boundary, axis_step],
            "z_range": [-axis_boundary, axis_boundary, axis_step],
            "axis_length": max(CONFIG["axes_length"], 2 * axis_boundary),
        }

    def create_tracking_billboard_label(
        self, 
        tex_string: str, 
        arrow_mobject: Arrow3D, 
        color: Union[str, Any] = WHITE, 
        offset_vector: np.ndarray = OUT * 0.25, 
        font_size: int = None
    ) -> Mobject:
        """
        Tạo một nhãn billboard bám theo điểm đầu cuối hiện tại của một mũi tên đang di chuyển.
        Được sử dụng trong các hoạt ảnh Gram-Schmidt nơi các vector được biến đổi linh hoạt.
        """
        if font_size is None:
            font_size = CONFIG["label_3d_font_size"]

        def build_label_logic() -> VGroup:
            label_object = MathTex(
                tex_string,
                color=color,
                font_size=font_size,
                background_stroke_width=3,
                background_stroke_color=BLACK,
            )
            label_object.rotate(self.camera.get_phi(), axis=RIGHT)
            label_object.rotate(self.camera.get_theta() + PI / 2, axis=OUT)
            label_object.move_to(arrow_mobject.get_end() + offset_vector)
            bg = BackgroundRectangle(label_object, color=BLACK, fill_opacity=0.6, buff=0.1)
            return VGroup(bg, label_object)

        return always_redraw(build_label_logic)

    def display_projection_formula_panel(self) -> VGroup:
        """
        Giới thiệu công thức chiếu vector tổng quát dưới dạng một bảng tham chiếu.
        Cung cấp ngữ cảnh lý thuyết cho quá trình Gram-Schmidt.
        """
        projection_formula = MathTex(
            r"\operatorname{proj}_{\mathbf{v}}(\mathbf{u})",
            r"=",
            r"\frac{\langle \mathbf{u},\mathbf{v}\rangle}{\|\mathbf{v}\|^2}\,\mathbf{v}",
            font_size=36,
            color=CONFIG["color_projection"],
        )
        projection_formula.to_edge(RIGHT, buff=0.5).shift(UP * 1.5)
        
        title = Text("Công thức hình chiếu:", font_size=24, font=CONFIG["DEFAULT_FONT"])
        title.next_to(projection_formula, UP, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(title, projection_formula)
        self.play(FadeIn(title), Write(projection_formula), run_time=1.7)
        return VGroup(title, projection_formula)

    def display_orthogonal_formula(self, formula_tex_string: str) -> MathTex:
        """
        Hiển thị một công thức bước trực giao hóa ở phía dưới màn hình.
        Được sử dụng để cho thấy phép tính chính xác đang được trực quan hóa trong 3D.
        """
        step_formula = MathTex(formula_tex_string, font_size=34, color=YELLOW)
        step_formula.to_edge(DOWN, buff=0.95)
        self.add_fixed_in_frame_mobjects(step_formula)
        self.play(FadeIn(step_formula, shift=UP * 0.2), run_time=1.4)
        return step_formula

    def describe_r_upper_triangular_structure(
        self, 
        list_matrix_a_columns: list, 
        list_matrix_q_columns: list, 
        anchor_title_mobject: Mobject
    ) -> VGroup:
        """
        Trực quan hóa cấu trúc của ma trận R thông qua các tích vô hướng.
        Giải thích tại sao R có dạng tam giác trên dựa trên tính trực giao của Q.
        """
        val_r11 = calculate_dot_product(list_matrix_a_columns[0], list_matrix_q_columns[0])
        val_r12 = calculate_dot_product(list_matrix_a_columns[1], list_matrix_q_columns[0])
        val_r13 = calculate_dot_product(list_matrix_a_columns[2], list_matrix_q_columns[0])
        val_r22 = calculate_dot_product(list_matrix_a_columns[1], list_matrix_q_columns[1])
        val_r23 = calculate_dot_product(list_matrix_a_columns[2], list_matrix_q_columns[1])
        val_r33 = calculate_dot_product(list_matrix_a_columns[2], list_matrix_q_columns[2])

        formula_r_definition = MathTex(
            r"R_{j,i}=\langle a_i,q_j\rangle\;(j\le i),\quad R_{j,i}=0\;(j>i)",
            font_size=34,
            color=CONFIG["color_matrix_R"],
        )
        formula_r_definition.next_to(anchor_title_mobject, DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(formula_r_definition)

        matrix_r_symbolic = MathTex(
            r"R=\begin{pmatrix}",
            r"\langle a_1,q_1\rangle & \langle a_2,q_1\rangle & \langle a_3,q_1\rangle \\",
            r"0 & \langle a_2,q_2\rangle & \langle a_3,q_2\rangle \\",
            r"0 & 0 & \langle a_3,q_3\rangle",
            r"\end{pmatrix}",
            font_size=30,
        )
        matrix_r_symbolic.to_corner(DR, buff=0.45)
        self.add_fixed_in_frame_mobjects(matrix_r_symbolic)

        matrix_r_numeric = Matrix([
            [f"{val_r11:.2f}", f"{val_r12:.2f}", f"{val_r13:.2f}"],
            ["0", f"{val_r22:.2f}", f"{val_r23:.2f}"],
            ["0", "0", f"{val_r33:.2f}"],
        ], left_bracket="(", right_bracket=")").scale(0.66)
        matrix_r_display_group = VGroup(MathTex("R =", color=CONFIG["color_matrix_R"]), matrix_r_numeric).arrange(RIGHT, buff=0.18)
        matrix_r_display_group.to_corner(DL, buff=0.45)
        self.add_fixed_in_frame_mobjects(matrix_r_display_group)

        self.play(FadeIn(formula_r_definition, shift=DOWN * 0.2), run_time=1.2)
        self.play(FadeIn(matrix_r_symbolic, shift=LEFT * 0.2), run_time=1.5)
        self.wait(CONFIG["pause_formula_long"])
        self.play(FadeIn(matrix_r_display_group, shift=UP * 0.2), run_time=1.4)
        self.wait(CONFIG["pause_formula_long"])

        upper_triangle_entries = VGroup(*[
            matrix_r_numeric.get_entries()[0], matrix_r_numeric.get_entries()[1], matrix_r_numeric.get_entries()[2],
            matrix_r_numeric.get_entries()[4], matrix_r_numeric.get_entries()[5], matrix_r_numeric.get_entries()[8],
        ])
        self.play(Indicate(upper_triangle_entries, color=CONFIG["color_matrix_R"], scale_factor=1.08), run_time=1.6)
        self.wait(CONFIG["pause_formula_short"])
        return VGroup(formula_r_definition, matrix_r_symbolic, matrix_r_display_group)

    def display_row_echelon_steps(
        self, 
        start_tex_string: str, 
        ref_tex_string: str, 
        conclusion_tex_string: str, 
        highlight_color: Union[str, Any]
    ) -> MathTex:
        """
        Trình bày một chuỗi sư phạm: phương trình -> Dạng bậc thang (REF) -> không gian nghiệm.
        Giúp sinh viên theo dõi các bước đại số để tìm các vector riêng.
        """
        formula_start = MathTex(start_tex_string, font_size=38).move_to(UP * 0.15)
        formula_ref = MathTex(ref_tex_string, font_size=34).move_to(UP * 0.15)
        formula_conclusion = MathTex(conclusion_tex_string, font_size=36, color=highlight_color).move_to(UP * 0.15)

        self.play(Write(formula_start), run_time=1.2)
        self.wait(CONFIG["pause_formula_short"])
        self.play(ReplacementTransform(formula_start, formula_ref), run_time=1.5)
        self.wait(CONFIG["pause_formula_long"])
        self.play(ReplacementTransform(formula_ref, formula_conclusion), run_time=1.5)
        self.wait(CONFIG["pause_formula_short"])
        return formula_conclusion

    def describe_q_orthogonal_properties(
        self,
        list_matrix_q_columns: list,
        anchor_title_mobject: Mobject
    ) -> VGroup:
        """
        Trình bày các đặc tính đại số của Q như tính trực giao và độ dài đơn vị, 
        giúp người xem hiểu lý do tại sao Q^T Q = I.
        """
        formula_q_definition = MathTex(
            r"\|q_i\| = 1,\quad \langle q_i, q_j \rangle = 0\;(i \neq j)",
            font_size=34,
            color=CONFIG["color_matrix_Q"],
        )
        formula_q_definition.next_to(anchor_title_mobject, DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(formula_q_definition)
        formula_q_definition.set_opacity(0)

        matrix_q_symbolic = MathTex(
            r"Q^T Q = \begin{pmatrix} "
            r"\langle q_1, q_1 \rangle & \langle q_1, q_2 \rangle & \langle q_1, q_3 \rangle \\ "
            r"\langle q_2, q_1 \rangle & \langle q_2, q_2 \rangle & \langle q_2, q_3 \rangle \\ "
            r"\langle q_3, q_1 \rangle & \langle q_3, q_2 \rangle & \langle q_3, q_3 \rangle "
            r"\end{pmatrix} = I",
            font_size=30,
        )
        matrix_q_symbolic.to_corner(DR, buff=0.45).shift(UP * 0.8)
        self.add_fixed_in_frame_mobjects(matrix_q_symbolic)
        matrix_q_symbolic.set_opacity(0)

        matrix_i_numeric = Matrix([
            ["1", "0", "0"],
            ["0", "1", "0"],
            ["0", "0", "1"],
        ], left_bracket="(", right_bracket=")").scale(0.66)
        matrix_i_display_group = VGroup(MathTex("Q^T Q = I =", color=CONFIG["color_matrix_Q"]), matrix_i_numeric).arrange(RIGHT, buff=0.18)
        matrix_i_display_group.to_corner(DL, buff=0.45).shift(UP * 0.8)
        self.add_fixed_in_frame_mobjects(matrix_i_display_group)
        matrix_i_display_group.set_opacity(0)

        # Kích hoạt hiển thị bằng cách thay đổi opacity thay vì FadeIn để tránh lỗi nháy hình
        self.play(
            formula_q_definition.animate.set_opacity(1).shift(DOWN * 0.2),
            run_time=1.2
        )
        self.play(
            matrix_q_symbolic.animate.set_opacity(1).shift(LEFT * 0.2),
            run_time=1.5
        )
        self.wait(CONFIG["pause_formula_long"])
        self.play(
            matrix_i_display_group.animate.set_opacity(1).shift(UP * 0.2),
            run_time=1.4
        )
        self.wait(CONFIG["pause_formula_long"])

        diag_entries = VGroup(*[
            matrix_i_numeric.get_entries()[0], matrix_i_numeric.get_entries()[4], matrix_i_numeric.get_entries()[8]
        ])
        self.play(Indicate(diag_entries, color=CONFIG["color_matrix_Q"], scale_factor=1.08), run_time=1.6)
        self.wait(CONFIG["pause_formula_short"])
        return VGroup(formula_q_definition, matrix_q_symbolic, matrix_i_display_group)



    # ==================================================================
    # construct() — Điều phối toàn bộ kịch bản
    # ==================================================================

    def construct(self) -> None:
        """Luồng xây dựng chính: 4 phần / 13 scene theo de_xuat_script.md."""
        self.render_part1_matrix_storm()
        self.wait(CONFIG["chapter_gap"])
        self.render_part2_gram_schmidt()
        self.wait(CONFIG["chapter_gap"])
        self.render_part3_eigen_transformation()
        self.wait(CONFIG["chapter_gap"])
        self.render_part4_conclusion()

    # ==================================================================
    # PHẦN 1: Cơn bão Ma trận (Scene 1–2)
    # ==================================================================

    def render_part1_matrix_storm(self) -> None:
        """Phần 1: Ma trận A bóp méo không gian 3D."""
        self._play_scene1_standard_space()
        self._play_scene2_distortion()

    def _play_scene1_standard_space(self) -> None:
        """Scene 1: Không gian 3D chuẩn, 3 vector cơ sở e1/e2/e3, hiển thị A."""
        self.set_camera_orientation(
            phi=CONFIG["camera_phi_main"],
            theta=CONFIG["camera_theta_main"],
        )

        chapter_title_storm = Text(
            "Phần 1: Cơn bão Ma trận",
            font_size=CONFIG["title_font_size"],
            font=CONFIG["DEFAULT_FONT"],
        )
        chapter_title_storm.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(chapter_title_storm)
        self.play(Write(chapter_title_storm), run_time=CONFIG["TRANSITION_TIME"])

        self.subtitle_manager.update_subtitle_text(
            "Trong ĐSTT, một ma trận A biểu diễn một phép biến đổi không gian."
        )

        # Trục tọa độ 3D
        all_cols = [self.COORD_ORIGINAL_COL_1, self.COORD_ORIGINAL_COL_2, self.COORD_ORIGINAL_COL_3]
        axes_spec = self.calculate_dynamic_axes_spec(all_cols)
        threed_axes = ThreeDAxes(
            x_range=axes_spec["x_range"],
            y_range=axes_spec["y_range"],
            z_range=axes_spec["z_range"],
            x_length=axes_spec["axis_length"],
            y_length=axes_spec["axis_length"],
            z_length=axes_spec["axis_length"],
            axis_config={"color": GRAY},
        )
        self.play(Create(threed_axes), run_time=2.6)

        axes_bound = axes_spec["bound"]
        label_axis_x = self.create_billboard_label("x", threed_axes.c2p(axes_bound + 0.3, 0, 0), font_size=30)
        label_axis_y = self.create_billboard_label("y", threed_axes.c2p(0, axes_bound + 0.3, 0), font_size=30)
        label_axis_z = self.create_billboard_label("z", threed_axes.c2p(0, 0, axes_bound + 0.3), font_size=30)
        label_axis_origin = self.create_billboard_label("O", threed_axes.c2p(0, 0, 0) + LEFT * 0.2 + DOWN * 0.2, font_size=28)
        axes_labels_group = VGroup(label_axis_x, label_axis_y, label_axis_z, label_axis_origin)
        self.play(*[FadeIn(label_obj) for label_obj in axes_labels_group], run_time=1.4)

        # 3 vector cơ sở chuẩn
        origin_point = threed_axes.get_origin()
        arrow_basis_e1 = Arrow3D(start=origin_point, end=threed_axes.c2p(1, 0, 0), color=RED)
        arrow_basis_e2 = Arrow3D(start=origin_point, end=threed_axes.c2p(0, 1, 0), color=GREEN)
        arrow_basis_e3 = Arrow3D(start=origin_point, end=threed_axes.c2p(0, 0, 1), color=BLUE)
        label_e1 = self.create_billboard_label("e_1", threed_axes.c2p(1, 0, 0) + OUT * 0.2, color=RED, font_size=30)
        label_e2 = self.create_billboard_label("e_2", threed_axes.c2p(0, 1, 0) + OUT * 0.2, color=GREEN, font_size=30)
        label_e3 = self.create_billboard_label("e_3", threed_axes.c2p(0, 0, 1) + OUT * 0.2, color=BLUE, font_size=30)

        self.play(
            GrowFromPoint(arrow_basis_e1, origin_point),
            GrowFromPoint(arrow_basis_e2, origin_point),
            GrowFromPoint(arrow_basis_e3, origin_point),
            FadeIn(label_e1), FadeIn(label_e2), FadeIn(label_e3),
            run_time=2.0,
        )

        # Ma trận A tham chiếu ở góc
        matrix_a_mobject = Matrix(
            self.MATRIX_A_ROWS, left_bracket="(", right_bracket=")"
        ).scale(0.42)
        matrix_a_display_group = VGroup(MathTex("A = "), matrix_a_mobject).arrange(RIGHT)
        matrix_a_display_group.to_corner(UL, buff=0.45)
        self.add_fixed_in_frame_mobjects(matrix_a_display_group)
        self.play(FadeIn(matrix_a_display_group), run_time=1.2)
        self.wait(CONFIG["pause_key"])

        # Lưu references cho Scene 2
        self._storm_axes = threed_axes
        self._storm_axes_labels = axes_labels_group
        self._storm_arrows = [arrow_basis_e1, arrow_basis_e2, arrow_basis_e3]
        self._storm_labels = [label_e1, label_e2, label_e3]
        self._storm_matrix_display = matrix_a_display_group
        self._storm_chapter_title = chapter_title_storm

    def _play_scene2_distortion(self) -> None:
        """Scene 2: A bóp méo không gian — e1/e2/e3 biến thành a1/a2/a3."""
        self.subtitle_manager.update_subtitle_text(
            "Khi tác động A, không gian bị bóp méo. Các trục mất đi tính vuông góc."
        )
        threed_axes = self._storm_axes
        origin_point = threed_axes.get_origin()

        # Tạo các arrow mới tại vị trí cột của A
        arrow_distorted_col1 = Arrow3D(start=origin_point, end=threed_axes.c2p(*self.COORD_ORIGINAL_COL_1), color=CONFIG["color_original"])
        arrow_distorted_col2 = Arrow3D(start=origin_point, end=threed_axes.c2p(*self.COORD_ORIGINAL_COL_2), color=CONFIG["color_original"])
        arrow_distorted_col3 = Arrow3D(start=origin_point, end=threed_axes.c2p(*self.COORD_ORIGINAL_COL_3), color=CONFIG["color_original"])
        label_col1 = self.create_billboard_label("a_1", threed_axes.c2p(*self.COORD_ORIGINAL_COL_1) + OUT * 0.25, color=CONFIG["color_original"], font_size=30)
        label_col2 = self.create_billboard_label("a_2", threed_axes.c2p(*self.COORD_ORIGINAL_COL_2) + OUT * 0.25, color=CONFIG["color_original"], font_size=30)
        label_col3 = self.create_billboard_label("a_3", threed_axes.c2p(*self.COORD_ORIGINAL_COL_3) + OUT * 0.25, color=CONFIG["color_original"], font_size=30)

        self.play(
            ReplacementTransform(self._storm_arrows[0], arrow_distorted_col1),
            ReplacementTransform(self._storm_arrows[1], arrow_distorted_col2),
            ReplacementTransform(self._storm_arrows[2], arrow_distorted_col3),
            ReplacementTransform(self._storm_labels[0], label_col1),
            ReplacementTransform(self._storm_labels[1], label_col2),
            ReplacementTransform(self._storm_labels[2], label_col3),
            run_time=CONFIG["VECTOR_TRANSFORM_TIME"],
        )

        # Xoay camera nhấn mạnh sự biến dạng
        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES, run_time=CONFIG["animation_camera_long"])

        self.subtitle_manager.update_subtitle_text(
            "Để kiểm soát ma trận này, chúng ta cần 'giải phẫu' nó."
        )
        self.wait(CONFIG["pause_key"])

        # Dọn dẹp
        self.play(
            AnimationGroup(
                FadeOut(arrow_distorted_col1), FadeOut(arrow_distorted_col2), FadeOut(arrow_distorted_col3),
                FadeOut(label_col1), FadeOut(label_col2), FadeOut(label_col3),
                FadeOut(self._storm_axes), FadeOut(self._storm_axes_labels),
                FadeOut(self._storm_matrix_display), FadeOut(self._storm_chapter_title),
                lag_ratio=0.04,
            ),
            run_time=1.8,
        )

    # ==================================================================
    # PHẦN 2: Phân rã QR — Gram-Schmidt (Scene 3–7)
    # ==================================================================

    def render_part2_gram_schmidt(self) -> None:
        """Phần 2: Trực quan hóa Gram-Schmidt với split-screen R."""
        self._play_scene3_split_screen_intro()
        self._play_scene4_normalize_col1()
        self._play_scene5_orthogonalize_col2()
        self._play_scene6_update_r_col2()
        self._play_scene7_orthogonalize_col3_and_finalize()
        self._play_scene7_5_analyze_q()

    def _play_scene3_split_screen_intro(self) -> None:
        """Scene 3: Giới thiệu QR — bên trái: 3 vector a, bên phải: R trống."""
        chapter_title_qr = Text(
            "Phần 2: Phân rã QR — Gram-Schmidt",
            font_size=CONFIG["title_font_size"],
            font=CONFIG["DEFAULT_FONT"],
        )
        chapter_title_qr.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(chapter_title_qr)
        self.play(Write(chapter_title_qr), run_time=CONFIG["TRANSITION_TIME"])

        self.subtitle_manager.update_subtitle_text(
            "Cách thứ nhất: Phân rã QR — xây giàn giáo trực giao bằng Gram-Schmidt."
        )

        # Lý thuyết QR
        qr_theory_block = VGroup(
            Text("Q: Ma trận trực giao (Phép xoay/Trực chuẩn)", font=CONFIG["DEFAULT_FONT"], font_size=24, color=CONFIG["color_matrix_Q"]),
            Text("R: Ma trận tam giác trên (Nén/Co giãn)", font=CONFIG["DEFAULT_FONT"], font_size=24, color=CONFIG["color_matrix_R"]),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).shift(UP * 0.5)
        self.add_fixed_in_frame_mobjects(qr_theory_block)
        self.play(FadeIn(qr_theory_block, shift=UP))
        self.wait(2)

        self.subtitle_manager.update_subtitle_text(
            "Gram-Schmidt 'dọn dẹp' các cột: bẻ thẳng chúng cho vuông góc nhau từng bước."
        )
        self.wait(CONFIG["pause_key"])
        self.play(FadeOut(qr_theory_block), FadeOut(chapter_title_qr), run_time=0.8)

        # Ma trận A tham chiếu ở góc UL
        matrix_a_ref_mobject = Matrix(self.MATRIX_A_ROWS, left_bracket="(", right_bracket=")").scale(0.42)
        self._matrix_a_display = VGroup(MathTex("A = "), matrix_a_ref_mobject).arrange(RIGHT)
        self._matrix_a_display.to_corner(UL, buff=0.45)
        self.add_fixed_in_frame_mobjects(self._matrix_a_display)
        self.play(FadeIn(self._matrix_a_display), run_time=1.2)
        self._matrix_a_ref_mobject = matrix_a_ref_mobject

        # Ma trận R trống (dạng tam giác trên) ở góc UR
        self._r_entry_values = [["?"] * 3 for _ in range(3)]
        self._r_entry_values[1][0] = "0"
        self._r_entry_values[2][0] = "0"
        self._r_entry_values[2][1] = "0"
        matrix_r_mobject = Matrix(
            self._r_entry_values, left_bracket="(", right_bracket=")"
        ).scale(0.42)
        self._matrix_r_display = VGroup(MathTex("R = ", color=CONFIG["color_matrix_R"]), matrix_r_mobject).arrange(RIGHT)
        self._matrix_r_display.to_corner(UR, buff=0.45)
        self.add_fixed_in_frame_mobjects(self._matrix_r_display)
        self.play(FadeIn(self._matrix_r_display), run_time=1.2)
        self._matrix_r_mobject = matrix_r_mobject

        # Highlight cột A
        self._column_highlight_rect = None

        # Thiết lập 3D
        self.set_camera_orientation(phi=CONFIG["camera_phi_main"], theta=CONFIG["camera_theta_main"])
        all_cols = [self.COORD_ORIGINAL_COL_1, self.COORD_ORIGINAL_COL_2, self.COORD_ORIGINAL_COL_3]
        axes_spec = self.calculate_dynamic_axes_spec(all_cols)
        self._threed_axes = ThreeDAxes(
            x_range=axes_spec["x_range"], y_range=axes_spec["y_range"], z_range=axes_spec["z_range"],
            x_length=axes_spec["axis_length"], y_length=axes_spec["axis_length"], z_length=axes_spec["axis_length"],
            axis_config={"color": GRAY},
        )
        self.play(Create(self._threed_axes), run_time=2.6)

        axes_bound = axes_spec["bound"]
        label_x = self.create_billboard_label("x", self._threed_axes.c2p(axes_bound + 0.3, 0, 0), font_size=30)
        label_y = self.create_billboard_label("y", self._threed_axes.c2p(0, axes_bound + 0.3, 0), font_size=30)
        label_z = self.create_billboard_label("z", self._threed_axes.c2p(0, 0, axes_bound + 0.3), font_size=30)
        label_origin = self.create_billboard_label("O", self._threed_axes.c2p(0, 0, 0) + LEFT * 0.2 + DOWN * 0.2, font_size=28)
        self._qr_axes_labels = VGroup(label_x, label_y, label_z, label_origin)
        self.play(*[FadeIn(label) for label in self._qr_axes_labels], run_time=1.4)

        self._projection_formula_panel = self.display_projection_formula_panel()
        self.wait(CONFIG["pause_formula_long"])

    def _highlight_matrix_column(self, column_index: int) -> None:
        """Di chuyển khung tập trung vào một cột cụ thể của ma trận A."""
        target_rect = SurroundingRectangle(
            self._matrix_a_ref_mobject.get_columns()[column_index], color=YELLOW, buff=0.06
        )
        if self._column_highlight_rect is None:
            self._column_highlight_rect = target_rect
            self.add_fixed_in_frame_mobjects(self._column_highlight_rect)
            self.play(Create(self._column_highlight_rect), run_time=0.5)
        else:
            self.play(Transform(self._column_highlight_rect, target_rect), run_time=0.5)

    def _update_r_entry_display(self, row_idx: int, col_idx: int, value_str: str) -> None:
        """Cập nhật một ô trong ma trận R hiển thị (animate)."""
        entry_index = row_idx * 3 + col_idx
        old_entry = self._matrix_r_mobject.get_entries()[entry_index]
        new_entry = MathTex(value_str, font_size=20, color=CONFIG["color_matrix_R"])
        new_entry.move_to(old_entry.get_center())
        self.add_fixed_in_frame_mobjects(new_entry)
        self.play(
            FadeOut(old_entry, shift=UP * 0.1), 
            FadeIn(new_entry, shift=UP * 0.1), 
            run_time=0.8
        )
        self.list_dynamic_r_entries.append(new_entry)

    def _play_scene4_normalize_col1(self) -> None:
        """Scene 4: a₁ → q₁ bằng chuẩn hóa. Cập nhật R₁₁ = ||a₁||."""
        self.subtitle_manager.update_subtitle_text(
            "Bước 1: Giữ nguyên hướng cột đầu, chuẩn hóa về 1 → q₁. Lưu ||a₁|| vào R."
        )
        self._highlight_matrix_column(0)

        # Dựng vector a₁
        arrow_original_col1, label_original_col1, construction_col1 = self.render_detailed_3d_vector(
            self._threed_axes, self.COORD_ORIGINAL_COL_1,
            color=CONFIG["color_original"], label_string="a_1",
        )
        self.highlight_active_vector(arrow_original_col1, [])
        self.wait(CONFIG["pause_key"])

        # Chuẩn hóa → q₁
        coord_orthonormal_col1 = normalize_to_unit_vector(self.COORD_ORIGINAL_COL_1)
        arrow_orthonormal_col1 = Arrow3D(
            start=ORIGIN, end=self._threed_axes.c2p(*coord_orthonormal_col1),
            color=CONFIG["color_matrix_Q"],
        )
        label_orthonormal_col1 = self.create_tracking_billboard_label(
            "q_1", arrow_orthonormal_col1,
            color=CONFIG["color_matrix_Q"], offset_vector=DOWN * 0.55,
        )

        self.play(
            ReplacementTransform(arrow_original_col1.copy(), arrow_orthonormal_col1),
            FadeIn(label_orthonormal_col1),
            arrow_original_col1.animate.set_opacity(0.3),
            FadeOut(construction_col1),
        )
        self.highlight_active_vector(arrow_orthonormal_col1, [arrow_original_col1])

        # Cập nhật R₁₁
        norm_a1 = calculate_euclidean_norm(self.COORD_ORIGINAL_COL_1)
        self._update_r_entry_display(0, 0, f"{norm_a1:.2f}")
        self.wait(CONFIG["pause_key"])

        # Lưu references
        self._arrow_original_col1 = arrow_original_col1
        self._arrow_orthonormal_col1 = arrow_orthonormal_col1
        self._label_original_col1 = label_original_col1
        self._label_orthonormal_col1 = label_orthonormal_col1
        self._coord_orthonormal_col1 = coord_orthonormal_col1

    def _play_scene5_orthogonalize_col2(self) -> None:
        """Scene 5: Trừ hình chiếu a₂ lên q₁, dựng phần dư → q₂."""
        self.subtitle_manager.update_subtitle_text(
            "Bước 2: Trừ phần 'bóng' của a₂ lên q₁ → phần dư vuông góc → chuẩn hóa thành q₂."
        )
        self._highlight_matrix_column(1)

        # Dựng vector a₂
        arrow_original_col2, label_original_col2, construction_col2 = self.render_detailed_3d_vector(
            self._threed_axes, self.COORD_ORIGINAL_COL_2,
            color=CONFIG["color_original"], label_string="a_2",
        )
        self.highlight_active_vector(arrow_original_col2, [self._arrow_original_col1, self._arrow_orthonormal_col1])

        formula_step_e2 = self.display_orthogonal_formula(
            r"e_2 = a_2 - \operatorname{proj}_{q_1}(a_2),\quad q_2=\frac{e_2}{\|e_2\|}"
        )
        self.wait(CONFIG["pause_formula_long"])

        # playSubtractProjection
        arrow_orthonormal_col2, coord_orthonormal_col2, shadow_lines_col2 = self.play_subtract_projection_animation(
            self._threed_axes,
            self.COORD_ORIGINAL_COL_2,
            [self._coord_orthonormal_col1],
            color_orthonormal=CONFIG["color_matrix_Q"],
        )
        label_orthonormal_col2 = self.create_tracking_billboard_label(
            "q_2", arrow_orthonormal_col2,
            color=CONFIG["color_matrix_Q"], offset_vector=RIGHT * 0.3,
        )
        self.play(FadeIn(label_orthonormal_col2), FadeOut(construction_col2), run_time=1.0)

        self.highlight_active_vector(
            arrow_orthonormal_col2,
            [self._arrow_original_col1, arrow_original_col2, self._arrow_orthonormal_col1],
        )
        self.play(
            FadeOut(formula_step_e2),
            *[FadeOut(shadow_line) for shadow_line in shadow_lines_col2],
            run_time=0.9,
        )
        self.wait(CONFIG["pause_key"])

        # Lưu references
        self._arrow_original_col2 = arrow_original_col2
        self._arrow_orthonormal_col2 = arrow_orthonormal_col2
        self._label_original_col2 = label_original_col2
        self._label_orthonormal_col2 = label_orthonormal_col2
        self._coord_orthonormal_col2 = coord_orthonormal_col2

    def _play_scene6_update_r_col2(self) -> None:
        """Scene 6: Cập nhật R₁₂ và R₂₂ — giải thích ý nghĩa từng entry."""
        self.subtitle_manager.update_subtitle_text(
            "R ghi lại lịch sử: a₂ = r₁₂·q₁ + r₂₂·q₂."
        )
        val_r12 = calculate_dot_product(self.COORD_ORIGINAL_COL_2, self._coord_orthonormal_col1)
        val_r22 = calculate_dot_product(self.COORD_ORIGINAL_COL_2, self._coord_orthonormal_col2)
        self._update_r_entry_display(0, 1, f"{val_r12:.2f}")
        self._update_r_entry_display(1, 1, f"{val_r22:.2f}")
        self.wait(CONFIG["pause_key"])

    def _play_scene7_orthogonalize_col3_and_finalize(self) -> None:
        """Scene 7: Trừ chiếu a₃ trên (q₁, q₂) → q₃. Cập nhật cột 3 R. Giải thích tam giác trên."""
        self.subtitle_manager.update_subtitle_text(
            "Bước 3: Trừ cả hai 'bóng' trên q₁ và q₂ khỏi a₃ → chuẩn hóa thành q₃."
        )
        self._highlight_matrix_column(2)

        # Dựng vector a₃
        arrow_original_col3, label_original_col3, construction_col3 = self.render_detailed_3d_vector(
            self._threed_axes, self.COORD_ORIGINAL_COL_3,
            color=CONFIG["color_original"], label_string="a_3",
        )
        self.highlight_active_vector(
            arrow_original_col3,
            [self._arrow_original_col1, self._arrow_original_col2, self._arrow_orthonormal_col1, self._arrow_orthonormal_col2],
        )

        formula_step_e3 = self.display_orthogonal_formula(
            r"e_3 = a_3 - \operatorname{proj}_{q_1}(a_3) - \operatorname{proj}_{q_2}(a_3)"
        )
        self.wait(CONFIG["pause_formula_long"])

        # Mặt phẳng (q₁, q₂)
        plane_scale = 1.45
        coord_q1 = self._coord_orthonormal_col1
        coord_q2 = self._coord_orthonormal_col2
        basis_plane = Polygon(
            self._threed_axes.c2p(*add_two_vectors(multiply_vector_by_scalar(plane_scale, coord_q1), multiply_vector_by_scalar(plane_scale, coord_q2))),
            self._threed_axes.c2p(*add_two_vectors(multiply_vector_by_scalar(-plane_scale, coord_q1), multiply_vector_by_scalar(plane_scale, coord_q2))),
            self._threed_axes.c2p(*add_two_vectors(multiply_vector_by_scalar(-plane_scale, coord_q1), multiply_vector_by_scalar(-plane_scale, coord_q2))),
            self._threed_axes.c2p(*add_two_vectors(multiply_vector_by_scalar(plane_scale, coord_q1), multiply_vector_by_scalar(-plane_scale, coord_q2))),
            color=CONFIG["color_projection"], fill_color=CONFIG["color_projection"],
            fill_opacity=0.18, stroke_opacity=0.45, stroke_width=2,
        )
        self.play(FadeIn(basis_plane), run_time=1.2)

        # playSubtractProjection
        arrow_orthonormal_col3, coord_orthonormal_col3, shadow_lines_col3 = self.play_subtract_projection_animation(
            self._threed_axes,
            self.COORD_ORIGINAL_COL_3,
            [coord_q1, coord_q2],
            color_orthonormal=CONFIG["color_matrix_Q"],
        )
        label_orthonormal_col3 = self.create_tracking_billboard_label(
            "q_3", arrow_orthonormal_col3,
            color=CONFIG["color_matrix_Q"], offset_vector=LEFT * 0.25,
        )
        self.play(
            FadeIn(label_orthonormal_col3),
            arrow_original_col3.animate.set_opacity(0.3),
            FadeOut(formula_step_e3), FadeOut(basis_plane), FadeOut(construction_col3),
            *[FadeOut(shadow_line) for shadow_line in shadow_lines_col3],
            run_time=2.0,
        )

        self.highlight_active_vector(
            arrow_orthonormal_col3,
            [self._arrow_original_col1, self._arrow_original_col2, arrow_original_col3,
             self._arrow_orthonormal_col1, self._arrow_orthonormal_col2],
        )

        # Cập nhật cột 3 của R
        val_r13 = calculate_dot_product(self.COORD_ORIGINAL_COL_3, coord_q1)
        val_r23 = calculate_dot_product(self.COORD_ORIGINAL_COL_3, coord_q2)
        val_r33 = calculate_dot_product(self.COORD_ORIGINAL_COL_3, coord_orthonormal_col3)
        self._update_r_entry_display(0, 2, f"{val_r13:.2f}")
        self._update_r_entry_display(1, 2, f"{val_r23:.2f}")
        self._update_r_entry_display(2, 2, f"{val_r33:.2f}")

        self.subtitle_manager.update_subtitle_text(
            "R tam giác trên: q₃ sinh sau cùng, không can thiệp quá khứ của a₁, a₂."
        )
        self.wait(CONFIG["pause_scene_detail"])

        # Lưu references để dùng cho scene 7.5
        self._arrow_orthonormal_col3 = arrow_orthonormal_col3
        self._coord_orthonormal_col3 = coord_orthonormal_col3
        self._label_orthonormal_col3 = label_orthonormal_col3
        self._arrow_original_col3 = arrow_original_col3
        self._label_original_col3 = label_original_col3

    def _play_scene7_5_analyze_q(self) -> None:
        """Scene 7.5: Giải phẫu Q — Hệ trục trực chuẩn và Q^T Q = I."""
        self.subtitle_manager.update_subtitle_text(
            "Nhìn lại ma trận Q: Các cột của nó tạo thành một hệ trục trực chuẩn hoàn hảo.",
            math_substrings=["Q"]
        )
        
        chapter_title_q = Text(
            "Giải phẫu Q — Hệ trục trực chuẩn",
            font_size=CONFIG["title_font_size"],
            font=CONFIG["DEFAULT_FONT"],
        )
        chapter_title_q.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(chapter_title_q)
        self.play(FadeIn(chapter_title_q, shift=DOWN))

        # Dọn dẹp các đối tượng từ Scene 7 để chuẩn bị cho bảng phân tích mới
        scene7_leftovers = [
            self._matrix_a_display, self._matrix_r_display,
            self._column_highlight_rect, self._projection_formula_panel,
            self._arrow_original_col1, self._arrow_original_col2, self._arrow_original_col3,
            self._label_original_col1, self._label_original_col2, self._label_original_col3,
            *self.list_dynamic_r_entries, # Dọn dẹp triệt để các số thập phân động
        ]
        fadeout_animations = [FadeOut(obj) for obj in scene7_leftovers if obj is not None]
        if fadeout_animations:
            self.play(*fadeout_animations, run_time=1.2)

        list_q_cols = [
            self._coord_orthonormal_col1,
            self._coord_orthonormal_col2,
            self._coord_orthonormal_col3
        ]

        q_analysis_group = self.describe_q_orthogonal_properties(
            list_matrix_q_columns=list_q_cols,
            anchor_title_mobject=chapter_title_q
        )

        self.subtitle_manager.update_subtitle_text(
            "Mọi cột đều có độ dài bằng 1 và vuông góc với nhau. Đây chính là bản chất của Q.",
            math_substrings=["1", "Q"]
        )
        self.wait(CONFIG["pause_scene_detail"])

        self.play(FadeOut(q_analysis_group), FadeOut(chapter_title_q), run_time=1.5)

        # Dọn dẹp cuối phần 2
        # Lưu ý: Các đối tượng đã bị FadeOut ở đầu scene 7.5 (ma trận A, R,
        # highlight, panel, arrows/labels gốc) được loại khỏi danh sách này
        # để tránh lỗi "Mobject not found".
        self.play(
            AnimationGroup(
                FadeOut(self._threed_axes),
                FadeOut(self._arrow_orthonormal_col1), FadeOut(self._arrow_orthonormal_col2), FadeOut(self._arrow_orthonormal_col3),
                FadeOut(self._label_orthonormal_col1), FadeOut(self._label_orthonormal_col2), FadeOut(self._label_orthonormal_col3),
                FadeOut(self._qr_axes_labels),
                lag_ratio=0.03,
            ),
            run_time=2.0,
        )

    # ==================================================================
    # PHẦN 3: Chéo hóa — Trị riêng & Vector riêng (Scene 8–11)
    # ==================================================================

    def render_part3_eigen_transformation(self) -> None:
        """Phần 3: Tìm eigenvectors, xây đường ống PDP⁻¹."""
        self._play_scene8_restore_distorted_space()
        self._play_scene9_filter_eigenvectors()
        self._play_scene10_build_pdp_matrices()
        self._play_scene11_pipeline_visualization()

    def _play_scene8_restore_distorted_space(self) -> None:
        """Scene 8: Phục hồi không gian bóp méo, đặt câu hỏi về 'bộ xương bất biến'."""
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        chapter_title_eigen = Text(
            "Phần 3: Chéo hóa — Vector riêng",
            font_size=CONFIG["title_font_size"],
            font=CONFIG["DEFAULT_FONT"],
        )
        chapter_title_eigen.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(chapter_title_eigen)
        self.play(Write(chapter_title_eigen), run_time=CONFIG["TRANSITION_TIME"])
        self._chapter_title_eigen = chapter_title_eigen

        self.subtitle_manager.update_subtitle_text(
            "QR cho hệ trục đẹp, nhưng không triệt tiêu biến dạng. Có 'bộ xương' nào miễn nhiễm?"
        )
        self.wait(CONFIG["pause_key"])

    def _play_scene9_filter_eigenvectors(self) -> None:
        """Scene 9: Lọc eigenvectors — các mũi tên ngẫu nhiên biến mất, chỉ còn v₁/v₂/v₃."""
        self.subtitle_manager.update_subtitle_text(
            "Trị riêng λ: giá trị mà tại đó A chỉ co giãn vector, không xoay (Av = λv).",
            math_substrings=["λ", "A"]
        )

        # Phương trình đặc trưng
        wait_time_fast = 1.6
        run_time_fast = 1.5

        formula_char_eq = MathTex(r"\det(A - \lambda I) = 0", font_size=46).shift(UP * 1.4)
        formula_char_det = MathTex(
            r"\begin{vmatrix} 2-\lambda & 1 & 1 \\ 1 & 2-\lambda & 1 \\ 1 & 1 & 2-\lambda \end{vmatrix} = 0",
            font_size=40,
        ).shift(UP * 1.4)
        formula_char_factored = MathTex(r"-(\lambda - 4)(\lambda - 1)^2 = 0", font_size=46).shift(UP * 1.4)
        formula_eigenvalues = MathTex(
            r"\lambda_1 = ", "4", r",\; \lambda_2 = ", "1", r",\; \lambda_3 = ", "1",
            font_size=40, color=YELLOW,
        ).shift(UP * 1.4)

        self.play(Write(formula_char_eq))
        self.wait(wait_time_fast)
        self.play(ReplacementTransform(formula_char_eq, formula_char_det), run_time=run_time_fast)
        self.wait(wait_time_fast)
        self.play(ReplacementTransform(formula_char_det, formula_char_factored), run_time=run_time_fast)
        self.wait(wait_time_fast)
        self.play(ReplacementTransform(formula_char_factored, formula_eigenvalues), run_time=run_time_fast)
        self.wait(0.8)
        self.play(formula_eigenvalues.animate.next_to(self._chapter_title_eigen, DOWN, buff=0.35), run_time=1.2)

        # Tìm vector riêng λ₁ = 4
        self.subtitle_manager.update_subtitle_text(
            "Với λ₁ = 4: giải (A − 4I)v = 0 → v₁ = (1,1,1).",
            math_substrings=["λ₁ = 4", "(A − 4I)v = 0", "v₁ = (1,1,1)"]
        )
        formula_eigvec_l1 = self.display_row_echelon_steps(
            r"(A - 4I)\mathbf{v}=0",
            r"\left[\begin{array}{ccc|c}-2&1&1&0\\1&-2&1&0\\1&1&-2&0\end{array}\right]\sim"
            r"\left[\begin{array}{ccc|c}1&0&-1&0\\0&1&-1&0\\0&0&0&0\end{array}\right]",
            r"x=z,\;y=z\Rightarrow \mathbf{v}_1=(1,1,1)",
            CONFIG["color_eigenvalue_1"],
        )

        # Tìm vector riêng λ₂ = λ₃ = 1
        self.subtitle_manager.update_subtitle_text(
            "Với λ₂ = λ₃ = 1: giải (A − I)v = 0 → v₂ = (−1,1,0), v₃ = (−1,0,1).",
            math_substrings=["λ₂ = λ₃ = 1", "(A − I)v = 0", "v₂ = (−1,1,0)", "v₃ = (−1,0,1)"]
        )
        # Dịch chuyển khối phương trình lên trên để nhường không gian cho bước tiếp theo
        self.play(formula_eigvec_l1.animate.shift(UP * 1.2), run_time=0.6)
        formula_eigvec_l23 = self.display_row_echelon_steps(
            r"(A-I)\mathbf{v}=0",
            r"\left[\begin{array}{ccc|c}1&1&1&0\\1&1&1&0\\1&1&1&0\end{array}\right]\sim"
            r"\left[\begin{array}{ccc|c}1&1&1&0\\0&0&0&0\\0&0&0&0\end{array}\right]",
            r"x+y+z=0\Rightarrow \mathbf{v}_2=(-1,1,0),\;\mathbf{v}_3=(-1,0,1)",
            CONFIG["color_eigenvalue_2"],
        )

        # Hiển thị 3 vector riêng
        self.subtitle_manager.update_subtitle_text(
            "Đó là các vector riêng — A không bẻ cong chúng, chỉ kéo dãn với hệ số λ.",
            math_substrings=["A", "λ"]
        )
        label_v1 = MathTex(r"v_1 = ", color=CONFIG["color_eigenvalue_1"])
        mat_v1 = Matrix([[1], [1], [1]], left_bracket="(", right_bracket=")")
        mat_v1.get_entries().set_color(CONFIG["color_eigenvalue_1"])
        v1_group = VGroup(label_v1, mat_v1).arrange(RIGHT)

        label_v2 = MathTex(r"v_2 = ", color=CONFIG["color_eigenvalue_2"])
        mat_v2 = Matrix([[-1], [1], [0]], left_bracket="(", right_bracket=")")
        mat_v2.get_entries().set_color(CONFIG["color_eigenvalue_2"])
        v2_group = VGroup(label_v2, mat_v2).arrange(RIGHT)

        label_v3 = MathTex(r"v_3 = ", color=CONFIG["color_eigenvalue_3"])
        mat_v3 = Matrix([[-1], [0], [1]], left_bracket="(", right_bracket=")")
        mat_v3.get_entries().set_color(CONFIG["color_eigenvalue_3"])
        v3_group = VGroup(label_v3, mat_v3).arrange(RIGHT)

        eigenvecs_group = VGroup(v1_group, v2_group, v3_group).arrange(RIGHT, buff=0.8).scale(0.9).move_to(ORIGIN + UP * 0.1)
        self.play(FadeOut(formula_eigvec_l1), FadeOut(formula_eigvec_l23), FadeIn(eigenvecs_group, shift=UP), run_time=1.4)
        self.wait(wait_time_fast)

        # Lưu references
        self._formula_eigenvalues = formula_eigenvalues
        self._eigenvecs_group = eigenvecs_group
        self._mat_v1 = mat_v1
        self._mat_v2 = mat_v2
        self._mat_v3 = mat_v3

    def _play_scene10_build_pdp_matrices(self) -> None:
        """Scene 10: Xây dựng P (cột = vector riêng) và D (đường chéo λ)."""
        self.subtitle_manager.update_subtitle_text(
            "P gom vector riêng thành cột, D đặt trị riêng trên đường chéo → A = PDP⁻¹.",
            math_substrings=["A", "P", "D", "⁻¹"]
        )

        matrix_p_mobject = Matrix(
            [[1, -1, -1], [1, 1, 0], [1, 0, 1]],
            left_bracket="(", right_bracket=")",
        )
        matrix_d_mobject = Matrix(
            [[4, 0, 0], [0, 1, 0], [0, 0, 1]],
            left_bracket="(", right_bracket=")",
        )
        # Khởi tạo các entries với opacity=0 để tạo hoạt ảnh sao chép từ vector riêng
        matrix_p_mobject.get_entries().set_opacity(0)
        matrix_d_mobject.get_entries().set_opacity(0)

        matrix_p_group = VGroup(MathTex("P = "), matrix_p_mobject).arrange(RIGHT)
        matrix_d_group = VGroup(MathTex("D = "), matrix_d_mobject).arrange(RIGHT)
        pd_display = VGroup(matrix_p_group, matrix_d_group).arrange(RIGHT, buff=1.0).shift(DOWN * 2.0)

        self.play(
            FadeIn(matrix_p_group[0]), FadeIn(matrix_p_mobject.get_brackets()),
            FadeIn(matrix_d_group[0]), FadeIn(matrix_d_mobject.get_brackets()),
            run_time=1.4,
        )
        self.wait(1.0)

        # Animate: copy eigenvectors → cột P
        p_cols = matrix_p_mobject.get_columns()
        p_cols[0].set_color(CONFIG["color_eigenvalue_1"]).set_opacity(0)
        p_cols[1].set_color(CONFIG["color_eigenvalue_2"]).set_opacity(0)
        p_cols[2].set_color(CONFIG["color_eigenvalue_3"]).set_opacity(0)

        self.play(TransformFromCopy(self._mat_v1.get_entries(), p_cols[0].set_opacity(1)), run_time=1.1)
        self.play(TransformFromCopy(self._mat_v2.get_entries(), p_cols[1].set_opacity(1)), run_time=1.1)
        self.play(TransformFromCopy(self._mat_v3.get_entries(), p_cols[2].set_opacity(1)), run_time=1.1)
        self.play(FadeOut(self._eigenvecs_group), run_time=0.8)
        self.wait(0.6)

        # Animate: copy eigenvalues → đường chéo D
        d_diag = VGroup(matrix_d_mobject.get_entries()[0], matrix_d_mobject.get_entries()[4], matrix_d_mobject.get_entries()[8])
        d_diag[0].set_color(CONFIG["color_eigenvalue_1"]).set_opacity(1)
        d_diag[1].set_color(CONFIG["color_eigenvalue_2"]).set_opacity(1)
        d_diag[2].set_color(CONFIG["color_eigenvalue_3"]).set_opacity(1)
        d_zeros = VGroup(*[matrix_d_mobject.get_entries()[idx] for idx in [1, 2, 3, 5, 6, 7]])
        d_zeros.set_opacity(1)

        self.play(
            TransformFromCopy(self._formula_eigenvalues[1], d_diag[0]),
            TransformFromCopy(self._formula_eigenvalues[3], d_diag[1]),
            TransformFromCopy(self._formula_eigenvalues[5], d_diag[2]),
            FadeIn(d_zeros),
            run_time=1.8,
        )
        self.wait(1.6)

        self.play(FadeOut(self._formula_eigenvalues), pd_display.animate.move_to(UP * 1.2), run_time=1.2)

        formula_diag = MathTex(r"A = P D P^{-1}", color=YELLOW, font_size=48)
        formula_diag.next_to(pd_display, DOWN, buff=0.5)
        self.play(Write(formula_diag), run_time=1.2)
        self.wait(1.0)

        self._pd_display = pd_display
        self._formula_diag = formula_diag
        self._matrix_p_mobject = matrix_p_mobject
        self._matrix_d_mobject = matrix_d_mobject

    def _play_scene11_pipeline_visualization(self) -> None:
        """Scene 11: Đường ống P⁻¹ → D → P — cầu nối QR vs PDP⁻¹."""
        self.subtitle_manager.update_subtitle_text(
            "P⁻¹ đổi sang cơ sở riêng → D co giãn → P đưa về ban đầu.",
            math_substrings=["P", "D", "⁻¹"]
        )

        # Sơ đồ chuỗi
        formula_chain = MathTex(
            r"P^{-1}\mathbf{v}\;\xrightarrow{\;D\;}\;D(P^{-1}\mathbf{v)}\;\xrightarrow{\;P\;}\;PDP^{-1}\mathbf{v}",
            font_size=34, color=CONFIG["color_matrix_D"],
        ).next_to(self._pd_display, DOWN, buff=0.8)
        formula_confirm = MathTex(r"PDP^{-1}=A", font_size=44, color=GREEN).move_to(formula_chain)

        self.play(FadeOut(self._formula_diag), run_time=0.6)
        self.play(Write(formula_chain), run_time=1.6)
        self.wait(CONFIG["pause_formula_short"])
        self.play(TransformMatchingTex(formula_chain, formula_confirm), run_time=1.6)
        self.wait(CONFIG["pause_formula_short"])

        # Loại bỏ thông báo kiểm chứng cũ để tránh đè chữ lên khối nội dung mới
        self.play(FadeOut(formula_confirm), run_time=0.8)

        # Cầu nối: Q vs P
        bridge_title = Text("Q và P đều là đổi cơ sở", font=CONFIG["DEFAULT_FONT"], font_size=24)
        bridge_q = Text("Q: cơ sở trực chuẩn (đẹp hình học)", font=CONFIG["DEFAULT_FONT"], font_size=22, color=CONFIG["color_matrix_Q"])
        bridge_p = Text("P: cơ sở vector riêng (đẹp đại số)", font=CONFIG["DEFAULT_FONT"], font_size=22, color=CONFIG["color_highlight"])
        bridge_group = VGroup(bridge_title, bridge_q, bridge_p).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        bridge_group.move_to(DOWN * 1.5)
        # Sử dụng thuộc tính opacity để kiểm soát hiển thị trong ThreeDScene
        self.add_fixed_in_frame_mobjects(bridge_group)
        bridge_group.set_opacity(0)
        self.play(
            bridge_group.animate.set_opacity(1).shift(RIGHT * 0.2), 
            run_time=1.4
        )
        self.wait(CONFIG["pause_formula_long"])

        # Kiểm chứng nhanh
        self.subtitle_manager.update_subtitle_text(
            "Kiểm chứng: PDP⁻¹ phải ra đúng A ban đầu.",
            math_substrings=["PDP⁻¹", "A"]
        )
        formula_verify = MathTex(
            r"PDP^{-1} = \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix} = A \;\checkmark",
            font_size=34, color=GREEN,
        ).move_to(DOWN * 0.3)
        # formula_confirm đã được FadeOut ở trên, chỉ cần xóa bridge_group.
        self.play(FadeOut(bridge_group), Write(formula_verify), run_time=1.4)
        self.wait(CONFIG["pause_key"])

        # Dọn dẹp
        self.play(
            AnimationGroup(
                FadeOut(self._chapter_title_eigen), FadeOut(self._pd_display),
                FadeOut(formula_verify),
                lag_ratio=0.03,
            ),
            run_time=1.8,
        )

    # ==================================================================
    # PHẦN 4: Kết luận (Scene 12–13)
    # ==================================================================

    def render_part4_conclusion(self) -> None:
        """Phần 4: So sánh QR vs PDP⁻¹, công thức Aⁿ, kết thúc."""
        self._play_scene12_split_comparison()
        self._play_scene13_power_formula()

    def _play_scene12_split_comparison(self) -> None:
        """Scene 12: Split screen — trái: QR (hình học), phải: PDP⁻¹ (đại số)."""
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        chapter_title_conclusion = Text(
            "Tổng kết: Hai góc nhìn Giải phẫu Ma trận",
            font_size=CONFIG["title_font_size"],
            font=CONFIG["DEFAULT_FONT"],
        )
        chapter_title_conclusion.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(chapter_title_conclusion)
        self.play(Write(chapter_title_conclusion), run_time=CONFIG["TRANSITION_TIME"])

        self.subtitle_manager.update_subtitle_text(
            "QR: ép A vào quy chuẩn trực giao, tối ưu giải hệ. PDP⁻¹: tìm bản chất đại số tự nhiên.",
            math_substrings=["QR", "A", "PDP⁻¹"]
        )

        # Bên trái: QR
        qr_title = Text("A = QR", font=CONFIG["DEFAULT_FONT"], font_size=32, color=CONFIG["color_matrix_Q"])
        qr_desc1 = Text("Gram-Schmidt → hệ trực chuẩn", font=CONFIG["DEFAULT_FONT"], font_size=20)
        qr_desc2 = Text("R: tam giác trên (hệ số tổ hợp)", font=CONFIG["DEFAULT_FONT"], font_size=20, color=CONFIG["color_matrix_R"])
        qr_desc3 = Text("Ứng dụng: giải hệ PT, least squares", font=CONFIG["DEFAULT_FONT"], font_size=20, color=GRAY_B)
        qr_block = VGroup(qr_title, qr_desc1, qr_desc2, qr_desc3).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        qr_block.move_to(LEFT * 3.2 + UP * 0.2)

        # Bên phải: PDP⁻¹
        # Sử dụng MathTex để đảm bảo các ký tự đặc biệt được render chính xác bằng LaTeX
        pdp_title = MathTex(r"A = PDP^{-1}", font_size=42, color=CONFIG["color_matrix_D"])
        pdp_desc1 = Text("Vector riêng → trục co giãn gốc", font=CONFIG["DEFAULT_FONT"], font_size=20)
        pdp_desc2 = Text("D: đường chéo (trị riêng λ)", font=CONFIG["DEFAULT_FONT"], font_size=20, color=CONFIG["color_eigenvalue_1"])
        pdp_desc3 = Text("Ứng dụng: lũy thừa Aⁿ, ổn định", font=CONFIG["DEFAULT_FONT"], font_size=20, color=GRAY_B)
        pdp_block = VGroup(pdp_title, pdp_desc1, pdp_desc2, pdp_desc3).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        pdp_block.move_to(RIGHT * 3.2 + UP * 0.2)

        # Đường chia
        divider_line = DashedLine(UP * 2, DOWN * 2, color=GRAY, stroke_width=1.5)

        self.add_fixed_in_frame_mobjects(qr_block, pdp_block, divider_line)
        self.play(
            FadeIn(qr_block, shift=RIGHT * 0.3),
            FadeIn(pdp_block, shift=LEFT * 0.3),
            Create(divider_line),
            run_time=2.0,
        )
        self.wait(CONFIG["pause_formula_long"])

        self._conclusion_mobjects = VGroup(chapter_title_conclusion, qr_block, pdp_block, divider_line)

    def _play_scene13_power_formula(self) -> None:
        """Scene 13: Công thức Aⁿ = PDⁿP⁻¹ nhấp nháy + thông điệp kết."""
        self.subtitle_manager.update_subtitle_text(
            "Chéo hóa biến lũy thừa ma trận phức tạp thành phép nhân vô hướng đơn giản."
        )

        formula_power = MathTex(
            r"A^n = P D^n P^{-1}",
            font_size=52,
            color=CONFIG["color_pipeline_arrow"],
        ).move_to(DOWN * 1.5)
        self.add_fixed_in_frame_mobjects(formula_power)
        self.play(Write(formula_power), run_time=1.4)

        # Nhấp nháy
        self.play(
            formula_power.animate.set_color(YELLOW).scale(1.05),
            run_time=0.8, rate_func=there_and_back,
        )
        self.wait(CONFIG["pause_key"])

        # Thông điệp kết
        self.play(FadeOut(self._conclusion_mobjects), run_time=1.0)
        closing_text = Text(
            "The Nature of Linear Algebra",
            font=CONFIG["DEFAULT_FONT"],
            font_size=44,
            color=WHITE,
        ).move_to(UP * 0.5)
        thanks_text = Text(
            "Cảm ơn các bạn đã theo dõi!",
            font=CONFIG["DEFAULT_FONT"],
            font_size=36,
        ).move_to(DOWN * 0.8)

        self.add_fixed_in_frame_mobjects(closing_text, thanks_text)
        self.play(
            FadeIn(closing_text, shift=DOWN * 0.3),
            formula_power.animate.move_to(ORIGIN),
            run_time=1.6,
        )
        self.play(FadeIn(thanks_text, shift=UP * 0.2), run_time=1.2)
        self.wait(CONFIG["pause_key"])

        self.play(
            FadeOut(closing_text), FadeOut(thanks_text), FadeOut(formula_power),
            run_time=1.5,
        )
        self.subtitle_manager.remove_subtitle_display()

