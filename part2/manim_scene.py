from manim import *
import math
import numpy as np
from typing import Union, List, Dict, Optional, Any, Tuple

# ==============================================================================
# CẤU HÌNH: Các hằng số, Màu sắc, Thời gian, Thiết lập Camera
# ==============================================================================

CONFIG = {
    # Màu sắc
    "color_original": WHITE,
    "color_orthogonal": GREEN,
    "color_orthonormal": YELLOW,
    "color_projection": "#00D4FF",  # Màu xanh lơ (Cyan)
    "color_highlight": "#20B2AA",   # Màu mòng két (Teal)
    "color_eigenvalue_1": RED,
    "color_eigenvalue_2": BLUE,
    "color_eigenvalue_3": "#FF00FF", # Màu đỏ cánh sen (Magenta) cho cặp thứ 3
    "color_matrix_Q": YELLOW,
    "color_matrix_R": "#00FFFF",    # Màu xanh lơ
    "color_matrix_D": ORANGE,
    "color_active_vector": YELLOW,
    "color_dim_vector": GRAY_B,
    
    # Thời gian (giây)
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
    
    # Bố cục (Phân vùng)
    "ZONE_TITLE": UP * 3.5,
    "ZONE_SUBTITLE": DOWN * 3.5,
    "ZONE_VISUAL": ORIGIN,
    "ZONE_MATH": UL * 2.5 + LEFT * 1.5,
    
    # Tỷ lệ, Độ dày nét & Phông chữ
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
# CÁC HÀM TRỢ GIÚP ĐẠI SỐ TUYẾN TÍNH (PYTHON THUẦN)
# ==============================================================================

def calculate_dot_product(vector_a: list, vector_b: list) -> float:
    """
    Tính tích vô hướng của hai vector.
    Được sử dụng cho các tính toán hình chiếu và kiểm tra tính trực giao.
    """
    return sum(val_a * val_b for val_a, val_b in zip(vector_a, vector_b))

def calculate_euclidean_norm(vector_v: list) -> float:
    """
    Tính chuẩn Euclid (độ dài) của một vector.
    Cần thiết cho việc chuẩn hóa các vector về độ dài đơn vị.
    """
    return math.sqrt(calculate_dot_product(vector_v, vector_v))

def multiply_vector_by_scalar(scalar_c: float, vector_v: list) -> list:
    """
    Thực hiện phép nhân vector với một số vô hướng.
    Được sử dụng để thay đổi tỷ lệ vector trong quá trình chiếu và chuẩn hóa.
    """
    return [scalar_c * val_x for val_x in vector_v]

def add_two_vectors(vector_a: list, vector_b: list) -> list:
    """
    Thực hiện phép cộng từng thành phần của hai vector.
    Được sử dụng để tổng hợp các thành phần vector trong các dựng hình hình học.
    """
    return [val_a + val_b for val_a, val_b in zip(vector_a, vector_b)]

def subtract_two_vectors(vector_a: list, vector_b: list) -> list:
    """
    Thực hiện phép trừ từng thành phần của hai vector (vector_a - vector_b).
    Thao tác cốt lõi trong Gram-Schmidt để loại bỏ các hình chiếu.
    """
    return [val_a - val_b for val_a, val_b in zip(vector_a, vector_b)]

def project_vector_onto(vector_u: list, vector_v: list) -> list:
    """
    Chiếu vector vector_u lên vector vector_v.
    Trả về thành phần của vector_u nằm theo hướng của vector_v.
    Xử lý trường hợp vector không để tránh lỗi chia cho 0 trong hoạt ảnh.
    """
    dot_vv = calculate_dot_product(vector_v, vector_v)
    if abs(dot_vv) < 1e-9:
        return [0.0, 0.0, 0.0]
    return multiply_vector_by_scalar(calculate_dot_product(vector_u, vector_v) / dot_vv, vector_v)

def calculate_cross_product_3d(vector_a: list, vector_b: list) -> list:
    """
    Tính tích có hướng của hai vector trong không gian 3 chiều.
    Hữu ích để tìm các vector pháp tuyến của các mặt phẳng trong không gian 3D.
    """
    return [
        vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1],
        vector_a[2] * vector_b[0] - vector_a[0] * vector_b[2],
        vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
    ]

def normalize_to_unit_vector(vector_v: list) -> list:
    """
    Chuẩn hóa một vector để có độ dài bằng 1.
    Chuyển đổi các vector trực giao thành các vector trực chuẩn (vector cơ sở).
    """
    magnitude = calculate_euclidean_norm(vector_v)
    if abs(magnitude) < 1e-9:
        return [0.0, 0.0, 0.0]
    return multiply_vector_by_scalar(1.0 / magnitude, vector_v)

def multiply_matrices_3x3(matrix_a: list, matrix_b: list) -> list:
    """
    Nhân hai ma trận 3x3 bằng thuật toán tích vô hướng tiêu chuẩn.
    Được sử dụng để kiểm chứng kết quả phân rã (QR hoặc PDP^-1).
    """
    result_matrix = [[0.0] * 3 for _ in range(3)]
    for idx_row in range(3):
        for idx_col in range(3):
            for idx_inner in range(3):
                result_matrix[idx_row][idx_col] += matrix_a[idx_row][idx_inner] * matrix_b[idx_inner][idx_col]
    return result_matrix

def calculate_determinant_3x3(matrix_a: list) -> float:
    """
    Tính định thức của ma trận 3x3 bằng quy tắc Sarrus hoặc khai triển.
    Cần thiết để giải phương trình đặc trưng trong chéo hóa ma trận.
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
    
    def update_subtitle_text(self, text_string: str, duration: float = None) -> None:
        """
        Cập nhật văn bản phụ đề hiển thị trên màn hình.
        Loại bỏ phụ đề cũ và thay thế bằng phụ đề mới để tránh chồng lấp.
        """
        if self.current_subtitle_group in self.scene.mobjects:
            self.scene.remove(self.current_subtitle_group)
        
        self.text_object = Text(
            text_string, 
            font_size=CONFIG["subtitle_font_size"], 
            color=WHITE, 
            font=CONFIG["DEFAULT_FONT"]
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
    Tạo nhãn 3D luôn hướng về phía camera (hiệu ứng billboard).
    Điều này đảm bảo các ký hiệu toán học vẫn có thể đọc được khi camera quay.
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
    Tạo một đối tượng mũi tên và hoạt ảnh 'mọc ra từ một điểm'.
    Đóng gói logic hình ảnh để giới thiệu các vector mới.
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
    Áp dụng hiệu ứng rung nhẹ cho camera (hiệu ứng 'thở').
    Ngăn cảnh phim cảm thấy bị tĩnh trong các phần giải thích dài.
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
        Khởi tạo cảnh với bộ quản lý phụ đề và các phân vùng được định nghĩa trước.
        Thiết lập cấu trúc cơ bản cho các hoạt ảnh gồm nhiều chương.
        """
        super().setup()
        self.subtitle_manager = SubtitleManager(self)
        self.layout_zones = {
            "title": CONFIG["ZONE_TITLE"],
            "subtitle": CONFIG["ZONE_SUBTITLE"],
            "visual": CONFIG["ZONE_VISUAL"],
            "math": CONFIG["ZONE_MATH"]
        }
        self.initialize_layout_boundaries()

    def initialize_layout_boundaries(self) -> None:
        """
        Tạo các ranh giới vô hình để duy trì tính nhất quán về hình ảnh giữa các cảnh.
        Chúng đóng vai trò là các neo để đặt tiêu đề và phụ đề.
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
        Xóa khu vực hình ảnh trung tâm trong khi vẫn giữ lại tiêu đề và các công thức toán học tham chiếu.
        Sử dụng một quy tắc dựa trên tọa độ Y để xác định các đối tượng ở trung tâm.
        """
        mobjects_to_remove = []
        for mobject in self.mobjects:
            pos_y = mobject.get_center()[1]
            if -3 < pos_y < 3:
                mobjects_to_remove.append(mobject)
        if mobjects_to_remove:
            self.play(*[FadeOut(m) for m in mobjects_to_remove], run_time=1)

    def create_billboard_label(
        self, 
        tex_string: str, 
        position: np.ndarray, 
        color: Union[str, Any] = WHITE, 
        font_size: int = None
    ) -> MathTex:
        """
        Tạo một nhãn 3D luôn hướng về phía camera một cách linh hoạt.
        Khác với phiên bản độc lập bằng cách sử dụng trạng thái camera của chính cảnh phim.
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
        base_label = label_object.copy()
        
        def update_face_camera_logic(mobject: Mobject) -> None:
            camera_phi = self.camera.get_phi()
            camera_theta = self.camera.get_theta()
            new_mobject = base_label.copy()
            new_mobject.rotate(camera_phi, axis=RIGHT)
            new_mobject.rotate(camera_theta + PI/2, axis=OUT)
            new_mobject.move_to(position)
            mobject.become(new_mobject)
            
        label_object.add_updater(update_face_camera_logic)
        return label_object

    def resize_and_reposition(self, mobject: Mobject, target_zone: str = "math") -> None:
        """
        Chuyển đổi một đối tượng từ trung tâm hình ảnh sang một góc tham chiếu.
        Thường được sử dụng để lưu trữ một phép tính đã hoàn thành để tham khảo sau này.
        """
        self.play(
            mobject.animate.scale(CONFIG["MATH_SCALE"]).move_to(self.layout_zones[target_zone]),
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
        Tạo hoạt ảnh từng bước dựng một vector 3D bằng các thành phần tọa độ của nó.
        Cung cấp một liên kết sư phạm rõ ràng giữa tọa độ và các mũi tên hình học.
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
        Làm nổi bật vector đang hoạt động bằng cách tăng độ đậm và làm mờ các vector khác.
        Tập trung sự chú ý của khán giả trong các quá trình lặp lại như Gram-Schmidt.
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

# ==============================================================================
# LỚP CẢNH CHÍNH (MAIN SCENE)
# ==============================================================================

class QRAndDiagonalization(MatrixProjectScene):
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

        def build_label_logic() -> MathTex:
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
            return label_object

        return always_redraw(build_label_logic)

    def display_projection_formula_panel(self) -> MathTex:
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
        projection_formula.to_corner(UR, buff=0.45)
        self.add_fixed_in_frame_mobjects(projection_formula)
        self.play(Write(projection_formula), run_time=1.7)
        return projection_formula

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

    def construct(self) -> None:
        """
        Luồng xây dựng chính cho toàn bộ dự án.
        Điều phối trình tự của các chương giáo dục.
        """
        self.render_intro_chapter()
        self.wait(CONFIG["chapter_gap"])
        
        self.render_qr_chapter()
        self.wait(CONFIG["chapter_gap"])
        
        self.render_diagonalization_chapter()
        self.wait(CONFIG["chapter_gap"])
        
        self.render_verification_chapter()
        self.wait(CONFIG["chapter_gap"])
        
        self.render_closing_scene()

    def render_intro_chapter(self) -> None:
        """
        Chương 1: Bản chất hình học của Đại số tuyến tính.
        Minh họa trực quan cách một ma trận biến đổi không gian 2D.
        """
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        
        # 1. Thiết lập tiêu đề
        chapter_title = Text("Ý nghĩa hình học của Ma trận", font_size=CONFIG["title_font_size"], font=CONFIG["DEFAULT_FONT"])
        chapter_title.move_to(self.layout_zones["title"])
        self.play_with_consistent_timing(Write(chapter_title))
        
        self.subtitle_manager.update_subtitle_text("Một vector trong 2D đơn giản là một cặp số (x, y).")
        
        # 2. Các trục và lưới tọa độ
        axes_2d = Axes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=7, y_length=7,
            axis_config={"color": GRAY},
            tips=False
        ).scale(CONFIG["INTRO_AXIS_SCALE"])
        coordinate_grid = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1],
            x_length=7, y_length=7,
            background_line_style={"stroke_color": GRAY, "stroke_width": 1, "stroke_opacity": 0.2}
        ).scale(CONFIG["INTRO_AXIS_SCALE"])
        axes_2d.move_to(self.layout_zones["visual"])
        coordinate_grid.move_to(self.layout_zones["visual"])
        self.play(Create(coordinate_grid), Create(axes_2d), run_time=CONFIG["INTRO_DRAW_TIME"])

        label_origin = Text("O", font_size=24, font=CONFIG["DEFAULT_FONT"]).next_to(axes_2d.get_origin(), DL, buff=0.08)
        label_axis_x = Text("x", font_size=24, font=CONFIG["DEFAULT_FONT"]).next_to(axes_2d.c2p(4, 0), DR, buff=0.1)
        label_axis_y = Text("y", font_size=24, font=CONFIG["DEFAULT_FONT"]).next_to(axes_2d.c2p(0, 4), UL, buff=0.1)
        self.play(FadeIn(label_origin), FadeIn(label_axis_x), FadeIn(label_axis_y), run_time=1.5)
        
        # 3. Vector v
        arrow_v = Arrow(axes_2d.get_origin(), axes_2d.c2p(1, 1), buff=0, color=CONFIG["color_original"], stroke_width=CONFIG["VEC_STROKE"])
        label_v = MathTex("v = (1, 1)", color=CONFIG["color_original"], font_size=30, background_stroke_width=2, background_stroke_color=BLACK).next_to(arrow_v, UR, buff=0.1)
        guide_line_vx = DashedLine(axes_2d.c2p(1, 1), axes_2d.c2p(1, 0), color=GRAY_B)
        guide_line_vy = DashedLine(axes_2d.c2p(1, 1), axes_2d.c2p(0, 1), color=GRAY_B)
        
        self.play(GrowArrow(arrow_v), FadeIn(label_v), run_time=CONFIG["VECTOR_DRAW_TIME"])
        self.play(Create(guide_line_vx), Create(guide_line_vy), run_time=1.6)
        self.wait(CONFIG["pause_key"])
        
        # 4. Tham chiếu Ma trận A
        self.subtitle_manager.update_subtitle_text("Ma trận là một bảng số thực hiện phép biến đổi không gian.")
        matrix_mobject_a = Matrix([[2, 1], [1, 2]], left_bracket="(", right_bracket=")")
        matrix_mobject_a.set_stroke(width=CONFIG["MATRIX_STROKE"])
        matrix_mobject_a.get_entries().set_stroke(width=CONFIG["MATRIX_STROKE"])
        matrix_group = VGroup(MathTex("A = "), matrix_mobject_a).arrange(RIGHT).scale(CONFIG["MATH_SCALE"])
        matrix_group[0].set_stroke(width=CONFIG["MATRIX_STROKE"])
        matrix_group.move_to(self.layout_zones["math"])
        self.play_with_consistent_timing(FadeIn(matrix_group))
        self.wait(CONFIG["pause_key"])
        
        # 5. Phép biến đổi
        self.subtitle_manager.update_subtitle_text("Khi nhân ma trận A với v, diện mạo không gian và vector bị thay đổi.")
        transformation_matrix_np = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 1]])
        guide_line_avx = DashedLine(axes_2d.c2p(3, 3), axes_2d.c2p(3, 0), color=GRAY_B)
        guide_line_avy = DashedLine(axes_2d.c2p(3, 3), axes_2d.c2p(0, 3), color=GRAY_B)
        label_av = MathTex("Av = (3, 3)", color=CONFIG["color_highlight"], font_size=30, background_stroke_width=2, background_stroke_color=BLACK).move_to(axes_2d.c2p(3, 3) + UR * 0.25)
        
        self.play(
            ApplyMatrix(transformation_matrix_np, coordinate_grid),
            ApplyMatrix(transformation_matrix_np, arrow_v),
            AnimationGroup(
                ReplacementTransform(guide_line_vx, guide_line_avx),
                ReplacementTransform(guide_line_vy, guide_line_avy),
                lag_ratio=0.0
            ),
            run_time=CONFIG["VECTOR_TRANSFORM_TIME"]
        )
        self.play(TransformMatchingTex(label_v, label_av), run_time=1.6)
        self.wait(CONFIG["pause_key"])
        
        # 6. Giải thích về Tổ hợp tuyến tính
        self.subtitle_manager.update_subtitle_text("Nhân với ma trận là tổ hợp tuyến tính của các cột: Av = x·c1 + y·c2.")
        combination_theory_formula = MathTex(
            r"A \mathbf{v} = 1 \cdot \begin{pmatrix} 2 \\ 1 \end{pmatrix} + 1 \cdot \begin{pmatrix} 1 \\ 2 \end{pmatrix}",
            " = \\begin{pmatrix} 3 \\\\ 3 \\end{pmatrix}",
            font_size=34,
            background_stroke_width=2,
            background_stroke_color=BLACK
        ).next_to(matrix_group, DOWN, buff=0.5)
        self.play(Write(combination_theory_formula))
        self.wait(CONFIG["pause_key"])
        
        # 7. Dọn dẹp
        self.play(
            AnimationGroup(
                FadeOut(arrow_v), FadeOut(label_av), FadeOut(guide_line_avx), FadeOut(guide_line_avy),
                FadeOut(matrix_group), FadeOut(combination_theory_formula), FadeOut(coordinate_grid), FadeOut(axes_2d),
                FadeOut(label_origin), FadeOut(label_axis_x), FadeOut(label_axis_y),
                FadeOut(chapter_title),
                lag_ratio=0.04
            ),
            run_time=1.8
        )
    
    def render_qr_chapter(self) -> None:
        """
        Chương 2: Phân rã QR thông qua Gram-Schmidt.
        Trực quan hóa 3D chi tiết về quá trình trực giao hóa và chuẩn hóa.
        """
        # 1. Tiêu đề
        chapter_title = Text("Phân rã QR: Trực quan hóa Gram-Schmidt", font_size=CONFIG["title_font_size"], font=CONFIG["DEFAULT_FONT"])
        chapter_title.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(chapter_title)
        self.play_with_consistent_timing(Write(chapter_title))
        
        # 2. Giới thiệu lý thuyết QR
        self.subtitle_manager.update_subtitle_text("Mục tiêu: Tách A thành Q (Xoay/Trực chuẩn) và R (Nén/Tam giác).")
        qr_decomposition_theory = VGroup(
            Text("Q: Ma trận trực giao (Phép xoay/Trực chuẩn)", font=CONFIG["DEFAULT_FONT"], font_size=24, color=CONFIG["color_matrix_Q"]),
            Text("R: Ma trận tam giác trên (Nén/Co giãn)", font=CONFIG["DEFAULT_FONT"], font_size=24, color=CONFIG["color_matrix_R"])
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).shift(UP * 0.5)
        self.add_fixed_in_frame_mobjects(qr_decomposition_theory)
        self.play(FadeIn(qr_decomposition_theory, shift=UP))
        self.wait(2)
        
        self.subtitle_manager.update_subtitle_text("Ý nghĩa: 'Bẻ thẳng' các vector ban đầu cho vuông góc với nhau.")
        self.wait(CONFIG["pause_key"])
        self.play(FadeOut(qr_decomposition_theory))

        # Ẩn tiêu đề chương trước khi vào sân khấu 3D để tránh che mất nhãn của các trục.
        self.play(FadeOut(chapter_title), run_time=0.8)

        # 3. Giữ lại tham chiếu ma trận A ở góc màn hình.
        matrix_a_reference_mobject = Matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]], left_bracket="(", right_bracket=")").scale(0.42)
        matrix_a_display_group = VGroup(MathTex("A = "), matrix_a_reference_mobject).arrange(RIGHT)
        matrix_a_display_group.to_corner(UL, buff=0.45)
        self.add_fixed_in_frame_mobjects(matrix_a_display_group)
        self.play(FadeIn(matrix_a_display_group), run_time=1.2)
        
        column_highlight_rect = None

        def highlight_matrix_column(column_index: int) -> None:
            """Di chuyển khung tập trung vào một cột cụ thể của ma trận A."""
            nonlocal column_highlight_rect
            target_rectangle = SurroundingRectangle(matrix_a_reference_mobject.get_columns()[column_index], color=YELLOW, buff=0.06)
            if column_highlight_rect is None:
                column_highlight_rect = target_rectangle
                self.add_fixed_in_frame_mobjects(column_highlight_rect)
                self.play(Create(column_highlight_rect), run_time=0.5)
            else:
                self.play(Transform(column_highlight_rect, target_rectangle), run_time=0.5)
        
        # 4. Quá trình Gram-Schmidt trong không gian 3D
        self.subtitle_manager.update_subtitle_text("Sử dụng Gram-Schmidt để tìm các vector trực chuẩn q1, q2, q3.")
        
        # Thiết lập góc nhìn 3D
        self.set_camera_orientation(phi=CONFIG["camera_phi_main"], theta=CONFIG["camera_theta_main"])
        val_vector_a1 = [2, 1, 1]
        val_vector_a2 = [1, 2, 1]
        val_vector_a3 = [1, 1, 2]
        axes_specifications = self.calculate_dynamic_axes_spec([val_vector_a1, val_vector_a2, val_vector_a3])
        threed_axes = ThreeDAxes(
            x_range=axes_specifications["x_range"],
            y_range=axes_specifications["y_range"],
            z_range=axes_specifications["z_range"],
            x_length=axes_specifications["axis_length"],
            y_length=axes_specifications["axis_length"],
            z_length=axes_specifications["axis_length"],
            axis_config={"color": GRAY},
        )
        self.play(Create(threed_axes), run_time=2.6)

        axes_bound = axes_specifications["bound"]
        label_axis_x_3d = self.create_billboard_label("x", threed_axes.c2p(axes_bound + 0.3, 0, 0), color=WHITE, font_size=30)
        label_axis_y_3d = self.create_billboard_label("y", threed_axes.c2p(0, axes_bound + 0.3, 0), color=WHITE, font_size=30)
        label_axis_z_3d = self.create_billboard_label("z", threed_axes.c2p(0, 0, axes_bound + 0.3), color=WHITE, font_size=30)
        label_axis_o_3d = self.create_billboard_label("O", threed_axes.c2p(0, 0, 0) + LEFT * 0.2 + DOWN * 0.2, color=WHITE, font_size=28)
        axes_labels_group = VGroup(label_axis_x_3d, label_axis_y_3d, label_axis_z_3d, label_axis_o_3d)
        self.play(*[FadeIn(label_obj) for label_obj in axes_labels_group], run_time=1.4)

        projection_formula_panel = self.display_projection_formula_panel()
        self.wait(CONFIG["pause_formula_long"])

        # Bước 1: Dựng vector a1 chi tiết
        self.subtitle_manager.update_subtitle_text("Bước 1: Vẽ vector a1 và chuẩn hóa để được q1.")
        highlight_matrix_column(0)
        arrow_a1, label_a1, construction_a1 = self.render_detailed_3d_vector(threed_axes, val_vector_a1, color=CONFIG["color_original"], label_string="a_1")
        self.highlight_active_vector(arrow_a1, [])
        self.wait(CONFIG["pause_key"])
        
        val_vector_q1 = normalize_to_unit_vector(val_vector_a1)
        arrow_q1 = Arrow3D(start=ORIGIN, end=threed_axes.c2p(*val_vector_q1), color=CONFIG["color_matrix_Q"])
        label_q1 = self.create_tracking_billboard_label("q_1", arrow_q1, color=CONFIG["color_matrix_Q"], offset_vector=DOWN * 0.55)
        
        self.play(
            ReplacementTransform(arrow_a1.copy(), arrow_q1),
            FadeIn(label_q1),
            arrow_a1.animate.set_opacity(0.3),
            FadeOut(construction_a1)
        )
        self.highlight_active_vector(arrow_q1, [arrow_a1])
        self.wait(CONFIG["pause_key"])
        
        # Bước 2: Trực giao hóa a2
        self.subtitle_manager.update_subtitle_text("Bước 2: Tìm phần trực giao của a2 so với q1, sau đó chuẩn hóa.")
        highlight_matrix_column(1)
        arrow_a2, label_a2, construction_a2 = self.render_detailed_3d_vector(threed_axes, val_vector_a2, color=CONFIG["color_original"], label_string="a_2")
        self.highlight_active_vector(arrow_a2, [arrow_a1, arrow_q1])

        formula_step_e2 = self.display_orthogonal_formula(r"e_2 = a_2 - \operatorname{proj}_{q_1}(a_2),\quad q_2=\frac{e_2}{\|e_2\|}")
        self.wait(CONFIG["pause_formula_long"])
        
        # Các đường kẻ hình chiếu
        val_projection_p1 = project_vector_onto(val_vector_a2, val_vector_q1)
        projection_line_a2 = DashedLine(threed_axes.c2p(*val_projection_p1), threed_axes.c2p(*val_vector_a2), color=GRAY)
        self.play(Create(projection_line_a2), run_time=1.8)
        
        val_residual_e2 = subtract_two_vectors(val_vector_a2, val_projection_p1)
        val_vector_q2 = normalize_to_unit_vector(val_residual_e2)
        arrow_q2 = Arrow3D(start=ORIGIN, end=threed_axes.c2p(*val_vector_q2), color=CONFIG["color_matrix_Q"])
        label_q2 = self.create_tracking_billboard_label("q_2", arrow_q2, color=CONFIG["color_matrix_Q"], offset_vector=RIGHT * 0.3)
        self.play(Create(arrow_q2), FadeIn(label_q2), FadeOut(construction_a2), run_time=2.0)
        self.highlight_active_vector(arrow_q2, [arrow_a1, arrow_a2, arrow_q1])
        self.play(FadeOut(formula_step_e2), run_time=0.9)
        self.wait(CONFIG["pause_key"])

        self.subtitle_manager.update_subtitle_text("Bước 3: Tách a3 thành phần nằm trên mặt phẳng q1-q2 và phần trực giao để dựng q3.")
        highlight_matrix_column(2)
        arrow_a3, label_a3, construction_a3 = self.render_detailed_3d_vector(threed_axes, val_vector_a3, color=CONFIG["color_original"], label_string="a_3")
        self.highlight_active_vector(arrow_a3, [arrow_a1, arrow_a2, arrow_q1, arrow_q2])
        formula_step_e3 = self.display_orthogonal_formula(r"e_3 = a_3 - \operatorname{proj}_{q_1}(a_3) - \operatorname{proj}_{q_2}(a_3)")
        self.wait(CONFIG["pause_formula_long"])

        val_projection_p31 = project_vector_onto(val_vector_a3, val_vector_q1)
        val_projection_p32 = project_vector_onto(val_vector_a3, val_vector_q2)
        val_projection_on_plane = add_two_vectors(val_projection_p31, val_projection_p32)
        val_residual_e3 = subtract_two_vectors(subtract_two_vectors(val_vector_a3, val_projection_p31), val_projection_p32)

        plane_scaling_factor = 1.45
        basis_plane_mobject = Polygon(
            threed_axes.c2p(*add_two_vectors(multiply_vector_by_scalar(plane_scaling_factor, val_vector_q1), multiply_vector_by_scalar(plane_scaling_factor, val_vector_q2))),
            threed_axes.c2p(*add_two_vectors(multiply_vector_by_scalar(-plane_scaling_factor, val_vector_q1), multiply_vector_by_scalar(plane_scaling_factor, val_vector_q2))),
            threed_axes.c2p(*add_two_vectors(multiply_vector_by_scalar(-plane_scaling_factor, val_vector_q1), multiply_vector_by_scalar(-plane_scaling_factor, val_vector_q2))),
            threed_axes.c2p(*add_two_vectors(multiply_vector_by_scalar(plane_scaling_factor, val_vector_q1), multiply_vector_by_scalar(-plane_scaling_factor, val_vector_q2))),
            color=CONFIG["color_projection"],
            fill_color=CONFIG["color_projection"],
            fill_opacity=0.18,
            stroke_opacity=0.45,
            stroke_width=2,
        )

        arrow_plane_projection = Arrow3D(start=ORIGIN, end=threed_axes.c2p(*val_projection_on_plane), color=GRAY_B)
        label_plane_projection = self.create_billboard_label("proj_{q_1,q_2}(a_3)", threed_axes.c2p(*val_projection_on_plane) + LEFT * 0.25, color=GRAY_B, font_size=30)
        residual_projection_line = DashedLine(threed_axes.c2p(*val_projection_on_plane), threed_axes.c2p(*val_vector_a3), color=CONFIG["color_projection"])
        arrow_u3 = Arrow3D(start=ORIGIN, end=threed_axes.c2p(*val_residual_e3), color=CONFIG["color_projection"])
        label_u3 = self.create_tracking_billboard_label("u_3", arrow_u3, color=CONFIG["color_projection"], offset_vector=RIGHT * 0.2)
        self.play(FadeIn(basis_plane_mobject), Create(arrow_plane_projection), FadeIn(label_plane_projection), Create(residual_projection_line), run_time=1.8)
        self.play(Create(arrow_u3), FadeIn(label_u3), run_time=1.8)

        val_vector_q3 = normalize_to_unit_vector(val_residual_e3)
        arrow_q3 = Arrow3D(start=ORIGIN, end=threed_axes.c2p(*val_vector_q3), color=CONFIG["color_matrix_Q"])
        label_q3 = self.create_tracking_billboard_label("q_3", arrow_q3, color=CONFIG["color_matrix_Q"], offset_vector=LEFT * 0.25)
        self.play(
            ReplacementTransform(arrow_u3, arrow_q3),
            FadeIn(label_q3),
            arrow_a3.animate.set_opacity(0.3),
            FadeOut(label_u3),
            FadeOut(formula_step_e3),
            FadeOut(arrow_plane_projection),
            FadeOut(label_plane_projection),
            FadeOut(basis_plane_mobject),
            FadeOut(residual_projection_line),
            FadeOut(construction_a3),
            run_time=2.2
        )
        self.highlight_active_vector(arrow_q3, [arrow_a1, arrow_a2, arrow_a3, arrow_q1, arrow_q2])
        self.wait(CONFIG["pause_scene_detail"])

        self.subtitle_manager.update_subtitle_text("Ma trận R lưu các tích vô hướng ở tam giác trên, còn phía dưới đường chéo là 0.")
        self.move_camera(
            phi=CONFIG["camera_phi_topdown"],
            theta=CONFIG["camera_theta_topdown"],
            run_time=CONFIG["animation_camera_long"],
        )
        matrix_r_explanation_group = self.describe_r_upper_triangular_structure(
            [val_vector_a1, val_vector_a2, val_vector_a3],
            [val_vector_q1, val_vector_q2, val_vector_q3],
            matrix_a_display_group,
        )
        self.move_camera(
            phi=CONFIG["camera_phi_main"],
            theta=CONFIG["camera_theta_main"],
            run_time=CONFIG["animation_camera_long"],
        )
        self.play(FadeOut(matrix_r_explanation_group), FadeOut(projection_formula_panel), run_time=1.1)
        
        # Dọn dẹp các đối tượng của chương
        self.play(
            AnimationGroup(
                FadeOut(threed_axes), FadeOut(arrow_q1), FadeOut(arrow_q2), FadeOut(arrow_q3),
                FadeOut(arrow_a1), FadeOut(arrow_a2), FadeOut(arrow_a3),
                FadeOut(label_q1), FadeOut(label_q2), FadeOut(label_q3),
                FadeOut(label_a1), FadeOut(label_a2), FadeOut(label_a3), FadeOut(projection_line_a2),
                FadeOut(axes_labels_group), FadeOut(matrix_a_display_group), FadeOut(column_highlight_rect),
                lag_ratio=0.03
            ),
            run_time=2.0
        )

    def render_diagonalization_chapter(self) -> None:
        """
        Chương 3: Chéo hóa - A = PDP⁻¹.
        Từng bước tìm các trị riêng và vector riêng.
        """
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        chapter_title = Text("Quy trình chéo hóa", font_size=CONFIG["title_font_size"], font=CONFIG["DEFAULT_FONT"])
        chapter_title.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(chapter_title)
        self.play_with_consistent_timing(Write(chapter_title))

        wait_time_fast = 1.6
        run_time_fast = 1.5

        self.subtitle_manager.update_subtitle_text("Đầu tiên ta giải phương trình đặc trưng để tìm các trị riêng.")
        formula_char_eq_start = MathTex(r"\det(A - \lambda I) = 0", font_size=46).shift(UP * 1.4)
        formula_char_eq_det = MathTex(r"\begin{vmatrix} 2-\lambda & 1 & 1 \\ 1 & 2-\lambda & 1 \\ 1 & 1 & 2-\lambda \end{vmatrix} = 0", font_size=40).shift(UP * 1.4)
        formula_char_eq_poly = MathTex(r"(2-\lambda)^3 - 3(2-\lambda) + 2 = 0", font_size=40).shift(UP * 1.4)
        formula_char_eq_factored = MathTex(r"-(\lambda - 4)(\lambda - 1)^2 = 0", font_size=46).shift(UP * 1.4)
        formula_eigenvalues_result = MathTex(
            r"\lambda_1 = ", "4", r",\; \lambda_2 = ", "1", r",\; \lambda_3 = ", "1",
            font_size=40,
            color=YELLOW
        ).shift(UP * 1.4)

        self.play(Write(formula_char_eq_start))
        self.wait(wait_time_fast)
        self.play(ReplacementTransform(formula_char_eq_start, formula_char_eq_det), run_time=run_time_fast)
        self.wait(wait_time_fast)
        self.play(ReplacementTransform(formula_char_eq_det, formula_char_eq_poly), run_time=run_time_fast)
        self.wait(wait_time_fast)
        self.play(ReplacementTransform(formula_char_eq_poly, formula_char_eq_factored), run_time=run_time_fast)
        self.wait(wait_time_fast)
        self.play(ReplacementTransform(formula_char_eq_factored, formula_eigenvalues_result), run_time=run_time_fast)
        self.wait(0.8)
        self.play(formula_eigenvalues_result.animate.next_to(chapter_title, DOWN, buff=0.35), run_time=1.2)

        self.subtitle_manager.update_subtitle_text("Với λ1 = 4, ta đi qua dạng bậc thang để thấy rõ ràng điều kiện x=y=z.")
        formula_eigenvector_l1_solution = self.display_row_echelon_steps(
            r"(A - 4I)\mathbf{v}=0",
            r"\left[\begin{array}{ccc|c}-2&1&1&0\\1&-2&1&0\\1&1&-2&0\end{array}\right]\sim"
            r"\left[\begin{array}{ccc|c}1&0&-1&0\\0&1&-1&0\\0&0&0&0\end{array}\right]",
            r"x=z,\;y=z\Rightarrow x=y=z\Rightarrow \mathbf{v}_1=(1,1,1)",
            CONFIG["color_eigenvalue_1"],
        )

        self.subtitle_manager.update_subtitle_text("Với λ2 = λ3 = 1, dạng bậc thang cho thấy không gian nghiệm có 2 bậc tự do.")
        self.play(formula_eigenvector_l1_solution.animate.shift(UP * 0.05), run_time=0.6)
        formula_eigenvector_l23_solution = self.display_row_echelon_steps(
            r"(A-I)\mathbf{v}=0",
            r"\left[\begin{array}{ccc|c}1&1&1&0\\1&1&1&0\\1&1&1&0\end{array}\right]\sim"
            r"\left[\begin{array}{ccc|c}1&1&1&0\\0&0&0&0\\0&0&0&0\end{array}\right]",
            r"x+y+z=0\Rightarrow \dim E_{\lambda=1}=2,\;\mathbf{v}_2=(-1,1,0),\;\mathbf{v}_3=(-1,0,1)",
            CONFIG["color_eigenvalue_2"],
        )

        self.subtitle_manager.update_subtitle_text("Từ đó thu được ba vector riêng để dựng ma trận P.")
        label_v1 = MathTex(r"v_1 = ", color=CONFIG["color_eigenvalue_1"])
        matrix_v1_vec = Matrix([[1], [1], [1]], left_bracket="(", right_bracket=")")
        matrix_v1_vec.get_entries().set_color(CONFIG["color_eigenvalue_1"])
        v1_display_group = VGroup(label_v1, matrix_v1_vec).arrange(RIGHT)

        label_v2 = MathTex(r"v_2 = ", color=CONFIG["color_eigenvalue_2"])
        matrix_v2_vec = Matrix([[-1], [1], [0]], left_bracket="(", right_bracket=")")
        matrix_v2_vec.get_entries().set_color(CONFIG["color_eigenvalue_2"])
        v2_display_group = VGroup(label_v2, matrix_v2_vec).arrange(RIGHT)

        label_v3 = MathTex(r"v_3 = ", color=CONFIG["color_eigenvalue_3"])
        matrix_v3_vec = Matrix([[-1], [0], [1]], left_bracket="(", right_bracket=")")
        matrix_v3_vec.get_entries().set_color(CONFIG["color_eigenvalue_3"])
        v3_display_group = VGroup(label_v3, matrix_v3_vec).arrange(RIGHT)

        eigenvectors_display_group = VGroup(v1_display_group, v2_display_group, v3_display_group).arrange(RIGHT, buff=0.8).scale(0.9).move_to(ORIGIN + UP * 0.1)
        self.play(FadeOut(formula_eigenvector_l23_solution), FadeIn(eigenvectors_display_group, shift=UP), run_time=1.4)
        self.wait(wait_time_fast)

        self.subtitle_manager.update_subtitle_text("Hiện khung rỗng của P và D, sau đó để vector và trị riêng bay vào vị trí.")
        matrix_mobject_p = Matrix([[1, -1, -1], [1, 1, 0], [1, 0, 1]], left_bracket="(", right_bracket=")")
        matrix_mobject_d = Matrix([[4, 0, 0], [0, 1, 0], [0, 0, 1]], left_bracket="(", right_bracket=")")
        matrix_mobject_p.get_entries().set_opacity(0)
        matrix_mobject_d.get_entries().set_opacity(0)

        matrix_p_group = VGroup(MathTex("P = "), matrix_mobject_p).arrange(RIGHT)
        matrix_d_group = VGroup(MathTex("D = "), matrix_mobject_d).arrange(RIGHT)
        matrices_display_group = VGroup(matrix_p_group, matrix_d_group).arrange(RIGHT, buff=1.0).shift(DOWN * 2.0)

        self.play(
            FadeIn(matrix_p_group[0]), FadeIn(matrix_mobject_p.get_brackets()),
            FadeIn(matrix_d_group[0]), FadeIn(matrix_mobject_d.get_brackets()),
            run_time=1.4
        )
        self.wait(1.0)

        p_column_mobjects = matrix_mobject_p.get_columns()
        p_column_mobjects[0].set_color(CONFIG["color_eigenvalue_1"]).set_opacity(0)
        p_column_mobjects[1].set_color(CONFIG["color_eigenvalue_2"]).set_opacity(0)
        p_column_mobjects[2].set_color(CONFIG["color_eigenvalue_3"]).set_opacity(0)

        self.play(
            TransformFromCopy(matrix_v1_vec.get_entries(), p_column_mobjects[0].set_opacity(1)),
            run_time=1.1
        )
        self.play(
            TransformFromCopy(matrix_v2_vec.get_entries(), p_column_mobjects[1].set_opacity(1)),
            run_time=1.1
        )
        self.play(
            TransformFromCopy(matrix_v3_vec.get_entries(), p_column_mobjects[2].set_opacity(1)),
            run_time=1.1
        )
        self.play(FadeOut(eigenvectors_display_group), run_time=0.8)
        self.wait(0.6)

        d_diagonal_entries = VGroup(matrix_mobject_d.get_entries()[0], matrix_mobject_d.get_entries()[4], matrix_mobject_d.get_entries()[8])
        d_diagonal_entries[0].set_color(CONFIG["color_eigenvalue_1"]).set_opacity(1)
        d_diagonal_entries[1].set_color(CONFIG["color_eigenvalue_2"]).set_opacity(1)
        d_diagonal_entries[2].set_color(CONFIG["color_eigenvalue_3"]).set_opacity(1)
        d_zero_entries = VGroup(*[matrix_mobject_d.get_entries()[idx] for idx in [1, 2, 3, 5, 6, 7]])
        d_zero_entries.set_opacity(1)

        self.play(
            TransformFromCopy(formula_eigenvalues_result[1], d_diagonal_entries[0]),
            TransformFromCopy(formula_eigenvalues_result[3], d_diagonal_entries[1]),
            TransformFromCopy(formula_eigenvalues_result[5], d_diagonal_entries[2]),
            FadeIn(d_zero_entries),
            run_time=1.8
        )
        self.wait(wait_time_fast)

        self.play(FadeOut(formula_eigenvalues_result), matrices_display_group.animate.move_to(UP * 1.2), run_time=1.2)

        formula_final_diagonalization = MathTex(r"A = P D P^{-1}", color=YELLOW, font_size=48).next_to(matrices_display_group, DOWN, buff=0.5)
        self.play(Write(formula_final_diagonalization), run_time=1.2)
        self.wait(1.0)

        bridge_explanation_title = Text("Cầu nối học thuật: Q và P đều là đổi cơ sở", font=CONFIG["DEFAULT_FONT"], font_size=24)
        bridge_explanation_title.next_to(chapter_title, DOWN, buff=0.9)
        bridge_explanation_q = Text("Q: đổi sang cơ sở trực chuẩn (đẹp về hình học)", font=CONFIG["DEFAULT_FONT"], font_size=22, color=CONFIG["color_matrix_Q"])
        bridge_explanation_p = Text("P: đổi sang cơ sở vector riêng (đẹp về đại số)", font=CONFIG["DEFAULT_FONT"], font_size=22, color=CONFIG["color_highlight"])
        bridge_display_group = VGroup(bridge_explanation_title, bridge_explanation_q, bridge_explanation_p).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        bridge_display_group.move_to(DOWN * 0.2 + LEFT * 1.9)
        self.add_fixed_in_frame_mobjects(bridge_display_group)
        self.play(FadeIn(bridge_display_group, shift=RIGHT * 0.2), run_time=1.4)
        self.wait(CONFIG["pause_formula_long"])

        matrix_p_copy = matrix_mobject_p.copy().scale(0.6)
        matrix_d_copy = matrix_mobject_d.copy().scale(0.6)
        matrix_p_inv_copy = matrix_mobject_p.copy().scale(0.6)
        label_a_equals = MathTex("A = ", font_size=40)
        expanded_diagonalization_group = VGroup(label_a_equals, matrix_p_copy, matrix_d_copy, matrix_p_inv_copy).arrange(RIGHT, buff=0.1)
        expanded_diagonalization_group.next_to(formula_final_diagonalization, DOWN, buff=0.4)
        label_inverse_exponent = MathTex("^{-1}", font_size=36)
        label_inverse_exponent.next_to(matrix_p_inv_copy.get_brackets()[1], UR, buff=0.05).shift(DOWN * 0.3 + LEFT * 0.1)
        self.play(FadeIn(expanded_diagonalization_group, shift=UP), FadeIn(label_inverse_exponent), run_time=1.4)
        self.wait(wait_time_fast)

        formula_chain_expr = MathTex(
            r"P^{-1}\mathbf{v}\;\xrightarrow{\;D\;}\;D(P^{-1}\mathbf{v)}\;\xrightarrow{\;P\;}\;PDP^{-1}\mathbf{v}",
            font_size=34,
            color=CONFIG["color_matrix_D"],
        ).next_to(expanded_diagonalization_group, DOWN, buff=0.3)
        formula_confirmation_result = MathTex(r"PDP^{-1}=A", font_size=44, color=GREEN).move_to(formula_chain_expr)
        self.play(Write(formula_chain_expr), run_time=1.6)
        self.wait(CONFIG["pause_formula_short"])
        self.play(TransformMatchingTex(formula_chain_expr, formula_confirmation_result), run_time=1.6)
        self.wait(CONFIG["pause_formula_short"])

        self.play(
            AnimationGroup(
                FadeOut(chapter_title), FadeOut(matrices_display_group),
                FadeOut(formula_final_diagonalization), FadeOut(expanded_diagonalization_group), FadeOut(label_inverse_exponent),
                FadeOut(formula_confirmation_result), FadeOut(bridge_display_group),
                lag_ratio=0.03
            ),
            run_time=1.8
        )

    def render_verification_chapter(self) -> None:
        """
        Chương 4: Kiểm chứng số học.
        Kiểm tra cuối cùng để đảm bảo kết quả phân rã là chính xác.
        """
        chapter_title = Text("Kiểm chứng kết quả", font_size=CONFIG["title_font_size"], font=CONFIG["DEFAULT_FONT"])
        chapter_title.move_to(self.layout_zones["title"])
        self.play_with_consistent_timing(Write(chapter_title))
        
        self.subtitle_manager.update_subtitle_text("Rút gọn kiểm chứng: trước hết tính PD.")
        formula_verification_pd = MathTex(
            r"PD = \begin{pmatrix} 4 & -1 & -1 \\ 4 & 1 & 0 \\ 4 & 0 & 1 \end{pmatrix}",
            font_size=34
        ).move_to(UP * 0.8)
        self.play(Write(formula_verification_pd), run_time=1.3)
        self.wait(1.3)

        self.subtitle_manager.update_subtitle_text("Tiếp theo nhân với P^{-1} để quay lại ma trận gốc.")
        formula_verification_pdp_inv = MathTex(
            r"(PD)P^{-1} = \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix}",
            font_size=34,
            color=GREEN
        ).move_to(ORIGIN)
        self.play(ReplacementTransform(formula_verification_pd.copy(), formula_verification_pdp_inv), run_time=1.4)
        self.wait(1.2)

        self.subtitle_manager.update_subtitle_text("So sánh với A ban đầu: kết quả trùng khớp hoàn toàn.")
        formula_original_a_check = MathTex(
            r"A = \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix}",
            font_size=34,
            color=YELLOW
        ).move_to(DOWN * 0.9)
        self.play(Write(formula_original_a_check), run_time=1.3)
        self.wait(CONFIG["pause_key"])

        self.play(FadeOut(chapter_title, formula_verification_pd, formula_verification_pdp_inv, formula_original_a_check), run_time=1.2)

    def render_closing_scene(self) -> None:
        """Thông điệp kết thúc."""
        self.subtitle_manager.update_subtitle_text("Cảm ơn các bạn đã theo dõi bài học!")
        thanks_mobject = Text("Cảm ơn các bạn đã theo dõi!", font="Arial", font_size=40)
        thanks_mobject.move_to(self.layout_zones["visual"])
        self.play_with_consistent_timing(Write(thanks_mobject))
        self.wait(CONFIG["pause_key"])
        self.play(FadeOut(thanks_mobject))
        self.subtitle_manager.remove_subtitle_display()