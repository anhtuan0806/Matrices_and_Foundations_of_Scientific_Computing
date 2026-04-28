import math
from typing import List, Tuple

# Hằng số quy định ngưỡng sai số làm tròn cho tính toán số thực
EPSILON = 1e-12

def get_matrix_dimensions(matrix_input: List[List[float]]) -> Tuple[int, int]:
    """Trả về số dòng và số cột của ma trận."""
    if not matrix_input:
        return 0, 0
    return len(matrix_input), len(matrix_input[0])

def create_zero_matrix(rows_count: int, cols_count: int) -> List[List[float]]:
    """Tạo ma trận toàn số 0 với kích thước rows x cols."""
    return [[0.0 for col_idx in range(cols_count)] for row_idx in range(rows_count)]

def extract_matrix_column(matrix_input: List[List[float]], column_index: int) -> List[float]:
    """Trích xuất một cột của ma trận dưới dạng vector."""
    return [row[column_index] for row in matrix_input]

def calculate_dot_product(vector_1: List[float], vector_2: List[float]) -> float:
    """Tính tích vô hướng của hai vector."""
    return sum(elem_1 * elem_2 for elem_1, elem_2 in zip(vector_1, vector_2))

def calculate_vector_norm(vector_input: List[float]) -> float:
    """Tính chuẩn Euclid (L2 norm) của vector."""
    return math.sqrt(calculate_dot_product(vector_input, vector_input))

def perform_matrix_multiplication(matrix_A: List[List[float]], matrix_B: List[List[float]]) -> List[List[float]]:
    """Thực hiện nhân hai ma trận A (m x n) và B (n x p)."""
    rows_A, cols_A = get_matrix_dimensions(matrix_A)
    rows_B, cols_B = get_matrix_dimensions(matrix_B)
    if cols_A != rows_B:
        raise ValueError("Kích thước ma trận không khớp để thực hiện phép nhân!")
    
    result_matrix = create_zero_matrix(rows_A, cols_B)
    for row_index in range(rows_A):
        for col_index in range(cols_B):
            dot_product_val = sum(matrix_A[row_index][inner_index] * matrix_B[inner_index][col_index] for inner_index in range(cols_A))
            result_matrix[row_index][col_index] = dot_product_val
    return result_matrix

def get_matrix_transpose(matrix_input: List[List[float]]) -> List[List[float]]:
    """Trả về ma trận chuyển vị."""
    rows_count, cols_count = get_matrix_dimensions(matrix_input)
    return [[matrix_input[row_index][col_index] for row_index in range(rows_count)] for col_index in range(cols_count)]

def display_matrix(label_str: str, matrix_input: List[List[float]]) -> None:
    """In ma trận ra màn hình với định dạng đẹp."""
    print(f"{label_str}:")
    for row in matrix_input:
        print("[" + ", ".join(f"{element_val:8.4f}" for element_val in row) + "]")
    print()

def perform_qr_decomposition(matrix_A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Phân rã QR sử dụng phương pháp Gram-Schmidt cải tiến (Modified Gram-Schmidt - MGS).
    A (m x n) = Q (m x n) * R (n x n).
    
    Lý thuyết: MGS ổn định hơn về mặt số học so với Gram-Schmidt cổ điển (CGS) vì nó
    giảm thiểu sự tích lũy sai số làm tròn, giúp ma trận Q giữ được tính trực chuẩn tốt hơn.
    """
    rows_count, cols_count = get_matrix_dimensions(matrix_A)
    if rows_count < cols_count:
        raise ValueError("Số dòng phải lớn hơn hoặc bằng số cột (m >= n) cho phân rã QR chuẩn.")
        
    # Tạo bản sao của các cột trong A để thực hiện trực giao hóa
    # Trong MGS, chúng ta cập nhật trực tiếp các vector còn lại sau mỗi bước
    vectors_V = [extract_matrix_column(matrix_A, col_idx) for col_idx in range(cols_count)]
    matrix_Q = create_zero_matrix(rows_count, cols_count)
    matrix_R = create_zero_matrix(cols_count, cols_count)
    
    for col_index in range(cols_count):
        # Bước 1: Tính chuẩn của vector hiện tại và gán vào đường chéo của R
        matrix_R[col_index][col_index] = calculate_vector_norm(vectors_V[col_index])
        
        if matrix_R[col_index][col_index] > EPSILON:
            # Bước 2: Chuẩn hóa vector để tạo cột cho Q
            for row_index in range(rows_count):
                matrix_Q[row_index][col_index] = vectors_V[col_index][row_index] / matrix_R[col_index][col_index]
        else:
            # Xử lý trường hợp các cột phụ thuộc tuyến tính
            for row_index in range(rows_count):
                matrix_Q[row_index][col_index] = 0.0
                
        # Bước 3: Trực giao hóa các vector còn lại đối với vector q_i vừa tìm được
        # Đây là điểm khác biệt của MGS: cập nhật ngay lập tức các vector chưa xử lý
        basis_vector_qi = extract_matrix_column(matrix_Q, col_index)
        for next_col_index in range(col_index + 1, cols_count):
            matrix_R[col_index][next_col_index] = calculate_dot_product(basis_vector_qi, vectors_V[next_col_index])
            for row_index in range(rows_count):
                vectors_V[next_col_index][row_index] -= matrix_R[col_index][next_col_index] * basis_vector_qi[row_index]
                
    return matrix_Q, matrix_R

# ==========================================
# TEST THUẬT TOÁN VÀ ĐÁNH GIÁ SAI SỐ
# ==========================================

if __name__ == "__main__":
    import numpy as np
    
    print("="*80)
    print("KIỂM THỬ TÍNH CHẤT MA TRẬN, SO SÁNH VỚI NUMPY & ĐÁNH GIÁ SAI SỐ")
    print("="*80)

    def verify_with_numpy(Q_custom, R_custom, Q_np, R_np):
        """Đo độ lệch giữa code tự cài và NumPy."""
        Q_c = np.abs(np.array(Q_custom, dtype=float))
        R_c = np.abs(np.array(R_custom, dtype=float))
        Q_n = np.abs(Q_np)
        R_n = np.abs(R_np)
        
        difference_Q = np.linalg.norm(Q_c - Q_n, ord='fro')
        difference_R = np.linalg.norm(R_c - R_n, ord='fro')
        return difference_Q, difference_R

    def check_orthogonality_error(Q_custom):
        """Kiểm tra sai số mất tính trực giao: ||Q^T * Q - I||_F"""
        Q_np = np.array(Q_custom, dtype=float)
        num_cols = Q_np.shape[1]
        identity_np = np.eye(num_cols)
        return np.linalg.norm(Q_np.T @ Q_np - identity_np, ord='fro')

    def test_and_evaluate_qr(test_name: str, matrix_data: List[List[float]]):
        print(f"\n{'-'*80}")
        print(f"{test_name}")
        print(f"{'-'*80}")
        display_matrix("Ma trận A", matrix_data)
        
        # 1. Chạy code tự cài
        try:
            Q_custom, R_custom = perform_qr_decomposition(matrix_data)
        except Exception as error_msg:
            print(f"Thuật toán tự cài gặp lỗi: {error_msg}")
            return
        
        # 2. Chạy bằng NumPy để đối chứng
        matrix_np = np.array(matrix_data, dtype=float)
        Q_np, R_np = np.linalg.qr(matrix_np)
        
        # 3. So sánh và đánh giá
        diff_Q, diff_R = verify_with_numpy(Q_custom, R_custom, Q_np, R_np)
        orthogonality_error = check_orthogonality_error(Q_custom)
        reconstruction_error = np.linalg.norm(matrix_np - np.array(Q_custom) @ np.array(R_custom), ord='fro')
        
        print(f">>> ĐÁNH GIÁ SAI SỐ THUẬT TOÁN TỰ CÀI (MGS):")
        print(f"    - Lệch ma trận Q so với NumPy   : {diff_Q:.5e}")
        print(f"    - Lệch ma trận R so với NumPy   : {diff_R:.5e}")
        print(f"    - Sai số trực giao ||Q^T Q - I|| : {orthogonality_error:.5e}")
        print(f"    - Sai số tái tạo A ≈ QR        : {reconstruction_error:.5e}")
        
        if reconstruction_error < 1e-9:
            print(">>> KẾT LUẬN: Phân rã QR THÀNH CÔNG")
        else:
            print(">>> KẾT LUẬN: Phân rã QR CÓ SAI SỐ LỚN")

    # Demo nhanh một trường hợp
    matrix_example = [[1, 2], [3, 4], [5, 6]]
    test_and_evaluate_qr("Demo 3x2 Matrix", matrix_example)