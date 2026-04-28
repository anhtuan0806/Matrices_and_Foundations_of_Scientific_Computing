import math
from typing import List, Tuple

# Hằng số quy định ngưỡng sai số làm tròn cho tính toán số thực
EPSILON = 1e-12

def get_matrix_dimensions(matrix: List[List[float]]) -> Tuple[int, int]:
    """Trả về số dòng và số cột của ma trận."""
    if not matrix:
        return 0, 0
    return len(matrix), len(matrix[0])

def create_zero_matrix(rows: int, cols: int) -> List[List[float]]:
    """Tạo ma trận toàn số 0 với kích thước rows x cols."""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def extract_matrix_column(matrix: List[List[float]], column_index: int) -> List[float]:
    """Trích xuất một cột của ma trận dưới dạng vector."""
    return [row[column_index] for row in matrix]

def calculate_dot_product(vector_1: List[float], vector_2: List[float]) -> float:
    """Tính tích vô hướng của hai vector."""
    return sum(x * y for x, y in zip(vector_1, vector_2))

def calculate_vector_norm(vector: List[float]) -> float:
    """Tính chuẩn Euclid (L2 norm) của vector."""
    return math.sqrt(calculate_dot_product(vector, vector))

def multiply_matrices(matrix_A: List[List[float]], matrix_B: List[List[float]]) -> List[List[float]]:
    """Thực hiện nhân hai ma trận A (m x n) và B (n x p)."""
    rows_A, cols_A = get_matrix_dimensions(matrix_A)
    rows_B, cols_B = get_matrix_dimensions(matrix_B)
    if cols_A != rows_B:
        raise ValueError("Kích thước ma trận không khớp để thực hiện phép nhân!")
    
    result_matrix = create_zero_matrix(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            result_matrix[i][j] = sum(matrix_A[i][k] * matrix_B[k][j] for k in range(cols_A))
    return result_matrix

def transpose_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Trả về ma trận chuyển vị."""
    rows, cols = get_matrix_dimensions(matrix)
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

def display_matrix(label: str, matrix: List[List[float]]) -> None:
    """In ma trận ra màn hình với định dạng đẹp."""
    print(f"{label}:")
    for row in matrix:
        print("[" + ", ".join(f"{element:8.4f}" for element in row) + "]")
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
    vectors_V = [extract_matrix_column(matrix_A, j) for j in range(cols_count)]
    matrix_Q = create_zero_matrix(rows_count, cols_count)
    matrix_R = create_zero_matrix(cols_count, cols_count)
    
    for i in range(cols_count):
        # Bước 1: Tính chuẩn của vector hiện tại và gán vào đường chéo của R
        matrix_R[i][i] = calculate_vector_norm(vectors_V[i])
        
        if matrix_R[i][i] > EPSILON:
            # Bước 2: Chuẩn hóa vector để tạo cột cho Q
            for k in range(rows_count):
                matrix_Q[k][i] = vectors_V[i][k] / matrix_R[i][i]
        else:
            # Xử lý trường hợp các cột phụ thuộc tuyến tính
            for k in range(rows_count):
                matrix_Q[k][i] = 0.0
                
        # Bước 3: Trực giao hóa các vector còn lại đối với vector q_i vừa tìm được
        # Đây là điểm khác biệt của MGS: cập nhật ngay lập tức các vector chưa xử lý
        q_i = extract_matrix_column(matrix_Q, i)
        for j in range(i + 1, cols_count):
            matrix_R[i][j] = calculate_dot_product(q_i, vectors_V[j])
            for k in range(rows_count):
                vectors_V[j][k] -= matrix_R[i][j] * q_i[k]
                
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
        n = Q_np.shape[1]
        identity_np = np.eye(n)
        return np.linalg.norm(Q_np.T @ Q_np - identity_np, ord='fro')

    def test_and_evaluate_qr(test_name: str, matrix_data: List[List[float]]):
        print(f"\n{'-'*80}")
        print(f"{test_name}")
        print(f"{'-'*80}")
        display_matrix("Ma trận A", matrix_data)
        
        # 1. Chạy code tự cài
        try:
            Q_custom, R_custom = perform_qr_decomposition(matrix_data)
        except Exception as e:
            print(f"Thuật toán tự cài gặp lỗi: {e}")
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
