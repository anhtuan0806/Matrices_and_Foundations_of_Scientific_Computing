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

def subtract_matrices(matrix_A: List[List[float]], matrix_B: List[List[float]]) -> List[List[float]]:
    """Tính hiệu của hai ma trận cùng kích thước: A - B."""
    rows_A, cols_A = get_matrix_dimensions(matrix_A)
    rows_B, cols_B = get_matrix_dimensions(matrix_B)
    if rows_A != rows_B or cols_A != cols_B:
        raise ValueError("Kích thước ma trận không tương thích để thực hiện phép trừ!")
    
    result = create_zero_matrix(rows_A, cols_A)
    for row_idx in range(rows_A):
        for col_idx in range(cols_A):
            result[row_idx][col_idx] = matrix_A[row_idx][col_idx] - matrix_B[row_idx][col_idx]
    return result

def calculate_frobenius_norm(matrix_input: List[List[float]]) -> float:
    """Tính chuẩn Frobenius của ma trận (căn bậc hai của tổng bình phương các phần tử)."""
    sum_squares = 0.0
    for row in matrix_input:
        for val in row:
            sum_squares += val ** 2
    return math.sqrt(sum_squares)

def invert_matrix(matrix_input: List[List[float]]) -> List[List[float]]:
    """
    Nghịch đảo ma trận vuông bằng phương pháp khử Gauss-Jordan.
    Ném ra ValueError nếu ma trận suy biến.
    """
    rows_count, cols_count = get_matrix_dimensions(matrix_input)
    if rows_count != cols_count:
        raise ValueError("Chỉ ma trận vuông mới có thể nghịch đảo!")
    
    n = rows_count
    # Tạo ma trận bổ sung [A | I]
    augmented_matrix = [row[:] + [1.0 if idx == row_idx else 0.0 for idx in range(n)] for row_idx, row in enumerate(matrix_input)]
    
    for pivot_idx in range(n):
        # Tìm dòng có phần tử pivot lớn nhất để đảm bảo tính ổn định số học
        max_row_idx = pivot_idx
        max_val = abs(augmented_matrix[pivot_idx][pivot_idx])
        for row_idx in range(pivot_idx + 1, n):
            if abs(augmented_matrix[row_idx][pivot_idx]) > max_val:
                max_val = abs(augmented_matrix[row_idx][pivot_idx])
                max_row_idx = row_idx
        
        # Kiểm tra ma trận suy biến
        if max_val < EPSILON:
            raise ValueError("Ma trận suy biến, không thể nghịch đảo!")
            
        # Hoán vị dòng
        augmented_matrix[pivot_idx], augmented_matrix[max_row_idx] = augmented_matrix[max_row_idx], augmented_matrix[pivot_idx]
        
        # Chuẩn hóa dòng pivot
        pivot_val = augmented_matrix[pivot_idx][pivot_idx]
        for col_idx in range(pivot_idx, 2 * n):
            augmented_matrix[pivot_idx][col_idx] /= pivot_val
            
        # Khử các dòng khác
        for row_idx in range(n):
            if row_idx != pivot_idx:
                factor = augmented_matrix[row_idx][pivot_idx]
                for col_idx in range(pivot_idx, 2 * n):
                    augmented_matrix[row_idx][col_idx] -= factor * augmented_matrix[pivot_idx][col_idx]
                    
    # Trích xuất ma trận nghịch đảo từ ma trận bổ sung
    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix

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
    print("="*80)
    print("KIỂM THỬ TÍNH CHẤT MA TRẬN VÀ ĐÁNH GIÁ SAI SỐ (PURE PYTHON)")
    print("="*80)

    def calculate_orthogonality_error(matrix_Q: List[List[float]]) -> float:
        """Kiểm tra sai số trực giao: ||Q^T * Q - I||_F"""
        rows, cols = get_matrix_dimensions(matrix_Q)
        QT = get_matrix_transpose(matrix_Q)
        QTQ = perform_matrix_multiplication(QT, matrix_Q)
        identity = [[1.0 if idx == row_idx else 0.0 for idx in range(cols)] for row_idx in range(cols)]
        diff = subtract_matrices(QTQ, identity)
        return calculate_frobenius_norm(diff)

    def test_and_evaluate_qr(test_name: str, matrix_data: List[List[float]]):
        print(f"\n{'-'*80}")
        print(f"{test_name}")
        print(f"{'-'*80}")
        display_matrix("Ma trận A", matrix_data)
        
        try:
            # 1. Chạy phân rã QR
            Q_custom, R_custom = perform_qr_decomposition(matrix_data)
            
            # 2. Đánh giá sai số
            # Sai số tái tạo: ||A - QR||
            QR = perform_matrix_multiplication(Q_custom, R_custom)
            diff_reconstruction = subtract_matrices(matrix_data, QR)
            reconstruction_error = calculate_frobenius_norm(diff_reconstruction)
            
            # Sai số trực giao: ||Q^T Q - I||
            orthogonality_error = calculate_orthogonality_error(Q_custom)
            
            print(f">>> ĐÁNH GIÁ SAI SỐ (PURE PYTHON):")
            print(f"    - Sai số trực giao ||Q^T Q - I|| : {orthogonality_error:.5e}")
            print(f"    - Sai số tái tạo ||A - QR||      : {reconstruction_error:.5e}")
            
            # --- KIỂM CHỨNG BẰNG NUMPY ---
            import numpy as np
            matrix_np = np.array(matrix_data, dtype=float)
            Q_np, R_np = np.linalg.qr(matrix_np)
            diff_R_norm = np.linalg.norm(np.abs(np.array(R_custom)) - np.abs(R_np))
            print(f">>> KIỂM CHỨNG VỚI NUMPY:")
            print(f"    - Sai lệch độ lớn ma trận R so với NumPy: {diff_R_norm:.5e}")
            
            if reconstruction_error < 1e-9:
                print(">>> KẾT LUẬN: Phân rã QR THÀNH CÔNG")
            else:
                print(">>> KẾT LUẬN: Phân rã QR CÓ SAI SỐ LỚN")
                
        except Exception as error_msg:
            print(f"LỖI trong quá trình kiểm thử: {error_msg}")

    # Demo nhanh một trường hợp
    matrix_example = [[1, 2], [3, 4], [5, 6]]
    test_and_evaluate_qr("Demo 3x2 Matrix", matrix_example)