from decomposition import (
    perform_qr_decomposition, perform_matrix_multiplication, create_zero_matrix, 
    get_matrix_dimensions, display_matrix, EPSILON, subtract_matrices, 
    calculate_frobenius_norm, invert_matrix
)
from typing import List, Tuple, Any
import random

def create_identity_matrix(size_dim: int) -> List[List[float]]:
    """Tạo ma trận đơn vị kích thước size x size."""
    matrix_identity = create_zero_matrix(size_dim, size_dim)
    for diag_index in range(size_dim):
        matrix_identity[diag_index][diag_index] = 1.0
    return matrix_identity

def perform_matrix_diagonalization(matrix_A: List[List[float]], max_iterations: int = 1000, tolerance: float = 1e-9) -> Tuple[List[List[Any]], List[List[Any]]]:
    """
    Thực hiện chéo hóa ma trận bằng thuật toán QR Iteration.
    Lý thuyết: Phép biến đổi A_{k+1} = R_k * Q_k là một phép biến đổi đồng dạng,
    bảo toàn các trị riêng. Khi k tiến tới vô cùng, A_k sẽ hội tụ về dạng tam giác (Schur form),
    với các trị riêng nằm trên đường chéo chính.
    """
    rows_count, cols_count = get_matrix_dimensions(matrix_A)
    if rows_count != cols_count:
        raise ValueError("Chỉ ma trận vuông mới có thể thực hiện chéo hóa.")
    
    # Khởi tạo ma trận A_k và ma trận chuyển cơ sở P
    matrix_Ak = [row[:] for row in matrix_A] 
    matrix_P = create_identity_matrix(cols_count)
    is_converged = False
    
    for iteration_index in range(max_iterations):
        try:
            # Phân rã QR ma trận hiện tại
            matrix_Q, matrix_R = perform_qr_decomposition(matrix_Ak)
        except Exception as error_msg:
            print(f"Phân rã QR thất bại tại vòng lặp {iteration_index}: {error_msg}")
            break
            
        # Cập nhật A_{k+1} = R_k * Q_k
        matrix_Ak = perform_matrix_multiplication(matrix_R, matrix_Q)
        
        # Cập nhật ma trận chuyển cơ sở P = P * Q
        matrix_P = perform_matrix_multiplication(matrix_P, matrix_Q)
        
        # Kiểm tra điều kiện hội tụ: tổng các phần tử dưới đường chéo chính xấp xỉ 0
        off_diagonal_magnitude = 0.0
        for row_index in range(cols_count):
            for col_index in range(row_index):
                off_diagonal_magnitude += abs(matrix_Ak[row_index][col_index])
        
        if off_diagonal_magnitude < tolerance:
            print(f"Thuật toán QR hội tụ thành công sau {iteration_index + 1} vòng lặp.\n".encode('ascii', 'ignore').decode())
            is_converged = True
            break
            
    # Trích xuất ma trận đường chéo D từ đường chéo chính của Ak
    matrix_D = create_zero_matrix(cols_count, cols_count)
    for diag_index in range(cols_count):
        matrix_D[diag_index][diag_index] = matrix_Ak[diag_index][diag_index]
        
    if not is_converged:
        print(f"Cảnh báo: Thuật toán không đạt ngưỡng hội tụ sau {max_iterations} vòng lặp.".encode('ascii', 'ignore').decode())
        print("Kết quả có thể là ma trận Schur (tam giác trên) thay vì ma trận chéo.")
        
    return matrix_D, matrix_P

def run_diagonalization_test(case_name: str, matrix_data: List[List[float]]) -> None:
    """Kiểm thử và so sánh kết quả chéo hóa với ma trận gốc (Pure Python)."""
    print("=" * 80)
    print(f" TEST CASE: {case_name}")
    print("=" * 80)
    
    try:
        matrix_D, matrix_P = perform_matrix_diagonalization(matrix_data)
        
        display_matrix("Ma trận đường chéo D (Custom)", matrix_D)
        
        # Kiểm chứng tính đúng đắn bằng cách tái cấu trúc A = P * D * P^-1
        try:
            # 1. Tính P^-1
            matrix_P_inv = invert_matrix(matrix_P)
            
            # 2. Tính P * D * P^-1
            PD = perform_matrix_multiplication(matrix_P, matrix_D)
            matrix_A_reconstructed = perform_matrix_multiplication(PD, matrix_P_inv)
            
            # 3. Tính sai số tái cấu trúc: ||A - PDP^-1||
            diff = subtract_matrices(matrix_data, matrix_A_reconstructed)
            reconstruction_error = calculate_frobenius_norm(diff)
            
            print(f"=> Sai số tái cấu trúc ||A - PDP^-1||: {reconstruction_error:.2e}")
            
            if reconstruction_error < 1e-7:
                print("=> ĐÁNH GIÁ: CHÉO HÓA THÀNH CÔNG")
            else:
                print("=> ĐÁNH GIÁ: CHÉO HÓA CÓ SAI SỐ (Ma trận có thể không chéo hóa được)")
                
        except ValueError as error_msg:
            # Thông thường là do ma trận P không khả nghịch (suy biến)
            print(f"=> Cảnh báo: Không thể kiểm chứng bằng tái cấu trúc: {error_msg}")
            
    except Exception as error_msg:
        print(f"-> LỖI: {error_msg}")
    print("\n")

if __name__ == "__main__":
    # Demo ma trận đối xứng đơn giản
    symmetric_matrix = [[4, 1], [1, 3]]
    run_diagonalization_test("Simple Symmetric Matrix 2x2", symmetric_matrix)
    
    # Demo ma trận đã chéo hóa sẵn
    diagonal_matrix = [[2, 0], [0, 5]]
    run_diagonalization_test("Already Diagonal Matrix 2x2", diagonal_matrix)