from decomposition import perform_qr_decomposition, multiply_matrices, create_zero_matrix, get_matrix_dimensions, display_matrix, EPSILON
from typing import List, Tuple, Any
import numpy as np
import random

def create_identity_matrix(size: int) -> List[List[float]]:
    """Tạo ma trận đơn vị kích thước size x size."""
    identity = create_zero_matrix(size, size)
    for i in range(size):
        identity[i][i] = 1.0
    return identity

def perform_matrix_diagonalization(matrix_A: List[List[float]], max_iterations: int = 1000, tolerance: float = 1e-9) -> Tuple[List[List[Any]], List[List[Any]]]:
    """
    Thực hiện chéo hóa ma trận bằng thuật toán QR Iteration.
    Lý thuyết: Phép biến đổi A_{k+1} = R_k * Q_k là một phép biến đổi đồng dạng,
    bảo toàn các trị riêng. Khi k tiến tới vô cùng, A_k sẽ hội tụ về dạng tam giác (Schur form),
    với các trị riêng nằm trên đường chéo chính.
    """
    rows, cols = get_matrix_dimensions(matrix_A)
    if rows != cols:
        raise ValueError("Chỉ ma trận vuông mới có thể thực hiện chéo hóa.")
    
    # Khởi tạo ma trận A_k và ma trận chuyển cơ sở P
    matrix_Ak = [row[:] for row in matrix_A] 
    matrix_P = create_identity_matrix(cols)
    is_converged = False
    
    for iteration in range(max_iterations):
        try:
            # Phân rã QR ma trận hiện tại
            matrix_Q, matrix_R = perform_qr_decomposition(matrix_Ak)
        except Exception as error:
            print(f"Phân rã QR thất bại tại vòng lặp {iteration}: {error}")
            break
            
        # Cập nhật A_{k+1} = R_k * Q_k
        matrix_Ak = multiply_matrices(matrix_R, matrix_Q)
        
        # Cập nhật ma trận chuyển cơ sở P = P * Q
        matrix_P = multiply_matrices(matrix_P, matrix_Q)
        
        # Kiểm tra điều kiện hội tụ: tổng các phần tử dưới đường chéo chính xấp xỉ 0
        off_diagonal_magnitude = sum(abs(matrix_Ak[i][j]) for i in range(cols) for j in range(i))
        
        if off_diagonal_magnitude < tolerance:
            print(f"Thuật toán QR hội tụ thành công sau {iteration + 1} vòng lặp.\n".encode('ascii', 'ignore').decode())
            is_converged = True
            break
            
    # Trích xuất ma trận đường chéo D từ đường chéo chính của Ak
    matrix_D = create_zero_matrix(cols, cols)
    for i in range(cols):
        matrix_D[i][i] = matrix_Ak[i][i]
        
    if not is_converged:
        print(f"Cảnh báo: Thuật toán không đạt ngưỡng hội tụ sau {max_iterations} vòng lặp.".encode('ascii', 'ignore').decode())
        print("Kết quả có thể là ma trận Schur (tam giác trên) thay vì ma trận chéo.")
        
    return matrix_D, matrix_P

def run_diagonalization_test(case_name: str, matrix_data: List[List[float]]) -> None:
    """Kiểm thử và so sánh kết quả chéo hóa với ma trận gốc."""
    print("=" * 80)
    print(f" TEST CASE: {case_name}")
    print("=" * 80)
    
    try:
        matrix_D, matrix_P = perform_matrix_diagonalization(matrix_data)
        size = len(matrix_data)
        
        display_matrix("Ma trận đường chéo D (Custom)", matrix_D)
        
        # Kiểm chứng tính đúng đắn bằng cách tái cấu trúc A = P * D * P^-1
        # Sử dụng NumPy chỉ cho mục đích kiểm tra (verification)
        matrix_P_np = np.array(matrix_P, dtype=float)
        matrix_D_np = np.array(matrix_D, dtype=float)
        
        try:
            matrix_P_inv_np = np.linalg.inv(matrix_P_np)
            matrix_A_reconstructed = matrix_P_np @ matrix_D_np @ matrix_P_inv_np
            
            reconstruction_error = np.linalg.norm(np.array(matrix_data) - matrix_A_reconstructed)
            print(f"=> Sai số tái cấu trúc ||A - PDP^-1||: {reconstruction_error:.2e}")
            
            if reconstruction_error < 1e-7:
                print("=> ĐÁNH GIÁ: CHÉO HÓA THÀNH CÔNG")
            else:
                print("=> ĐÁNH GIÁ: CHÉO HÓA CÓ SAI SỐ (Ma trận có thể không chéo hóa được)")
        except np.linalg.LinAlgError:
            print("=> Cảnh báo: Ma trận P không khả nghịch, không thể kiểm chứng bằng tái cấu trúc.")
            
    except Exception as error:
        print(f"-> LỖI: {error}")
    print("\n")

if __name__ == "__main__":
    # Demo ma trận đối xứng đơn giản
    symmetric_matrix = [[4, 1], [1, 3]]
    run_diagonalization_test("Simple Symmetric Matrix 2x2", symmetric_matrix)
    
    # Demo ma trận đã chéo hóa sẵn
    diagonal_matrix = [[2, 0], [0, 5]]
    run_diagonalization_test("Already Diagonal Matrix 2x2", diagonal_matrix)