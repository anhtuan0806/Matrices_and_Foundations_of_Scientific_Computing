import math
from typing import List, Any
from part1.gaussian import perform_gaussian_elimination, EPSILON
from part2.decomposition import perform_qr_decomposition, transpose_matrix, calculate_vector_norm

def calculate_residual_error(matrix_A: List[List[float]], solution_x: List[float], vector_b: List[float]) -> float:
    """
    Tính sai số tương đối của nghiệm: ||Ax - b||_2 / ||b||_2.
    Đây là chỉ số quan trọng để đánh giá độ chính xác của phương pháp giải.
    """
    number_of_rows = len(matrix_A)
    
    # Tính Ax
    matrix_product_Ax = [sum(matrix_A[row_index][col_index] * solution_x[col_index] for col_index in range(len(solution_x))) 
                         for row_index in range(number_of_rows)]
    
    # Tính sai số (Ax - b)
    difference_vector = [matrix_product_Ax[row_index] - vector_b[row_index] for row_index in range(number_of_rows)]
    
    norm_residual = calculate_vector_norm(difference_vector)
    norm_b = calculate_vector_norm(vector_b)
    
    return norm_residual / norm_b if norm_b > EPSILON else norm_residual

def solve_system_via_gauss(matrix_A: List[List[float]], vector_b: List[float]) -> List[float]:
    """
    Giải hệ phương trình Ax = b bằng phương pháp khử Gauss từ Part 1.
    Phương pháp trực tiếp, phù hợp với hầu hết các ma trận khả nghịch.
    """
    _, solution_x, _ = perform_gaussian_elimination(matrix_A, vector_b, verbose=False)
    
    # Kiểm tra nếu solution_x là chuỗi thông báo lỗi (vô nghiệm) hoặc tuple (vô số nghiệm)
    if isinstance(solution_x, (str, tuple)):
        raise ValueError(f"Không thể tìm nghiệm duy nhất bằng Gauss: {solution_x}")
        
    return solution_x

def solve_system_via_qr(matrix_A: List[List[float]], vector_b: List[float]) -> List[float]:
    """
    Giải hệ phương trình Ax = b bằng phân rã QR (A = QR).
    Lý thuyết: Ax = b => QRx = b => Rx = Q^T * b.
    Giải hệ tam giác trên R bằng phép thế ngược.
    """
    matrix_Q, matrix_R = perform_qr_decomposition(matrix_A)
    number_of_vars = len(matrix_A[0])
    
    # Tính y = Q^T * b
    matrix_QT = transpose_matrix(matrix_Q)
    vector_y = [sum(matrix_QT[row_index][k_index] * vector_b[k_index] for k_index in range(len(vector_b))) 
                for row_index in range(number_of_vars)]
    
    # Thế ngược giải hệ Rx = y
    solution_x = [0.0] * number_of_vars
    for row_index in range(number_of_vars - 1, -1, -1):
        back_sum = sum(matrix_R[row_index][col_index] * solution_x[col_index] for col_index in range(row_index + 1, number_of_vars))
        if abs(matrix_R[row_index][row_index]) > EPSILON:
            solution_x[row_index] = (vector_y[row_index] - back_sum) / matrix_R[row_index][row_index]
        else:
            solution_x[row_index] = 0.0
            
    return solution_x

def solve_system_via_gauss_seidel(matrix_A: List[List[float]], vector_b: List[float], 
                                  max_iterations: int = 1000, tolerance: float = 1e-9) -> List[float]:
    """
    Giải hệ phương trình Ax = b bằng phương pháp lặp Gauss-Seidel.
    Điều kiện hội tụ: Ma trận A chéo trội nghiêm ngặt hoặc đối xứng xác định dương.
    """
    number_of_vars = len(matrix_A)
    solution_x = [0.0] * number_of_vars

    # Kiểm tra điều kiện chéo trội để đưa ra cảnh báo hội tụ
    is_diagonally_dominant = True
    for row_index in range(number_of_vars):
        diagonal_value = abs(matrix_A[row_index][row_index])
        off_diagonal_sum = sum(abs(matrix_A[row_index][col_index]) for col_index in range(number_of_vars) if row_index != col_index)
        if diagonal_value <= off_diagonal_sum:
            is_diagonally_dominant = False
            break
            
    if not is_diagonally_dominant:
        # Sử dụng encode/decode để tránh lỗi Unicode trên Windows console
        print("Cảnh báo: Ma trận không chéo trội, Gauss-Seidel có thể không hội tụ.".encode('ascii', 'ignore').decode())

    # Vòng lặp Gauss-Seidel
    for iteration_index in range(max_iterations):
        max_absolute_change = 0.0
        for row_index in range(number_of_vars):
            current_sum = sum(matrix_A[row_index][col_index] * solution_x[col_index] for col_index in range(number_of_vars) if row_index != col_index)
            
            if abs(matrix_A[row_index][row_index]) < EPSILON:
                raise ZeroDivisionError(f"Phần tử đường chéo tại dòng {row_index} quá nhỏ.")
                
            new_xi_val = (vector_b[row_index] - current_sum) / matrix_A[row_index][row_index]
            
            # Cập nhật sai lệch để kiểm tra điều kiện dừng
            max_absolute_change = max(max_absolute_change, abs(new_xi_val - solution_x[row_index]))
            solution_x[row_index] = new_xi_val

        # Nếu sự thay đổi giữa 2 bước lặp nhỏ hơn ngưỡng cho phép, ta coi là đã hội tụ
        if max_absolute_change < tolerance:
            break
            
    return solution_x