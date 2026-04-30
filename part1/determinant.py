
from typing import List
from gaussian import gaussian_eliminate, EPSILON

def determinant(A: List[List[float]]) -> float:
    matrix_A = A
    """
    Tính định thức của ma trận vuông thông qua phép khử Gauss.
    Lý thuyết: det(A) = (-1)^s * tích các phần tử trên đường chéo của ma trận tam giác trên U,
    trong đó s là số lần hoán đổi dòng.
    """
    number_of_rows = len(matrix_A)
    
    # Kiểm tra điều kiện ma trận vuông để đảm bảo tính hợp lệ của định thức
    if any(len(row) != number_of_rows for row in matrix_A):
        raise ValueError("Định thức chỉ xác định cho ma trận vuông.")
        
    # Tạo vector b giả (toàn 0) vì perform_gaussian_elimination yêu cầu hệ Ax = b
    dummy_vector_b = [0.0] * number_of_rows
    
    try:
        # Thực hiện khử Gauss để đưa về dạng tam giác trên
        # verbose=False để tránh in nghiệm của hệ giả
        augmented_matrix, _, swap_count = gaussian_eliminate(matrix_A, dummy_vector_b, verbose=False)
        
        # Dấu của định thức phụ thuộc vào số lần hoán đổi dòng
        determinant_value = (-1.0) ** swap_count
        
        # Định thức bằng tích các phần tử trên đường chéo chính của ma trận U
        for row_index in range(number_of_rows):
            determinant_value *= augmented_matrix[row_index][row_index]
            
        # Làm sạch kết quả nếu định thức cực nhỏ (do sai số làm tròn)
        return 0.0 if abs(determinant_value) < EPSILON else determinant_value
        
    except ZeroDivisionError:
        # Nếu trong quá trình khử gặp cột toàn số 0 tại vị trí chốt, định thức bằng 0
        return 0.0
    except Exception:
        # Xử lý các lỗi tính toán phát sinh khác
        return 0.0
