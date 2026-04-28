

from typing import List
from gaussian import EPSILON

def calculate_matrix_inverse(matrix_A: List[List[float]]) -> List[List[float]]:
    """
    Tìm ma trận nghịch đảo bằng phương pháp Gauss-Jordan.
    Lý thuyết: Thực hiện biến đổi dòng sơ cấp đồng thời trên ma trận ghép [A | I]
    cho đến khi ma trận bên trái trở thành ma trận đơn vị [I | A^-1].
    """
    number_of_rows = len(matrix_A)
    
    # Kiểm tra tính hợp lệ: Chỉ ma trận vuông mới có khả năng nghịch đảo
    if any(len(row) != number_of_rows for row in matrix_A):
        raise ValueError("Chỉ ma trận vuông mới có khả năng nghịch đảo.")
        
    # Tạo ma trận đơn vị I cùng kích thước với A
    identity_matrix = [[1.0 if i == j else 0.0 for j in range(number_of_rows)] 
                       for i in range(number_of_rows)]
    
    # Tạo ma trận tăng cường [A | I]
    augmented_matrix = [matrix_A[i][:] + identity_matrix[i][:] for i in range(number_of_rows)]
    
    for k in range(number_of_rows):
        # Chọn phần tử chốt lớn nhất trên cột k (Partial Pivoting) để tăng tính ổn định
        max_absolute_row_idx = k
        for i in range(k + 1, number_of_rows):
            if abs(augmented_matrix[i][k]) > abs(augmented_matrix[max_absolute_row_idx][k]):
                max_absolute_row_idx = i
                
        # Nếu phần tử chốt lớn nhất xấp xỉ 0, ma trận không khả nghịch (suy biến)
        if abs(augmented_matrix[max_absolute_row_idx][k]) < EPSILON:
            raise ValueError(f"Ma trận không khả nghịch (không tìm thấy pivot tại cột {k+1}).")
            
        # Hoán đổi dòng hiện tại với dòng chứa phần tử chốt
        augmented_matrix[k], augmented_matrix[max_absolute_row_idx] = \
            augmented_matrix[max_absolute_row_idx], augmented_matrix[k]
            
        # Chuẩn hóa dòng k để phần tử chốt (pivot) bằng 1
        pivot_value = augmented_matrix[k][k]
        augmented_matrix[k] = [element / pivot_value for element in augmented_matrix[k]]
        
        # Triệt tiêu tất cả các phần tử khác trên cột k (cả trên và dưới pivot)
        # Đây là sự khác biệt giữa Gauss-Jordan (tạo RREF) và Gauss thông thường (tạo REF)
        for i in range(number_of_rows):
            if i != k:
                factor = augmented_matrix[i][k]
                augmented_matrix[i][k] = 0.0  # Triệt tiêu phần tử
                for j in range(k + 1, 2 * number_of_rows):
                    augmented_matrix[i][j] -= factor * augmented_matrix[k][j]
                    
    # Trích xuất phần bên phải của ma trận tăng cường chính là A^-1
    inverse_matrix = [row[number_of_rows:] for row in augmented_matrix]
    return inverse_matrix