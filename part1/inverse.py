

from typing import List
from gaussian import EPSILON

def inverse(A: List[List[float]]) -> List[List[float]]:
    matrix_A = A
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
    identity_matrix = [[1.0 if row_idx == col_idx else 0.0 for col_idx in range(number_of_rows)] 
                       for row_idx in range(number_of_rows)]
    
    # Tạo ma trận tăng cường [A | I]
    augmented_matrix = [matrix_A[row_index][:] + identity_matrix[row_index][:] for row_index in range(number_of_rows)]
    
    for pivot_index in range(number_of_rows):
        # Chọn phần tử chốt lớn nhất trên cột k (Partial Pivoting) để tăng tính ổn định
        max_absolute_row_idx = pivot_index
        for row_index in range(pivot_index + 1, number_of_rows):
            if abs(augmented_matrix[row_index][pivot_index]) > abs(augmented_matrix[max_absolute_row_idx][pivot_index]):
                max_absolute_row_idx = row_index
                
        # Nếu phần tử chốt lớn nhất xấp xỉ 0, ma trận không khả nghịch (suy biến)
        if abs(augmented_matrix[max_absolute_row_idx][pivot_index]) < EPSILON:
            raise ValueError(f"Ma trận không khả nghịch (không tìm thấy pivot tại cột {pivot_index+1}).")
            
        # Hoán đổi dòng hiện tại với dòng chứa phần tử chốt
        augmented_matrix[pivot_index], augmented_matrix[max_absolute_row_idx] = \
            augmented_matrix[max_absolute_row_idx], augmented_matrix[pivot_index]
            
        # Chuẩn hóa dòng k để phần tử chốt (pivot) bằng 1
        pivot_value = augmented_matrix[pivot_index][pivot_index]
        augmented_matrix[pivot_index] = [element / pivot_value for element in augmented_matrix[pivot_index]]
        
        # Triệt tiêu tất cả các phần tử khác trên cột k (cả trên và dưới pivot)
        # Đây là sự khác biệt giữa Gauss-Jordan (tạo RREF) và Gauss thông thường (tạo REF)
        for row_index in range(number_of_rows):
            if row_index != pivot_index:
                factor = augmented_matrix[row_index][pivot_index]
                augmented_matrix[row_index][pivot_index] = 0.0  # Triệt tiêu phần tử
                for col_index in range(pivot_index + 1, 2 * number_of_rows):
                    augmented_matrix[row_index][col_index] -= factor * augmented_matrix[pivot_index][col_index]
                    
    # Trích xuất phần bên phải của ma trận tăng cường chính là A^-1
    inverse_matrix = [row[number_of_rows:] for row in augmented_matrix]
    return inverse_matrix