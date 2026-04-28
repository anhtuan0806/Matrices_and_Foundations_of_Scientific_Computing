
from typing import List, Tuple
from gaussian import perform_gaussian_elimination, EPSILON

def calculate_rank_and_bases(matrix_A: List[List[float]]) -> Tuple[int, List[List[float]], List[List[float]], List[List[float]]]:
    """
    Tính hạng và tìm cơ sở của các không gian con liên quan đến ma trận A:
    1. Hạng (Rank): Số cột chốt (hoặc số dòng khác 0 trong dạng REF).
    2. Không gian cột (Column Space): Cơ sở là các cột của A gốc ứng với vị trí cột chốt.
    3. Không gian dòng (Row Space): Cơ sở là các dòng khác 0 của ma trận ở dạng REF/RREF.
    4. Không gian nghiệm (Null Space): Tập các vector x thỏa mãn Ax = 0.
    """
    rows_count, cols_count = len(matrix_A), len(matrix_A[0])
    
    # Thực hiện khử Gauss để đưa ma trận về dạng bậc thang (REF)
    # Dùng dummy vector b vì thuật toán yêu cầu ma trận tăng cường
    augmented_matrix, _, _ = perform_gaussian_elimination(matrix_A, [0.0] * rows_count, verbose=False)
    upper_matrix_U = [row[:cols_count] for row in augmented_matrix]
    
    # Xác định các cột chốt (pivot columns)
    pivot_columns = []
    current_pivot_row = 0
    for j in range(cols_count):
        if current_pivot_row < rows_count and abs(upper_matrix_U[current_pivot_row][j]) > EPSILON:
            pivot_columns.append(j)
            current_pivot_row += 1
            
    # 1. Hạng của ma trận
    rank_value = len(pivot_columns)
    
    # 2. Cơ sở không gian dòng: Các dòng khác 0 trong REF
    row_space_basis = [upper_matrix_U[i][:] for i in range(rank_value)]
    
    # 3. Cơ sở không gian cột: Các cột tương ứng trong ma trận gốc A
    column_space_basis = [[matrix_A[i][j] for i in range(rows_count)] for j in pivot_columns]
    
    # 4. Cơ sở không gian nghiệm: Giải hệ thuần nhất Ax = 0
    # Các ẩn không phải ẩn chốt được coi là ẩn tự do
    free_columns = [j for j in range(cols_count) if j not in pivot_columns]
    null_space_basis = []
    
    for free_idx in free_columns:
        # Tạo vector nghiệm cơ bản bằng cách đặt ẩn tự do hiện tại = 1, các ẩn tự do khác = 0
        solution_vector = [0.0] * cols_count
        solution_vector[free_idx] = 1.0
        
        # Giải ngược từ dưới lên để tìm giá trị các ẩn chốt theo ẩn tự do này
        for i in range(rank_value - 1, -1, -1):
            p_col = pivot_columns[i]
            # Tính tổng các thành phần đã biết của dòng i
            known_sum = sum(upper_matrix_U[i][j] * solution_vector[j] for j in range(p_col + 1, cols_count))
            solution_vector[p_col] = -known_sum / upper_matrix_U[i][p_col]
            
        null_space_basis.append(solution_vector)
        
    return rank_value, column_space_basis, row_space_basis, null_space_basis