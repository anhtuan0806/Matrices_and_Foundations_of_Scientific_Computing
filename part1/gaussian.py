
from typing import List, Tuple, Union, Any

# Hằng số quy định ngưỡng sai số làm tròn cho tính toán số thực
# Giúp xác định các giá trị xấp xỉ 0 để tránh lỗi chia cho 0 và tăng tính ổn định
EPSILON = 1e-12

def format_term_expression(coefficient_vector: List[float], free_variable_names: List[str]) -> str:
    """
    Biểu diễn một thành phần của nghiệm (x_i) dưới dạng chuỗi toán học kết hợp các ẩn tự do.
    Việc sử dụng format này giúp người dùng dễ dàng theo dõi nghiệm tổng quát của hệ.
    """
    parts = []
    constant_value = round(coefficient_vector[0], 10)
    
    # Chỉ thêm hằng số vào biểu thức nếu nó đáng kể
    if abs(constant_value) > EPSILON:
        parts.append(f"{constant_value:g}")
        
    for index, name in enumerate(free_variable_names):
        coefficient = round(coefficient_vector[1 + index], 10)
        
        # Bỏ qua các ẩn tự do có hệ số bằng 0 để làm gọn biểu thức
        if abs(coefficient) < EPSILON:
            continue
            
        # Xử lý các trường hợp đặc biệt của hệ số (1, -1) để biểu thức tự nhiên hơn
        if abs(abs(coefficient) - 1.0) < 1e-9:
            term = name if coefficient > 0 else f"-{name}"
        else:
            term = f"{coefficient:g}*{name}"
        parts.append(term)
        
    if not parts:
        return "0"
        
    expression = parts[0]
    for part in parts[1:]:
        # Kết hợp các thành phần bằng dấu + hoặc - phù hợp
        expression += f" - {part[1:]}" if part.startswith('-') else f" + {part}"
    return expression

def display_general_solution(solution_coefficients: List[List[float]], free_column_indices: List[int]) -> None:
    """
    Hiển thị nghiệm tổng quát dưới dạng đại số và dạng vector.
    Hàm này phục vụ mục đích trình bày kết quả cho các hệ vô số nghiệm.
    """
    number_of_variables = len(solution_coefficients)
    free_variable_names = [f"t{index+1}" for index in range(len(free_column_indices))]
    
    # Hàm lambda để làm sạch các giá trị quá nhỏ, tránh gây nhiễu khi hiển thị
    clean_value = lambda value: 0.0 if abs(value) < EPSILON else round(value, 10)

    print("\n  ── Nghiệm tổng quát ──")
    for col_index in range(number_of_variables):
        print(f"  x{col_index+1} = {format_term_expression(solution_coefficients[col_index], free_variable_names)}")

    print("\n  ── Dạng vector ──")
    particular_solution = [f"{clean_value(solution_coefficients[col_index][0]):g}" for col_index in range(number_of_variables)]
    print(f"  x = ({', '.join(particular_solution)})")
    
    for free_index, name in enumerate(free_variable_names):
        basis_vector = [f"{clean_value(solution_coefficients[col_index][1+free_index]):g}" for col_index in range(number_of_variables)]
        print(f"      + {name} * ({', '.join(basis_vector)})")

def solve_back_substitution(upper_triangular_matrix: List[List[float]], constant_vector: List[float]) -> Union[List[float], Tuple[List[List[float]], List[int]], str]:
    """
    Giải hệ phương trình tam giác trên bằng phương pháp thế ngược.
    Hỗ trợ cả trường hợp nghiệm duy nhất, vô số nghiệm và vô nghiệm.
    """
    rows_count, cols_count = len(upper_triangular_matrix), len(upper_triangular_matrix[0])
    pivot_rows, pivot_cols = [], []
    current_row = 0
    
    # Xác định các vị trí chốt (pivot) để phân loại ẩn chính và ẩn tự do
    for col_index in range(cols_count):
        if current_row < rows_count and abs(upper_triangular_matrix[current_row][col_index]) > EPSILON:
            pivot_rows.append(current_row)
            pivot_cols.append(col_index)
            current_row += 1
            
    # Kiểm tra tính tương thích của hệ: nếu dòng 0 có hệ số tự do khác 0 thì vô nghiệm
    for row_index in range(current_row, rows_count):
        if abs(constant_vector[row_index]) > EPSILON:
            return "Hệ phương trình vô nghiệm."
            
    free_cols = [col_index for col_index in range(cols_count) if col_index not in pivot_cols]
    num_free_vars = len(free_cols)
    
    # Khởi tạo ma trận hệ số cho nghiệm tổng quát (kết hợp hằng số và các ẩn tự do)
    # Cấu trúc: [hằng số, hệ số_t1, hệ số_t2, ...]
    solution_coefficients = [[0.0] * (1 + num_free_vars) for _ in range(cols_count)]
    
    # Gán giá trị mặc định cho các ẩn tự do (t_i tương ứng với chính nó)
    for index, free_idx in enumerate(free_cols):
        solution_coefficients[free_idx][1 + index] = 1.0
        
    # Thực hiện thế ngược từ dưới lên để tìm biểu thức của các ẩn chính
    for pivot_index in range(len(pivot_cols) - 1, -1, -1):
        p_col, p_row = pivot_cols[pivot_index], pivot_rows[pivot_index]
        temp_coefficients = [0.0] * (1 + num_free_vars)
        temp_coefficients[0] = constant_vector[p_row]
        
        for col_index in range(p_col + 1, cols_count):
            coefficient_val = upper_triangular_matrix[p_row][col_index]
            if abs(coefficient_val) > EPSILON:
                for coeff_index in range(1 + num_free_vars):
                    temp_coefficients[coeff_index] -= coefficient_val * solution_coefficients[col_index][coeff_index]
                    
        pivot_value = upper_triangular_matrix[p_row][p_col]
        solution_coefficients[p_col] = [val / pivot_value for val in temp_coefficients]
        
    # Trả về kết quả phù hợp với số lượng nghiệm
    if num_free_vars == 0:
        return [solution_coefficients[col_index][0] for col_index in range(cols_count)]
    return (solution_coefficients, free_cols)

def perform_gaussian_elimination(matrix_A: List[List[float]], vector_b: List[float], verbose: bool = True) -> Tuple[List[List[float]], Any, int]:
    """
    Thực hiện phép khử Gauss với kỹ thuật chọn phần tử chốt một phần (Partial Pivoting).
    Kỹ thuật này cực kỳ quan trọng để đảm bảo tính ổn định số học, tránh chia cho các số quá nhỏ
    làm khuếch đại sai số làm tròn trong tính toán dấu phẩy động.
    """
    rows_count, cols_count = len(matrix_A), len(matrix_A[0])
    swap_count = 0
    
    # Tạo ma trận tăng cường [A | b] để thực hiện biến đổi dòng đồng thời
    augmented_matrix = [row[:] + [vector_b[i]] for i, row in enumerate(matrix_A)]
    current_pivot_row = 0
    
    for col_index in range(cols_count):
        if current_pivot_row >= rows_count:
            break
            
        # Tìm phần tử có giá trị tuyệt đối lớn nhất trên cột hiện tại để làm chốt
        # Điều này giúp giảm thiểu sai số tuyệt đối khi thực hiện phép chia
        max_absolute_val = abs(augmented_matrix[current_pivot_row][col_index])
        swap_target_row = current_pivot_row
        
        for row_index in range(current_pivot_row + 1, rows_count):
            if abs(augmented_matrix[row_index][col_index]) > max_absolute_val:
                max_absolute_val = abs(augmented_matrix[row_index][col_index])
                swap_target_row = row_index
                
        # Nếu cột toàn số 0 (hoặc cực nhỏ), bỏ qua và chuyển sang cột tiếp theo
        if max_absolute_val < EPSILON:
            continue
            
        # Hoán đổi dòng hiện tại với dòng chứa phần tử chốt tốt nhất
        if swap_target_row != current_pivot_row:
            augmented_matrix[current_pivot_row], augmented_matrix[swap_target_row] = \
                augmented_matrix[swap_target_row], augmented_matrix[current_pivot_row]
            swap_count += 1
            
        # Triệt tiêu các phần tử bên dưới chốt
        for row_index in range(current_pivot_row + 1, rows_count):
            factor = augmented_matrix[row_index][col_index] / augmented_matrix[current_pivot_row][col_index]
            augmented_matrix[row_index][col_index] = 0.0  # Đặt cứng về 0 để tránh nhiễu số thực
            for target_col_index in range(col_index + 1, cols_count + 1):
                augmented_matrix[row_index][target_col_index] -= factor * augmented_matrix[current_pivot_row][target_col_index]
        
        current_pivot_row += 1
        
    # Tách ma trận tam giác trên U và vector c sau khi khử
    upper_matrix_U = [row[:-1] for row in augmented_matrix]
    constant_vector_c = [row[-1] for row in augmented_matrix]
    
    # Giải hệ bằng thế ngược
    solution_x = solve_back_substitution(upper_matrix_U, constant_vector_c)
    
    # Hỗ trợ hiển thị kết quả nếu hệ có vô số nghiệm
    if verbose and isinstance(solution_x, tuple):
        display_general_solution(solution_x[0], solution_x[1])
        
    return augmented_matrix, solution_x, swap_count