import os
import sys
import time
import random
import math
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from solvers import (
        solve_system_via_gauss,
        solve_system_via_qr,
        solve_system_via_gauss_seidel,
        calculate_residual_error,
    )
except ModuleNotFoundError:
    # When imported as package (e.g., from part3 import benchmark)
    from part3.solvers import (
        solve_system_via_gauss,
        solve_system_via_qr,
        solve_system_via_gauss_seidel,
        calculate_residual_error,
    )

EPSILON = 1e-12

def generate_diagonally_dominant_system(size: int) -> Tuple[List[List[float]], List[float], List[float]]:
    """
    Tạo một hệ phương trình Ax = b ngẫu nhiên với ma trận A chéo trội nghiêm ngặt.
    Điều này đảm bảo phương pháp lặp Gauss-Seidel sẽ hội tụ.
    """
    matrix_A = [[random.uniform(0, 1) for _ in range(size)] for _ in range(size)]
    for row_index in range(size):
        row_sum = sum(abs(matrix_A[row_index][col_index]) for col_index in range(size) if row_index != col_index)
        # Gán phần tử đường chéo lớn hơn tổng các phần tử khác trên cùng dòng
        matrix_A[row_index][row_index] = row_sum + random.uniform(1.0, 5.0)
        
    # Tạo nghiệm thực x_true ngẫu nhiên và tính vector b tương ứng
    x_true = [random.uniform(-10, 10) for _ in range(size)]
    vector_b = [sum(matrix_A[row_index][col_index] * x_true[col_index] for col_index in range(size)) for row_index in range(size)]
    
    return matrix_A, vector_b, x_true

def generate_hilbert_matrix(size: int) -> List[List[float]]:
    return [[1.0 / (row_index + col_index + 1) for col_index in range(size)] for row_index in range(size)]

def transpose_matrix(matrix_A: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*matrix_A)]

def multiply_matrices(matrix_A: List[List[float]], matrix_B: List[List[float]]) -> List[List[float]]:
    rows_count = len(matrix_A)
    cols_count = len(matrix_B[0])
    inner_count = len(matrix_B)
    result = [[0.0] * cols_count for _ in range(rows_count)]
    for row_index in range(rows_count):
        for col_index in range(cols_count):
            result[row_index][col_index] = sum(matrix_A[row_index][k_index] * matrix_B[k_index][col_index] for k_index in range(inner_count))
    return result

def generate_spd_matrix(size: int) -> List[List[float]]:
    random_matrix = [[random.uniform(-1.0, 1.0) for _ in range(size)] for _ in range(size)]
    transpose_random = transpose_matrix(random_matrix)
    product_matrix = multiply_matrices(transpose_random, random_matrix)
    for index in range(size):
        product_matrix[index][index] += size
    return product_matrix

def invert_matrix(matrix_A: List[List[float]]) -> List[List[float]]:
    size = len(matrix_A)
    if any(len(row) != size for row in matrix_A):
        raise ValueError("Chỉ ma trận vuông mới có khả năng nghịch đảo.")

    identity = [[1.0 if row_index == col_index else 0.0 for col_index in range(size)] for row_index in range(size)]
    augmented = [matrix_A[row_index][:] + identity[row_index][:] for row_index in range(size)]

    for pivot_index in range(size):
        max_row_index = pivot_index
        for row_index in range(pivot_index + 1, size):
            if abs(augmented[row_index][pivot_index]) > abs(augmented[max_row_index][pivot_index]):
                max_row_index = row_index

        if abs(augmented[max_row_index][pivot_index]) < EPSILON:
            raise ValueError("Ma trận không khả nghịch (pivot xấp xỉ 0).")

        augmented[pivot_index], augmented[max_row_index] = augmented[max_row_index], augmented[pivot_index]

        pivot_value = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot_value for value in augmented[pivot_index]]

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index][pivot_index] = 0.0
            for col_index in range(pivot_index + 1, 2 * size):
                augmented[row_index][col_index] -= factor * augmented[pivot_index][col_index]

    return [row[size:] for row in augmented]

def calculate_matrix_infinity_norm(matrix_A: List[List[float]]) -> float:
    return max(sum(abs(value) for value in row) for row in matrix_A)

def calculate_condition_number_infinity(matrix_A: List[List[float]]) -> float:
    inverse_matrix = invert_matrix(matrix_A)
    return calculate_matrix_infinity_norm(matrix_A) * calculate_matrix_infinity_norm(inverse_matrix)

def prepare_report_images_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_images_dir = os.path.abspath(os.path.join(current_dir, "..", "report", "images"))
    os.makedirs(report_images_dir, exist_ok=True)
    return report_images_dir

def perform_benchmark_analysis() -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Thực hiện phân tích hiệu năng của 3 phương pháp giải hệ phương trình:
    Gauss, QR và Gauss-Seidel với các kích thước ma trận khác nhau.
    """
    matrix_sizes = [50, 100, 200, 500, 1000]
    solving_methods = {
        "Gauss Elimination": solve_system_via_gauss,
        "QR Decomposition": solve_system_via_qr,
        "Gauss-Seidel": solve_system_via_gauss_seidel
    }
    
    execution_times: Dict[str, List[float]] = {name: [] for name in solving_methods}
    residual_errors: Dict[str, List[float]] = {name: [] for name in solving_methods}
    
    print(f"{'n':<6} | {'Phương pháp':<20} | {'Thời gian (s)':<15} | {'Sai số tương đối'}")
    print("-" * 75)
    
    for size in matrix_sizes:
        # Tạo dữ liệu kiểm thử
        matrix_A, vector_b, _ = generate_diagonally_dominant_system(size)
        
        for name, solver_func in solving_methods.items():
            # Đo thời gian thực thi (chạy nhiều lần để lấy trung bình nếu n nhỏ)
            run_counts = 5 if size < 500 else 1
            start_time = time.perf_counter()
            
            for run_index in range(run_counts):
                solution_x = solver_func(matrix_A, vector_b)
                
            end_time = time.perf_counter()
            average_time = (end_time - start_time) / run_counts
            
            # Tính sai số tương đối ||Ax - b|| / ||b||
            error = calculate_residual_error(matrix_A, solution_x, vector_b)
            
            execution_times[name].append(average_time)
            residual_errors[name].append(error)
            
            print(f"{size:<6} | {name:<20} | {average_time:<15.6f} | {error:.2e}")
        print("-" * 75)
        
    # Vẽ đồ thị Log-Log để đánh giá độ phức tạp thuật toán
    plt.figure(figsize=(12, 7))
    for name in solving_methods:
        plt.plot(matrix_sizes, execution_times[name], marker='o', label=name, linewidth=2)
        
    # Vẽ đường tham chiếu O(n^3) - Độ phức tạp lý thuyết của Gauss/QR
    reference_scale = execution_times["Gauss Elimination"][0] / (matrix_sizes[0]**3)
    theoretical_complexity = [reference_scale * (size_val**3) for size_val in matrix_sizes]
    plt.plot(matrix_sizes, theoretical_complexity, color='black', linestyle='--', label='Lý thuyết $O(n^3)$')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Kích thước ma trận $n$ (Log scale)', fontsize=12)
    plt.ylabel('Thời gian thực thi trung bình (s) (Log scale)', fontsize=12)
    plt.title('Đồ thị Log-Log: Đánh giá hiệu năng giải hệ phương trình', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    report_images_dir = prepare_report_images_dir()
    plot_path = os.path.join(report_images_dir, "log_log_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return execution_times, residual_errors

def perform_condition_analysis() -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    size = 10
    solving_methods = {
        "Gauss Elimination": solve_system_via_gauss,
        "QR Decomposition": solve_system_via_qr,
        "Gauss-Seidel": solve_system_via_gauss_seidel
    }

    matrices = {
        "Hilbert": generate_hilbert_matrix(size),
        "SPD": generate_spd_matrix(size)
    }

    condition_numbers: Dict[str, float] = {}
    residual_table: Dict[str, Dict[str, float]] = {name: {} for name in matrices}

    for matrix_name, matrix_A in matrices.items():
        condition_numbers[matrix_name] = calculate_condition_number_infinity(matrix_A)
        x_true = [random.uniform(-5.0, 5.0) for _ in range(size)]
        vector_b = [sum(matrix_A[row_index][col_index] * x_true[col_index] for col_index in range(size)) for row_index in range(size)]

        for solver_name, solver_func in solving_methods.items():
            solution_x = solver_func(matrix_A, vector_b)
            error = calculate_residual_error(matrix_A, solution_x, vector_b)
            residual_table[matrix_name][solver_name] = error

    return condition_numbers, residual_table

def plot_hilbert_condition_growth() -> List[Tuple[int, float]]:
    sizes = list(range(2, 11))
    condition_values = []
    for size in sizes:
        hilbert_matrix = generate_hilbert_matrix(size)
        condition_values.append(calculate_condition_number_infinity(hilbert_matrix))

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, condition_values, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Kích thước ma trận n", fontsize=12)
    plt.ylabel("Số điều kiện kappa vô cùng (log)", fontsize=12)
    plt.title("Sự tăng trưởng số điều kiện của ma trận Hilbert", fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    report_images_dir = prepare_report_images_dir()
    plot_path = os.path.join(report_images_dir, "condition_number_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return list(zip(sizes, condition_values))

if __name__ == "__main__":
    random.seed(42) # Để kết quả có khả năng tái lập
    perform_benchmark_analysis()

    condition_numbers, residual_table = perform_condition_analysis()
    hilbert_growth = plot_hilbert_condition_growth()

    print("\nKết quả phân tích số điều kiện (chuẩn vô cùng):")
    for name, value in condition_numbers.items():
        print(f"- {name}: {value:.6e}")

    print("\nSai số dư tương đối (n=10):")
    for matrix_name, solver_results in residual_table.items():
        print(f"{matrix_name}:")
        for solver_name, error in solver_results.items():
            print(f"  {solver_name:<20} | {error:.6e}")

    print("\nTăng trưởng số điều kiện Hilbert:")
    for size, cond_val in hilbert_growth:
        print(f"  n={size:<2} | kappa_inf={cond_val:.6e}")
