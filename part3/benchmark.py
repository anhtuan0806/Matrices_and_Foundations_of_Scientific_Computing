import time
import random
import matplotlib.pyplot as plt
from typing import List, Dict
from solvers import (
    solve_system_via_gauss, 
    solve_system_via_qr, 
    solve_system_via_gauss_seidel, 
    calculate_residual_error
)

def generate_diagonally_dominant_system(size: int):
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

def perform_benchmark_analysis():
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
            run_counts = 3 if size < 500 else 1
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
    
    # Lưu đồ thị làm artifact (tùy chọn) hoặc hiển thị
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    random.seed(42) # Để kết quả có khả năng tái lập
    perform_benchmark_analysis()