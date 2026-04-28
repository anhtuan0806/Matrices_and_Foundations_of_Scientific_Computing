import math
import random
import time
import matplotlib.pyplot as plt
from solvers import solve_gauss_pure,solve_qr_pure,solve_gauss_seidel_pure,matrix_vector_mult,calc_relative_error


sizes = [50, 100, 200, 500, 1000]
methods = {
    "Gauss": solve_gauss_pure,
    "QR": solve_qr_pure,
    "Gauss-Seidel": solve_gauss_seidel_pure
}

results_time = {name: [] for name in methods}
results_error = {name: [] for name in methods}

print(f"{'n':<6} | {'Phương pháp':<15} | {'Thời gian (s)':<15} | {'Sai số tương đối'}")
print("-" * 65)

for n in sizes:
    # 1. Tạo dữ liệu (Pure Python)
    # Tạo ma trận ngẫu nhiên thỏa mãn đường chéo trội để Gauss-Seidel hội tụ
    A = [[random.uniform(0, 1) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if i != j)
        A[i][i] = row_sum + random.uniform(1, 2) # Cộng thêm vào đường chéo

    x_true = [random.uniform(0, 1) for _ in range(n)]
    b = matrix_vector_mult(A, x_true)

    # 2. Đo thời gian 3 phương pháp
    for name, func in methods.items():
        times = []
        # Chạy 3-5 lần lấy trung bình (Với n=1000 chỉ chạy 1 lần để tiết kiệm thời gian)
        runs = 5 if n < 500 else 1
        for _ in range(runs):
            start = time.perf_counter()
            x_pred = func(A, b)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        error = calc_relative_error(A, x_pred, b)

        results_time[name].append(avg_time)
        results_error[name].append(error)

        print(f"{n:<6} | {name:<15} | {avg_time:<15.6f} | {error:.6e}")
    print("-" * 65)

# ================= VẼ ĐỒ THỊ LOG-LOG =================
plt.figure(figsize=(10, 6))

for name in methods:
    plt.plot(sizes, results_time[name], marker='o', label=name)

# Vẽ đường lý thuyết O(n^3) để so sánh
c = results_time["Gauss"][0] / (sizes[0]**3)
theoretical_time = [c * (n**3) for n in sizes]
plt.plot(sizes, theoretical_time, 'k--', label='Lý thuyết $O(n^3)$')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Kích thước ma trận $n$ (Log scale)', fontsize=12)
plt.ylabel('Thời gian thực thi trung bình (s) (Log scale)', fontsize=12)
plt.title('Đồ thị Log-Log: Thời gian thực thi vs Kích thước (Python thuần)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")
plt.show()