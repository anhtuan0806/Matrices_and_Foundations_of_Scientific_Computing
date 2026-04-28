import math
import random
import time
import matplotlib.pyplot as plt

# ==========================================
# CÁC HÀM TIỆN ÍCH TOÁN HỌC CƠ BẢN
# ==========================================
def vector_norm(v):
    return math.sqrt(sum(x**2 for x in v))

def matrix_vector_mult(A, x):
    n = len(A)
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]

def calc_relative_error(A, x, b):
    Ax = matrix_vector_mult(A, x)
    diff = [Ax[i] - b[i] for i in range(len(b))]
    norm_diff = vector_norm(diff)
    norm_b = vector_norm(b)
    return norm_diff / norm_b if norm_b != 0 else norm_diff

# ==========================================
# 1. PHƯƠNG PHÁP KHỬ GAUSS (Partial Pivoting)
# ==========================================
def solve_gauss_pure(A_input, b_input):
    n = len(A_input)
    # Deep copy để không làm hỏng ma trận gốc
    A = [row[:] for row in A_input]
    b = b_input[:]

    # Bước thuận: Khử Gauss
    for i in range(n):
        # Tìm phần tử chốt (Partial Pivoting)
        max_row = i
        max_val = abs(A[i][i])
        for k in range(i + 1, n):
            if abs(A[k][i]) > max_val:
                max_val = abs(A[k][i])
                max_row = k

        # Hoán vị dòng
        if i != max_row:
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]

        if abs(A[i][i]) < 1e-12: continue

        # Khử các phần tử dưới chốt
        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i + 1, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]

    # Bước nghịch: Thế ngược
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / A[i][i]
    return x

# ==========================================
# 2. PHƯƠNG PHÁP PHÂN RÃ QR (Gram-Schmidt)
# ==========================================
def solve_qr_pure(A_input, b_input):
    n = len(A_input)
    A = [row[:] for row in A_input]
    Q = [[0.0]*n for _ in range(n)]
    R = [[0.0]*n for _ in range(n)]

    # Phân rã QR (A = QR)
    for j in range(n):
        v = [A[i][j] for i in range(n)]
        for i in range(j):
            qi = [Q[k][i] for k in range(n)]
            R[i][j] = sum(qi[k] * v[k] for k in range(n))
            v = [v[k] - R[i][j]*qi[k] for k in range(n)]
        R[j][j] = vector_norm(v)
        if R[j][j] > 1e-12:
            for k in range(n): Q[k][j] = v[k] / R[j][j]

    # Giải Rx = Q^T b
    # 1. Tính y = Q^T b
    y = [0.0] * n
    for i in range(n):
        y[i] = sum(Q[k][i] * b_input[k] for k in range(n))

    # 2. Thế ngược giải Rx = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(R[i][j] * x[j] for j in range(i + 1, n))
        if abs(R[i][i]) > 1e-12:
            x[i] = (y[i] - s) / R[i][i]
    return x

# ==========================================
# 3. PHƯƠNG PHÁP LẶP GAUSS-SEIDEL
# ==========================================
def solve_gauss_seidel_pure(A, b, max_iters=1000, tol=1e-9):
    n = len(A)
    x = [0.0] * n

    # Kiểm tra điều kiện đường chéo trội (Diagonally Dominant)
    is_dominant = True
    for i in range(n):
        diag_val = abs(A[i][i])
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if i != j)
        if diag_val <= off_diag_sum:
            is_dominant = False
            break
    if not is_dominant:
        print("Cảnh báo: Ma trận không phải đường chéo trội nghiêm ngặt, có thể không hội tụ.")

    # Vòng lặp Gauss-Seidel
    for it in range(max_iters):
        max_diff = 0.0
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new = (b[i] - s) / A[i][i]
            max_diff = max(max_diff, abs(x_new - x[i]))
            x[i] = x_new

        if max_diff < tol:
            break
    return x