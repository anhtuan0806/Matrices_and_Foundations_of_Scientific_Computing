

def inverse(A):
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Chỉ ma trận vuông mới có khả năng nghịch đảo.")
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    M = [A[i][:] + I[i][:] for i in range(n)]
    for k in range(n):
        max_idx = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[max_idx][k]):
                max_idx = i
        if abs(M[max_idx][k]) < 1e-12:
            return f"Thông báo: Không có pivot tại cột {k+1}. Ma trận không khả nghịch."
        M[k], M[max_idx] = M[max_idx], M[k]
        pivot = M[k][k]
        M[k] = [x / pivot for x in M[k]]
        for i in range(n):
            if i != k:
                factor = M[i][k]
                M[i][k] = 0.0
                for j in range(k + 1, 2 * n):
                    M[i][j] -= factor * M[k][j]
    return [row[n:] for row in M]