
from gaussian import gaussian_eliminate

def determinant(A):
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Định thức chỉ xác định cho ma trận vuông.")
    b_dummy = [0.0] * n
    try:
        M_aug, _, s = gaussian_eliminate(A, b_dummy, verbose=False)
        det = (-1.0) ** s
        for i in range(n):
            det *= M_aug[i][i]
        return 0.0 if abs(det) < 1e-12 else det
    except:
        return 0.0