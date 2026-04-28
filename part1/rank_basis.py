
from gaussian import gaussian_eliminate

def rank_and_basis(A):
    m, n = len(A), len(A[0])
    eps = 1e-12
    M_aug, _, _ = gaussian_eliminate(A, [0.0] * m, verbose=False)
    U = [row[:n] for row in M_aug]
    
    pivot_cols = []
    r = 0
    for j in range(n):
        if r < m and abs(U[r][j]) > eps:
            pivot_cols.append(j)
            r += 1
    rank = len(pivot_cols)
    row_basis = [U[i][:] for i in range(rank)]
    col_basis = [[A[i][j] for i in range(m)] for j in pivot_cols]
    
    free_cols = [j for j in range(n) if j not in pivot_cols]
    null_basis = []
    for fj in free_cols:
        x = [0.0] * n
        x[fj] = 1.0
        for i in range(rank - 1, -1, -1):
            pc = pivot_cols[i]
            val = sum(U[i][j] * x[j] for j in range(pc + 1, n))
            x[pc] = -val / U[i][pc]
        null_basis.append(x)
        
    return rank, col_basis, row_basis, null_basis