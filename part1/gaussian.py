
def _fmt_term(coef_vec, free_names):
    eps = 1e-12
    parts = []
    const = round(coef_vec[0], 10)
    if abs(const) > eps:
        parts.append(f"{const:g}")
    for k, name in enumerate(free_names):
        c = round(coef_vec[1 + k], 10)
        if abs(c) < eps:
            continue
        if abs(abs(c) - 1.0) < 1e-9:
            term = name if c > 0 else f"-{name}"
        else:
            term = f"{c:g}*{name}"
        parts.append(term)
    if not parts:
        return "0"
    expr = parts[0]
    for p in parts[1:]:
        expr += f" - {p[1:]}" if p.startswith('-') else f" + {p}"
    return expr

def _print_general_solution(x_coef, free_cols):
    n = len(x_coef)
    free_names = [f"t{i+1}" for i in range(len(free_cols))]
    eps = 1e-12
    clean = lambda v: 0.0 if abs(v) < eps else round(v, 10)

    print("\n  ── Nghiệm tổng quát ──")
    for j in range(n):
        print(f"  x{j+1} = {_fmt_term(x_coef[j], free_names)}")

    print("\n  ── Dạng vector ──")
    x_p = [f"{clean(x_coef[j][0]):g}" for j in range(n)]
    print(f"  x = ({', '.join(x_p)})")
    for k, name in enumerate(free_names):
        vk = [f"{clean(x_coef[j][1+k]):g}" for j in range(n)]
        print(f"      + {name} * ({', '.join(vk)})")

def back_substitution(U, c):
    m, n = len(U), len(U[0])
    eps = 1e-12
    pivot_rows, pivot_cols = [], []
    r = 0
    for j in range(n):
        if r < m and abs(U[r][j]) > eps:
            pivot_rows.append(r)
            pivot_cols.append(j)
            r += 1
    for i in range(r, m):
        if abs(c[i]) > eps:
            return "Hệ phương trình vô nghiệm."
    free_cols = [j for j in range(n) if j not in pivot_cols]
    num_free = len(free_cols)
    x_coef = [[0.0] * (1 + num_free) for _ in range(n)]
    for idx, fj in enumerate(free_cols):
        x_coef[fj][1 + idx] = 1.0
    for i in range(len(pivot_cols) - 1, -1, -1):
        pc, pr = pivot_cols[i], pivot_rows[i]
        coef = [0.0] * (1 + num_free)
        coef[0] = c[pr]
        for j in range(pc + 1, n):
            uij = U[pr][j]
            if abs(uij) > eps:
                for k in range(1 + num_free):
                    coef[k] -= uij * x_coef[j][k]
        piv = U[pr][pc]
        x_coef[pc] = [v / piv for v in coef]
    if num_free == 0:
        return [x_coef[j][0] for j in range(n)]
    return (x_coef, free_cols)

def gaussian_eliminate(A, b, verbose=True):
    m, n = len(A), len(A[0])
    s = 0
    eps = 1e-12
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    r = 0
    for j in range(n):
        if r >= m:
            break
        max_val = abs(M[r][j])
        p = r
        for i in range(r + 1, m):
            if abs(M[i][j]) > max_val:
                max_val = abs(M[i][j])
                p = i
        if max_val < eps:
            continue
        if p != r:
            M[r], M[p] = M[p], M[r]
            s += 1
        for i in range(r + 1, m):
            factor = M[i][j] / M[r][j]
            M[i][j] = 0.0
            for k in range(j + 1, n + 1):
                M[i][k] -= factor * M[r][k]
        r += 1
    U = [row[:-1] for row in M]
    c = [row[-1] for row in M]
    x = back_substitution(U, c)
    if verbose and isinstance(x, tuple):
        _print_general_solution(x[0], x[1])
    return M, x, s