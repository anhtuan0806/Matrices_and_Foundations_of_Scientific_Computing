import math

# ==========================================
# CÁC HÀM TIỆN ÍCH XỬ LÝ MA TRẬN (CÀI CHAY)
# ==========================================

def get_shape(A):
    if not A: return 0, 0
    return len(A), len(A[0])

def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def get_column(A, col_idx):
    return [row[col_idx] for row in A]

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def norm(v):
    return math.sqrt(dot_product(v, v))

def matrix_multiply(A, B):
    m, n = get_shape(A)
    p, q = get_shape(B)
    if n != p:
        raise ValueError("Kích thước ma trận không khớp để nhân!")
    
    C = zeros(m, q)
    for i in range(m):
        for j in range(q):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    return C

def transpose(A):
    m, n = get_shape(A)
    return [[A[i][j] for i in range(m)] for j in range(n)]

def print_matrix(name, A):
    print(f"{name}:")
    for row in A:
        print("[" + ", ".join(f"{val:8.4f}" for val in row) + "]")
    print()

# ==========================================
# THUẬT TOÁN PHÂN RÃ QR (GRAM-SCHMIDT)
# ==========================================

def qr_decomposition(A):
    """
    Phân rã QR sử dụng Gram-Schmidt.
    A (m x n) = Q (m x n) * R (n x n)
    """
    m, n = get_shape(A)
    if m < n:
        raise ValueError("Số dòng phải lớn hơn hoặc bằng số cột (m >= n)")
    if not A or not A[0]:
        raise ValueError("Ma trận không được rỗng")
    if any(len(row) != n for row in A):
        raise ValueError("Ma trận không đồng nhất (các hàng có độ dài khác nhau)")
    
    Q = zeros(m, n)
    R = zeros(n, n)
    
    for j in range(n):
        v = get_column(A, j)
        for i in range(j):
            q_i = get_column(Q, i) # Lấy vector trực chuẩn q_i đã tính trước đó
            R[i][j] = dot_product(q_i, v)
            # Trừ đi hình chiếu
            v = [v_k - R[i][j] * q_i_k for v_k, q_i_k in zip(v, q_i)]
            
        R[j][j] = norm(v)
        
        # Chuẩn hóa vector phần dư để tạo cột j cho Q
        if R[j][j] > 1e-10:
            for k in range(m):
                Q[k][j] = v[k] / R[j][j]
        else:
            for k in range(m):
                Q[k][j] = 0.0 # Bắt lỗi phụ thuộc tuyến tính
                
    return Q, R


# ==========================================
# TEST THUẬT TOÁN VÀ ĐÁNH GIÁ SAI SỐ
# ==========================================

if __name__ == "__main__":
    import numpy as np
    
    print("="*80)
    print("KIỂM THỬ TÍNH CHẤT MA TRẬN, SO SÁNH VỚI NUMPY & ĐÁNH GIÁ SAI SỐ")
    print("="*80)

    def compare_with_numpy(Q_custom, R_custom, Q_np, R_np):
        """
        Đo độ lệch giữa code tự cài và NumPy.
        Dùng trị tuyệt đối (np.abs) để tránh lỗi lệch dấu do khác biệt về bản chất thuật toán.
        """
        Q_c = np.abs(np.array(Q_custom, dtype=float))
        R_c = np.abs(np.array(R_custom, dtype=float))
        Q_n = np.abs(Q_np)
        R_n = np.abs(R_np)
        
        diff_Q = np.linalg.norm(Q_c - Q_n, ord='fro')
        diff_R = np.linalg.norm(R_c - R_n, ord='fro')
        return diff_Q, diff_R

    def check_orthogonality(Q_custom):
        """
        Kiểm tra sai số mất tính trực giao: ||Q^T * Q - I||_F
        """
        Q_np = np.array(Q_custom, dtype=float)
        n = Q_np.shape[1]
        I_np = np.eye(n)
        return np.linalg.norm(Q_np.T @ Q_np - I_np, ord='fro')

    def test_and_compare(test_name, A_list):
        print(f"\n{'-'*80}")
        print(f"{test_name}")
        print(f"{'-'*80}")
        print_matrix("Ma trận A", A_list)
        
        # 1. Chạy code tự cài
        try:
            Q_custom, R_custom = qr_decomposition(A_list)
        except Exception as e:
            print(f"Thuật toán tự cài gặp lỗi: {e}")
            return
        
        # 2. Chạy bằng NumPy
        A_np = np.array(A_list, dtype=float)
        Q_np, R_np = np.linalg.qr(A_np)
        
        # In kết quả để đối chiếu
        print_matrix("Q (Tự cài)", Q_custom)
        print("Q (NumPy):")
        print(np.round(Q_np, 4), "\n")
        
        print_matrix("R (Tự cài)", R_custom)
        print("R (NumPy):")
        print(np.round(R_np, 4), "\n")
        
        # 3. SO SÁNH VỚI NUMPY & ĐÁNH GIÁ TÍNH ỔN ĐỊNH SỐ HỌC
        diff_Q, diff_R = compare_with_numpy(Q_custom, R_custom, Q_np, R_np)
        orth_err = check_orthogonality(Q_custom)
        reconstruct_err = np.linalg.norm(A_np - np.array(Q_custom) @ np.array(R_custom), ord='fro')
        
        print(f">>> ĐÁNH GIÁ SAI SỐ THUẬT TOÁN TỰ CÀI (GRAM-SCHMIDT):")
        print(f"    - Lệch ma trận Q so với NumPy   : {diff_Q:.5e}")
        print(f"    - Lệch ma trận R so với NumPy   : {diff_R:.5e}")
        print(f"    - Sai số trực giao ||Q^T Q - I|| : {orth_err:.5e}")
        print(f"    - Sai số tái tạo A ≈ QR        : {reconstruct_err:.5e}")
        TOL = 1e-9

        if diff_Q < TOL and diff_R < TOL and orth_err < TOL and reconstruct_err < TOL:
            print(">>> KẾT LUẬN:Phân rã QR ĐÚNG (sai số nhỏ)")
        elif orth_err >= TOL:
            print(">>> KẾT LUẬN: Q bị MẤT TRỰC GIAO (Gram-Schmidt không ổn định số)")
        else:
            print(">>> KẾT LUẬN: Phân rã QR SAI (sai số lớn)")


    # ---------------------------------------------------------
    # TEST CASE 1: Ma trận số nguyên (Integer)
    # ---------------------------------------------------------
    A_int = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]
    test_and_compare("[TEST 1] Ma trận số nguyên (Integer) 3x2", A_int)

    # ---------------------------------------------------------
    # TEST CASE 2: Ma trận số thực (Float) chứa số âm
    # ---------------------------------------------------------
    A_float = [
        [1.5, -2.1,  3.0],
        [-1.0, 1.2, -1.5],
        [0.5,  1.1, -2.0]
    ]
    test_and_compare("[TEST 2] Ma trận vuông số thực (Float) 3x3", A_float)

    # ---------------------------------------------------------
    # TEST CASE 3: Ma trận Phụ thuộc tuyến tính (Hạng thiếu)
    # ---------------------------------------------------------
    A_dependent = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],  
        [3.0, 6.0, 9.0]   
    ]
    test_and_compare("[TEST 3] Ma trận Phụ thuộc tuyến tính (Linear Dependent)", A_dependent)

    # ---------------------------------------------------------
    # TEST CASE 4: Ma trận Đơn vị (Identity Matrix)
    # ---------------------------------------------------------
    A_identity = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    test_and_compare("[TEST 4] Ma trận Đơn vị (Identity)", A_identity)

    # ---------------------------------------------------------
    # TEST CASE 5: Ma trận Đường chéo (Diagonal Matrix)
    # ---------------------------------------------------------
    A_diag = [
        [4.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 2.0]
    ]
    test_and_compare("[TEST 5] Ma trận Đường chéo (Diagonal)", A_diag)

    # ---------------------------------------------------------
    # TEST CASE 6: Ma trận Chữ nhật đứng (m > n) 
    # ---------------------------------------------------------
    A_tall = [
        [1.0, -1.0],
        [1.0, 4.0],
        [1.0, -1.0],
        [1.0, 4.0]
    ]
    test_and_compare("[TEST 6] Ma trận Chữ nhật đứng (Tall Matrix 4x2)", A_tall)

    # ---------------------------------------------------------
    # TEST CASE 7: Các cột đã trực giao sẵn
    # ---------------------------------------------------------
    A_orthogonal_cols = [
        [1.0,  1.0,  1.0],
        [-1.0, 1.0,  1.0],
        [0.0, -2.0,  1.0],
        [0.0,  0.0, -3.0]
    ]
    test_and_compare("[TEST 7] Ma trận có các cột trực giao sẵn (4x3)", A_orthogonal_cols)

    # ---------------------------------------------------------
    # TEST CASE 8: Ma trận có số rất lớn và rất nhỏ (Test scale)
    # ---------------------------------------------------------
    A_scale = [
        [1e6,  2e6,  3e6],
        [1e-6, 2e-6, 3e-6],
        [0.5,  0.5,  0.5]
    ]
    test_and_compare("[TEST 8] Ma trận chênh lệch Scale cực độ", A_scale)

    # ---------------------------------------------------------
    # TEST CASE 9: Ma trận ngẫu nhiên (Well-conditioned) 4x4
    # ---------------------------------------------------------
    A_random = [
        [ 0.8147,  0.0975,  0.1576,  0.1419],
        [ 0.9058,  0.2785,  0.9706,  0.4218],
        [ 0.1270,  0.5469,  0.9572,  0.9157],
        [ 0.9134,  0.9575,  0.4854,  0.7922]
    ]
    test_and_compare("[TEST 9] Ma trận ngẫu nhiên bình thường (4x4)", A_random)

    # ---------------------------------------------------------
    # TEST CASE 10: Ma trận Hilbert 3x3 (Ill-conditioned)
    # ---------------------------------------------------------
    A_hilbert = [
        [1.0,       1.0/2.0,   1.0/3.0],
        [1.0/2.0,   1.0/3.0,   1.0/4.0],
        [1.0/3.0,   1.0/4.0,   1.0/5.0]
    ]
    test_and_compare("[TEST 10] Ma trận Hilbert 3x3 (Ill-conditioned)", A_hilbert)

    # ---------------------------------------------------------
    # TEST CASE 11: Ma trận 2x2 đơn giản
    # ---------------------------------------------------------
    A_2x2 = [
        [1.0, 2.0],
        [3.0, 5.0]
    ]
    test_and_compare("[TEST 11] Ma trận 2x2 đơn giản", A_2x2)

    # ---------------------------------------------------------
    # TEST CASE 12: Ma trận toàn 0
    # ---------------------------------------------------------
    A_zero = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ]
    test_and_compare("[TEST 12] Ma trận toàn 0 (Edge case)", A_zero)

    # ---------------------------------------------------------
    # TEST CASE 13: Hạng = 1 (Rank-1 Matrix)
    # ---------------------------------------------------------
    A_rank1 = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ]
    test_and_compare("[TEST 13] Hạng = 1 (Rank-1 Matrix)", A_rank1)

    # ---------------------------------------------------------
    # TEST CASE 14: Cột gần song song (Numerically Dependent)
    # ---------------------------------------------------------
    A_nearly_parallel = [
        [1.0, 1.0 + 1e-15],
        [1.0, 1.0],
        [1.0, 1.0]
    ]
    test_and_compare("[TEST 14] Cột gần song song (Numerically Dependent)", A_nearly_parallel)

    # ---------------------------------------------------------
    # TEST CASE 15: Ma trận vuông lớn 6x6
    # ---------------------------------------------------------
    A_6x6 = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30],
        [31, 32, 33, 34, 35, 36]
    ]
    test_and_compare("[TEST 15] Ma trận vuông 6x6 (Lớn)", A_6x6)