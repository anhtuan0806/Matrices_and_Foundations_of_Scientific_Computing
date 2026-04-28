from decomposition import qr_decomposition, matrix_multiply, zeros, get_shape, print_matrix
import numpy as np
import random

def identity_matrix(n):
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I

def diagonalize(A, max_iters=1000, tol=1e-9):
    """
    Chéo hóa ma trận bằng QR Iteration.
    
    Lưu ý: Nếu thuật toán không hội tụ sau max_iters (do eigenvalues phức 
    hoặc bội số), sẽ fallback sang numpy.linalg.eig.
    """
    m, n = get_shape(A)
    if m != n:
        raise ValueError("Không thể chéo hóa ma trận vì m≠n")
    
    A_k = [row[:] for row in A] 
    P = identity_matrix(n)
    converged = False
    
    for iteration in range(max_iters):
        try:
            Q, R = qr_decomposition(A_k)
        except Exception as e:
            print(f"QR decomposition thất bại: {e}")
            break
            
        A_next = matrix_multiply(R, Q)
        P = matrix_multiply(P, Q)
        
        off_diagonal_sum = sum(abs(A_next[i][j]) for i in range(n) for j in range(i))
        A_k = A_next
        
        if off_diagonal_sum < tol:
            print(f"Thuật toán QR hội tụ sau {iteration+1} vòng lặp.\n")
            converged = True
            break
    
    # Nếu KHÔNG hội tụ sau max_iters → fallback NumPy
    if not converged:
        print(f"QR iteration không hội tụ sau {max_iters} vòng lặp")
        print("=> Chuyển sang sử dụng NumPy fallback...\n")
        A_np = np.array(A, dtype=float)
        eigenvalues, eigenvectors_np = np.linalg.eig(A_np)
        
        # Trích eigenvalues (giữ phần phức nếu có)
        if any(isinstance(x, (complex, np.complexfloating)) for x in eigenvalues):
            D = [[0.0j for _ in range(n)] for _ in range(n)]
        else:
            D = zeros(n, n)
        for i in range(n):
            D[i][i] = eigenvalues[i]
        
        # Lấy hết eigenvectors (kể cả phần phức nếu có)
        P = [[eigenvectors_np[i, j] for j in range(n)] 
             for i in range(n)]
        
        return D, P
    
    # Nếu hội tụ → trích xuất D từ A_k
    D = zeros(n, n)
    for i in range(n):
        D[i][i] = A_k[i][i]
    
    return D, P
# ==========================================
# KHỐI TEST SO SÁNH VỚI NUMPY
# ==========================================


def test_and_compare(name, A):
    print("="*90)
    print(f" TEST CASE: {name}")
    print("="*90)
    
    A_np = np.array(A, dtype=float)
    n = len(A)
    
    try:
        D_custom, P_custom = diagonalize(A, max_iters=1000)
        
        # Kiểm tra eigenvalue có phức không
        has_complex = any(isinstance(D_custom[i][i], (complex, np.complexfloating)) for i in range(n))
        if has_complex:
            print("   Thuật toán phát hiện eigenvalue PHỨC!")
        
        # Sắp xếp eigenvalues: nếu phức thì dùng magnitude, nếu thực dùng giá trị
        eigen_custom_list = [D_custom[i][i] for i in range(n)]
        eigen_custom = sorted(eigen_custom_list, key=lambda x: abs(x) if isinstance(x, complex) else x)
        
        # 2. Chạy NumPy để đối chiếu
        eigen_np, _ = np.linalg.eig(A_np)
        # Convert numpy types về Python types thường
        eigen_np_converted = [complex(x) if isinstance(x, (complex, np.complexfloating)) else float(x) for x in eigen_np]
        eigen_np_sorted = sorted(eigen_np_converted, key=lambda x: abs(x) if isinstance(x, complex) else x) 

        print(f"\n[Ma trận D - Tự cài (Size {n}x{n})]")
        if n <= 5:
            # Kiểm tra nếu có complex
            has_complex = any(isinstance(D_custom[i][j], (complex, np.complexfloating)) for i in range(n) for j in range(n))
            if has_complex:
                print("(Ma trận có eigenvalue phức - In dạng complex)")
            for row in D_custom:
                if has_complex:
                    print("[" + ", ".join(f"{val}" for val in row) + "]")
                else:
                    print("[" + ", ".join(f"{val:8.4f}" for val in row) + "]")
        
        # Format eigenvalues đẹp
        def format_eigen(x):
            if isinstance(x, complex):
                if abs(x.imag) < 1e-10:
                    return round(x.real, 4)
                else:
                    return complex(round(x.real, 4), round(x.imag, 4))
            else:
                return round(x, 4)
        
        print(f"\n-> Trị riêng (Custom): {[format_eigen(x) for x in eigen_custom]}")
        print(f"-> Trị riêng (NumPy) : {[format_eigen(x) for x in eigen_np_sorted]}")

        # So sánh: nếu có complex so sánh magnitude, nếu thực so sánh trực tiếp
        diff = sum(abs(c - n_val) for c, n_val in zip(eigen_custom, eigen_np_sorted))
        print(f"\n=> Tổng sai số so với NumPy: {diff:.2e}")
        
        # Kiểm tra tái cấu trúc
        P_np = np.array(P_custom, dtype=complex)  
        D_np = np.array(D_custom, dtype=complex)
        
        P_inv = np.linalg.inv(P_np)
        A_rec = P_np @ D_np @ P_inv
        
        # Tính sai số giữa A ban đầu và A tái cấu trúc
        A_np_complex = np.array(A_np, dtype=complex)
        reconstruction_diff = np.linalg.norm(A_np_complex - A_rec)
        print(f"=> Sai số tái cấu trúc (A - PDP^-1): {reconstruction_diff:.2e}")
        
        # --- CẬP NHẬT LẠI LOGIC ĐÁNH GIÁ ---
        if diff < 1e-4 and reconstruction_diff < 1e-4:
            print("=> ĐÁNH GIÁ: CHÍNH XÁC & CHÉO HÓA THÀNH CÔNG ")
        elif diff < 1e-4:
            print("=> ĐÁNH GIÁ: TÌM ĐÚNG TRỊ RIÊNG NHƯNG KHÔNG THỂ CHÉO HÓA (Defective) ")
        else:
            print("=> ĐÁNH GIÁ: CÓ SAI SỐ LỚN ")

    except Exception as e:
        print(f"-> LỖI: {e}")
    print("\n")

    
def generate_random_sym(n):
    mat = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = round(random.uniform(-10, 10), 2)
            mat[i][j] = mat[j][i] = v
    return mat

if __name__ == "__main__":
    print("BẮT ĐẦU CHẠY BỘ TEST CASE ĐÁNH GIÁ THUẬT TOÁN CHÉO HÓA\n")

    # 1. Trường hợp dễ nhất: Ma trận đã chéo hóa sẵn (Diagonal)
    test_and_compare("1. Ma trận đường chéo 3x3", 
                     [[2, 0, 0], [0, -1, 0], [0, 0, 3]])

    # 2. Trường hợp lý tưởng: Ma trận đối xứng chuẩn
    test_and_compare("2. Ma trận đối xứng 4x4", 
                     [[4, 1, -1, 0], [1, 2, 0, 1], [-1, 0, 3, 2], [0, 1, 2, 4]])

    # 3. Trị riêng lặp (Repeated Eigenvalues): Có nhiều trị riêng giống nhau
    # Trị riêng: 5, 2, 2
    test_and_compare("3. Ma trận đối xứng 3x3 có trị riêng lặp (Bội nghiệm)", 
                     [[3, 1, 1], [1, 3, 1], [1, 1, 3]])

    # 4. Sát thủ tốc độ: Trị riêng cực kỳ sát nhau
    # Tỷ lệ lambda_1/lambda_2 xấp xỉ 1 -> Thuật toán lặp sẽ chạy vô cùng chậm
    test_and_compare("4. Trị riêng sát nhau (1.000 và 1.001)", 
                     [[1.0, 0.01, 0], [0.01, 1.01, 0], [0, 0, 5.0]])

    # 5. Ma trận suy biến (Singular Matrix): Định thức = 0
    # Phải có ít nhất 1 trị riêng bằng 0
    test_and_compare("5. Ma trận suy biến 3x3", 
                     [[1, 2, 3], [4, 5, 6], [5, 7, 9]])

    # 6. Ma trận bất đối xứng có nghiệm thực
    # QR sẽ kéo nó về dạng tam giác trên (Schur form) thay vì chéo hóa hoàn toàn
    test_and_compare("6. Ma trận bất đối xứng 3x3 (Nghiệm thực)", 
                     [[1, 2, 0], [-1, 4, 0], [0, 0, 5]])

    # 7. Ma trận khiếm khuyết (Defective Matrix - Jordan Block)
    # Không có đủ vector riêng độc lập tuyến tính, không thể chéo hóa
    test_and_compare("7. Ma trận Không thể chéo hóa (Khối Jordan 3x3)", 
                     [[2, 1, 0], [0, 2, 1], [0, 0, 2]])

    # 8. Ma trận phi đối xứng nhưng eigenvalue thực
    test_and_compare("8. Ma trận bất đối xứng 3x3 (Eigenvalue thực)", 
                     [[1, 2, 0], [-1, 4, 0], [0, 0, 5]])

    # 9. Ma trận Hilbert Matrix 5x5
    # Bài toán nổi tiếng về "sai số làm tròn". Số nhỏ sẽ bị khuếch đại.
    H5 = [[1.0 / (i + j + 1) for j in range(5)] for i in range(5)]
    test_and_compare("9. Ma trận Hilbert 5x5", H5)

    # 10. ma trận 10x10
    test_and_compare("10. Stress Test: Ma trận đối xứng ngẫu nhiên 10x10", 
                     generate_random_sym(10))

    # 11. Ma trận 2x2 đơn giản
    test_and_compare("11. Ma trận 2x2 đơn giản", 
                     [[1, 2], [2, 3]])

    # 12. Ma trận với tất cả trị riêng âm
    test_and_compare("12. Ma trận với tất cả eigenvalue âm", 
                     [[-1, 0.5], [0.5, -2]])

    # 13. Ma trận scale cực đoan (1e-5 đến 1e5)
    test_and_compare("13. Ma trận scale cực đoan (1e-5 đến 1e5)", 
                     [[1e-5, 0, 0], [0, 1, 0], [0, 0, 1e5]])

    # 14. Ma trận toàn 0
    test_and_compare("14. Ma trận toàn 0", 
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0]])