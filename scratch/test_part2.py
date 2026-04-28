import numpy as np
from part2.decomposition import perform_qr_decomposition, multiply_matrices
from part2.diagonalization import perform_matrix_diagonalization

def test_part2():
    print("=== Testing Part 2 ===")
    
    # Test Case 1: QR Decomposition
    A = [[1, 2], [3, 4], [5, 6]]
    Q, R = perform_qr_decomposition(A)
    print("QR Decomposition calculated.")
    A_rec = np.array(Q) @ np.array(R)
    assert np.allclose(A_rec, np.array(A))
    
    # Check orthogonality of Q
    Q_np = np.array(Q)
    assert np.allclose(Q_np.T @ Q_np, np.eye(2))
    
    # Test Case 2: Diagonalization
    A_sym = [[4, 1], [1, 3]]
    D, P = perform_matrix_diagonalization(A_sym)
    print("Diagonalization calculated.")
    P_np = np.array(P)
    D_np = np.array(D)
    A_rec_diag = P_np @ D_np @ np.linalg.inv(P_np)
    assert np.allclose(A_rec_diag, np.array(A_sym))
    
    print("All Part 2 tests passed!")

if __name__ == "__main__":
    test_part2()
