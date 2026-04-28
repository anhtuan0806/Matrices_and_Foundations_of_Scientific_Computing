import numpy as np
from part3.solvers import solve_system_via_gauss, solve_system_via_qr, solve_system_via_gauss_seidel

def test_part3():
    print("=== Testing Part 3 ===")
    
    # Test Case 1: 3x3 System (Diagonally Dominant for GS)
    A = [[10, 2, 1], [1, 5, 1], [2, 3, 10]]
    b = [13, 7, 15]
    
    x_gauss = solve_system_via_gauss(A, b)
    print(f"Gauss Solution: {x_gauss}")
    assert np.allclose(np.array(A) @ np.array(x_gauss), np.array(b))
    
    x_qr = solve_system_via_qr(A, b)
    print(f"QR Solution: {x_qr}")
    assert np.allclose(np.array(A) @ np.array(x_qr), np.array(b))
    
    x_gs = solve_system_via_gauss_seidel(A, b)
    print(f"Gauss-Seidel Solution: {x_gs}")
    assert np.allclose(np.array(A) @ np.array(x_gs), np.array(b), atol=1e-7)
    
    print("All Part 3 tests passed!")

if __name__ == "__main__":
    test_part3()
