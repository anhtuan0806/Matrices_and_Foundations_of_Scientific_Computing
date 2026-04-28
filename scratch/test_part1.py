import numpy as np
from part1.gaussian import perform_gaussian_elimination
from part1.determinant import calculate_matrix_determinant
from part1.inverse import calculate_matrix_inverse
from part1.rank_basis import calculate_rank_and_bases

def test_part1():
    print("=== Testing Part 1 ===")
    
    # Test Case 1: Unique solution
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    _, x, _ = perform_gaussian_elimination(A, b)
    print(f"Gauss (Unique): {x}")
    assert np.allclose(np.array(x), [2, 3, -1])
    
    # Test Case 2: Determinant
    det = calculate_matrix_determinant(A)
    print(f"Determinant: {det}")
    assert np.isclose(det, np.linalg.det(np.array(A)))
    
    # Test Case 3: Inverse
    inv = calculate_matrix_inverse(A)
    print("Inverse calculated.")
    assert np.allclose(np.array(inv) @ np.array(A), np.eye(3))
    
    # Test Case 4: Rank and Basis
    A2 = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]
    rank, col_b, row_b, null_b = calculate_rank_and_bases(A2)
    print(f"Rank: {rank}")
    assert rank == 2
    
    print("All Part 1 tests passed!")

if __name__ == "__main__":
    test_part1()
