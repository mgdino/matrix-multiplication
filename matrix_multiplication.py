import numpy as np

def naive_matrix_multiplication(A, B):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return np.array(C)

def split(matrix):
    row, col = matrix.shape
    row2, col2 = row // 2, col // 2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

def strassen(x, y):
    if len(x) == 1:
        return x * y
    
    a, b, c, d = split(x)
    e, f, g, h = split(y)
    
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)
    
    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7
    
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
    return c

def coppersmith_winograd(x, y):
    n = x.shape[0]
    
    if n == 1:
        return np.dot(x, y)
    
    a = np.random.randint(0, 2, size=(n, 1))
    
    xa = np.dot(x, a)
    ya = np.dot(y, a)
    
    a11, a12, a21, a22 = split(x)
    b11, b12, b21, b22 = split(y)
    
    m1 = np.dot(a11 + a22, b11 + b22)
    m2 = np.dot(a21 + a22, b11)
    m3 = np.dot(a11, b12 - b22)
    m4 = np.dot(a22, b21 - b11)
    m5 = np.dot(a11 + a12, b22)
    m6 = np.dot(a21 - a11, b11 + b12)
    m7 = np.dot(a12 - a22, b21 + b22)
    
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 + m3 - m2 + m6
    
    result = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
    
    if not np.allclose(np.dot(x, ya), np.dot(result, a)):
        raise ValueError("Matrix multiplication verification failed")
    
    return result

def verify_multiplication(A, B, C, tolerance=1e-10):
    actual_product = np.dot(A, B)
    return np.allclose(actual_product, C, rtol=tolerance)
