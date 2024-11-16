import numpy as np 
from matrix_multiplication import naive_matrix_multiplication, strassen, coppersmith_winograd


A = np.random.randint(1, 11, size=(4, 4))
B = np.random.randint(1, 11, size=(4, 4))

print("Input matrices:")
print("A:\n", A)
print("B:\n", B)

from matrix_multiplication import naive_matrix_multiplication, strassen, coppersmith_winograd

result_naive = naive_matrix_multiplication(A, B)
result_strassen = strassen(A, B)
result_coppersmith = coppersmith_winograd(A, B)

print("\nNaive matrix multiplication:\n", result_naive)
print("\nStrassen matrix multiplication:\n", result_strassen)
print("\nCoppersmith-Winograd matrix multiplication:\n", result_coppersmith)

