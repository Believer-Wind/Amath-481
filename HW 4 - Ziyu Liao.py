import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

# Define parameters
L = 10
m = 8  # N value in x and y directions
n = m * m  # Total size of matrix
delta = (2 * L) / m  # Distance

# Construct initial vectors
e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))  # vector of ones
e2 = np.copy(e1)  # copy the one vector
e4 = np.copy(e0)  # copy the zero vector

# ====== Construct Martrix A ======
for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements of A
diagonals_A = [e1.flatten(), e1.flatten(), e5.flatten(),
               e2.flatten(), -4 * e1.flatten(), e3.flatten(),
               e4.flatten(), e1.flatten(), e1.flatten()]
offsets_A = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]

A = spdiags(diagonals_A, offsets_A, n, n) / (delta ** 2)

# # Visualize A
# plt.spy(A)
# plt.title('Matrix Structure A')
# plt.show()

# ====== Contruct Matrix B ======
# Place diagonal elements of B
diagonals_B = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]
B = spdiags(diagonals_B, offsets_B, n, n) / (2 * delta)

# # Visualize B
# plt.spy(B)
# plt.title('Matrix Structure B (∂x)')
# plt.show()

# ====== Contruct Matrix C ======
# Adjust e1
for i in range(1, n):
    e1[i] = e4[i - 1]

# Place diagonal elements of C
diagonals_C = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m + 1, -1, 1,  m - 1]
C = spdiags(diagonals_C, offsets_C, n, n) / (2 * delta)

# # Visualize C
# plt.spy(C)
# plt.title('Matrix Structure C (∂y)')
# plt.show()

# ====== Convert Matrix A, B, C into a Dense Matrix ======
A_dense = A.toarray()
B_dense = B.toarray()
C_dense = C.toarray()

A1 = A_dense
A2 = B_dense
A3 = C_dense

print("Matrix A:\n", A_dense)
print("\nMatrix B (∂x):\n", B_dense)
print("\nMatrix C (∂y):\n", C_dense)
