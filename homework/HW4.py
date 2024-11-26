import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags


# A Matrix

m = 8    # N value in x and y directions
n = m * m  # total size of matrix
L = 10
dx = 2*L/m
dy = dx

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

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

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

matA = spdiags(diagonals, offsets, n, n).toarray()
matA /= dx**2

# Plot matrix structure
# fig, ax = plt.subplots()
# cax = ax.matshow(matA)
# fig.colorbar(cax)
# plt.show()

# B Matrix

diagonals = [-1*np.ones(n).flatten(), np.ones(n).flatten(), -1*np.ones(n).flatten(), np.ones(n).flatten()]
offsets = [n-m, m, -m, -n+m]
matB = spdiags(diagonals, offsets, n, n).toarray()
matB /= 2*dx

# Plot matrix structure
# fig, ax = plt.subplots()
# cax = ax.matshow(matB)
# fig.colorbar(cax)
# plt.show()

# C Matrix

# e1 = np.zeros((n, 1))
# e2 = np.ones((n, 1))
# for i in range(0, n):
#     if (i%m == 0):
#         e1[i] = 1
#     if (i%m == m-1):
#         e2[i] = 0

# e3 = np.zeros_like(e3)
# e3[1:n] = e2[0:n-1]
# e3[0] = e2[n-1]

# e4 = np.zeros_like(e1)
# e4[1:n] = e1[0:n-1]
# e4[0] = e1[n-1]

diagonals = [-1*e4.flatten(), e3.flatten(), -1*e2.flatten(), e5.flatten()]
offsets = [m-1, 1, -1, -m+1]
matC = spdiags(diagonals, offsets, n, n).toarray()
matC /= 2*dy

# Plot matrix structure
# fig, ax = plt.subplots()
# cax = ax.matshow(matC)
# fig.colorbar(cax)
# plt.show()

A1 = matA
A2 = matB
A3 = matC

print(type(A1))