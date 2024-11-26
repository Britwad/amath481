import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags
from scipy.sparse import csr_matrix
from scipy.linalg import lu, solve_triangular
from scipy.sparse.linalg import spsolve, gmres, bicgstab
import time

def get_matrices(m, L):
    n = m * m
    dx = (2*L) / m 

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

    matA = spdiags(diagonals, offsets, n, n)/ (dx**2)

    ## construct matrix b

    diagonals = [e1.flatten(),-1*e1.flatten(), e1.flatten(),
                -1*e1.flatten()]
    offsets = [-(n-m),-m, m, n-m]

    matB = spdiags(diagonals, offsets, n, n)/ (2*dx)

    ## construct matrix c
    diagonals = [ e5.flatten(), 
                -1*e2.flatten(), e3.flatten(), 
                -1*e4.flatten() ]
    offsets = [ -m+1, -1,  1, m-1]

    matC = spdiags(diagonals, offsets, n, n) / (dx*2)
    return matA, matB, matC

def plot(sol, x, y, actually=True, rows=3, cols=3):
    if not actually:
        return
    split = 1
    for j, t in enumerate(tspan):
        if (j % split == 0):
            w = sol[:, :, j]
            plt.subplot(rows, cols, j//split + 1)
            plt.pcolor(x, y, w, shading='auto')
            plt.title(f'Time: {t}')
            plt.colorbar()

    plt.tight_layout()
    plt.show()

part_a = True
part_b = True
part_c = True

# Define parameters
tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
L = 20
n = 64
N = n * n

# Define spatial domain and initial conditions
x2 = np.linspace(-L/2, L/2, n + 1)
x = x2[:n]
y2 = np.linspace(-L/2, L/2, n + 1)
y = y2[:n]
X, Y = np.meshgrid(x, y)
w0 = np.exp(-X**2 - Y**2/20)


# Define spectral k values
kx = (2 * np.pi / L) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / L) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# PART A

def part_a_rhs(t, w, n, K, nu, A, B, C):
    wt = fft2(w.reshape(n, n))
    psi = np.real(ifft2(-wt / K)).flatten()

    wx = B.dot(w)
    wy = C.dot(w)
    px = B.dot(psi)
    py = C.dot(psi)

    return py*wx - px*wy + nu*A.dot(w)

A, B, C = get_matrices(n, L/2)

if part_a:
    start_time = time.time() # Record the start time
    sol = solve_ivp(part_a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(n, K, nu, A, B, C))
    print(f"FFT Elapsed time: {time.time()-start_time:.2f} seconds")
    A1 = sol.y.reshape(n, n, len(tspan))
    plot(A1, x, y, actually=True)

# PART B
if part_b:
    A = csr_matrix(A)
    A[0, 0] = 2

    # A \ B
    def part_b_AB_rhs(t, w, nu, A, B, C):
        psi = spsolve(A, w)

        wx = B.dot(w)
        wy = C.dot(w)
        px = B.dot(psi)
        py = C.dot(psi)

        return py*wx - px*wy + nu*A.dot(w)

    start_time = time.time() # Record the start time
    sol = solve_ivp(part_b_AB_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(nu, A, B, C))
    print(f"A\\B Elapsed time: {time.time()-start_time:.2f} seconds")
    A2 = sol.y.reshape(n, n, len(tspan))
    plot(A2, x, y, actually=True)

    # LU decomp
    def part_b_LU_rhs(t, w, nu, A, B, C, P, L, U):
        Pb = np.dot(P, w)
        y = solve_triangular(L, Pb, lower=True)
        psi = solve_triangular(U, y)

        wx = B.dot(w)
        wy = C.dot(w)
        px = B.dot(psi)
        py = C.dot(psi)

        return py*wx - px*wy + nu*A.dot(w)

    start_time = time.time()
    P, L, U = lu(A.toarray())
    sol = solve_ivp(part_b_LU_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(nu, A, B, C, P, L, U))
    print(f"LU Elapsed time: {time.time()-start_time:.2f} seconds")
    A3 = sol.y.reshape(n, n, len(tspan))
    plot(A3, x, y, actually=True)


    # BICGSTAB

    def part_b_BICGSTAB_rhs(t, w, nu, A, B, C):
        psi, _ = bicgstab(A, w)

        wx = B.dot(w)
        wy = C.dot(w)
        px = B.dot(psi)
        py = C.dot(psi)

        return py*wx - px*wy + nu*A.dot(w)

    start_time = time.time()
    sol = solve_ivp(part_b_BICGSTAB_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(nu, A, B, C))
    print(f"BICGSTAB Elapsed time: {time.time()-start_time:.2f} seconds")
    plot(sol.y.reshape(n, n, len(tspan)), x, y, actually=True)

    # GMRES

    def part_b_GMRES_rhs(t, w, nu, A, B, C):
        psi, _ = gmres(A, w)

        wx = B.dot(w)
        wy = C.dot(w)
        px = B.dot(psi)
        py = C.dot(psi)

        return py*wx - px*wy + nu*A.dot(w)

    start_time = time.time()
    sol = solve_ivp(part_b_GMRES_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(nu, A, B, C))
    print(f"GMRES Elapsed time: {time.time()-start_time:.2f} seconds")
    plot(sol.y.reshape(n, n, len(tspan)), x, y, actually=True)

if part_a and part_b:
    A1 = A1.reshape(N, len(tspan))
    A2 = A2.reshape(N, len(tspan))
    A3 = A3.reshape(N, len(tspan))

# PART C

if part_c:
    # reset parameters
    tspan = np.arange(0, 12, 1)
    nu = 0.001
    L = 20
    n = 64

    A, B, C = get_matrices(n, L/2)

    # now the fun begins
    # Two oppositely “charged” Gaussian vorticies next to each other, i.e. one with positive amplitude, the
    # other with negative amplitude.
    w0 = np.exp(-X**2 - Y**2/20) - np.exp(-(X-3)**2 - Y**2/20)
    sol = solve_ivp(part_a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(n, K, nu, A, B, C))
    plot(sol.y.reshape(n, n, len(tspan)), x, y, actually=True, rows=4)

    # Two same “charged” Gaussian vorticies next to each other.
    w0 = np.exp(-X**2 - Y**2/20) + np.exp(-(X-3)**2 - Y**2/20)
    sol = solve_ivp(part_a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(n, K, nu, A, B, C))
    plot(sol.y.reshape(n, n, len(tspan)), x, y, actually=True, rows=4)
    # Two pairs of oppositely “charged” vorticies which can be made to collide with each other.
    w0 = np.exp(-X**2 - Y**2/20) - np.exp(-(X-3)**2 - Y**2/20) + np.exp(-(X-6)**2 - Y**2/20) - np.exp(-(X-9)**2 - Y**2/20)
    sol = solve_ivp(part_a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(n, K, nu, A, B, C))
    plot(sol.y.reshape(n, n, len(tspan)), x, y, actually=True, rows=4)
    #  A random assortment (in position, strength, charge, ellipticity, etc.) of vorticies on the periodic domain.
    #  Try 10-15 vorticies and watch what happens.
    w0 = 0
    for i in range(10):
        xpos = 3*L/4 - 1.5*L*np.random.rand()
        ypos = 3*L/4 - 1.5*L*np.random.rand()
        charge = 4*np.random.rand() - 2
        xelip = 1 + 10*np.random.rand()
        yelip = 1 + 10*np.random.rand()
        w0 += charge*np.exp(-((X - xpos)**2/xelip + (Y - ypos)**2)/yelip)
    sol = solve_ivp(part_a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(n, K, nu, A, B, C))
    plot(sol.y.reshape(n, n, len(tspan)), x, y, actually=True, rows=4)
