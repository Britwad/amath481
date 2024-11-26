import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation  # Import for animations
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags

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

def animate_plot(sol, x, y, tspan, save=False, filename='animation.mp4', interval = 100):
    """
    Creates an animation of the solution over time.

    Parameters:
    - sol: numpy array of shape (n, n, num_times)
    - x, y: spatial coordinates
    - tspan: array of time points
    - rows, cols: layout of subplots (not used here but kept for compatibility)
    - save: bool, whether to save the animation
    - filename: string, filename for saving the animation
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.pcolor(x, y, sol[:, :, 0], shading='auto', cmap='jet')
    ax.set_title(f'Time: {tspan[0]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(cax, ax=ax, label='Vorticity')

    def update(frame):
        ax.clear()
        cax = ax.pcolor(x, y, sol[:, :, frame], shading='auto', cmap='jet')
        ax.set_title(f'Time: {tspan[frame]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return cax,

    ani = animation.FuncAnimation(
        fig, update, frames=range(sol.shape[2]), blit=False, interval=interval, repeat=False
    )

    if save:
        # To save as MP4, ensure ffmpeg is installed
        ani.save(filename, writer='ffmpeg')
        print(f'Animation saved as {filename}')
    else:
        plt.show()

# Define parameters
tspan = np.arange(0, 30.5, 0.5)
nu = 0.001
L = 20
n = 128
N = n * n

# Define spatial domain and initial conditions
x2 = np.linspace(-L/2, L/2, n + 1)
x = x2[:n]
y2 = np.linspace(-L/2, L/2, n + 1)
y = y2[:n]
X, Y = np.meshgrid(x, y)


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

w0 = np.exp(-X**2 - Y**2/20)
sol = solve_ivp(part_a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(n, K, nu, A, B, C))
A1 = sol.y.reshape(n, n, len(tspan))
animate_plot(A1, x, y, tspan)

# Two oppositely “charged” Gaussian vorticies next to each other, i.e. one with positive amplitude, the
# other with negative amplitude.
w0 = np.exp(-X**2 - Y**2/20) - np.exp(-(X-3)**2 - Y**2/20)
sol = solve_ivp(part_a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(n, K, nu, A, B, C))
animate_plot(sol.y.reshape(n, n, len(tspan)), x, y, tspan)

# Two same “charged” Gaussian vorticies next to each other.
w0 = np.exp(-X**2 - Y**2/20) + np.exp(-(X-3)**2 - Y**2/20)
sol = solve_ivp(part_a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(n, K, nu, A, B, C))
animate_plot(sol.y.reshape(n, n, len(tspan)), x, y, tspan)
# Two pairs of oppositely “charged” vorticies which can be made to collide with each other.
w0 = np.exp(-(X+4.5)**2 - Y**2/20) - np.exp(-(X+1.5)**2 - Y**2/20) + np.exp(-(X-1.5)**2 - Y**2/20) - np.exp(-(X-4.5)**2 - Y**2/20)
sol = solve_ivp(part_a_rhs, [tspan[0], tspan[-1]], w0.flatten(), method="RK45", t_eval=tspan, args=(n, K, nu, A, B, C))
animate_plot(sol.y.reshape(n, n, len(tspan)), x, y, tspan)
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
animate_plot(sol.y.reshape(n, n, len(tspan)), x, y, tspan)
