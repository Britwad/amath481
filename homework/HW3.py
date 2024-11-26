import numpy as np
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_ivp
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from scipy.integrate import quad

graph = False

# part (a)
print("Starting part (a)")
def shoot(x, y, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors

L = 4
xshoot =  np.linspace(-L, L, 81)
epsilon_start = 0.2  # beginning value of epsilon

A1 = np.zeros((81, 5)) 
A2 = np.zeros((5,))

for modes in range(0, 5):  # begin mode loop
    epsilon = epsilon_start  # initial value of eigenvalue epsilon
    y0 = [1, np.sqrt(L**2 - epsilon)]
    d_epsilon = 0.01  # default step size in epsilon
    for _ in range(1000):  # begin convergence loop for epsilon
        sol = solve_ivp(shoot, [xshoot[0], xshoot[-1]], y0, t_eval=xshoot, args=(epsilon,)) 
        y = sol.y.T
        boundary_R = y[-1, 1] + y[-1, 0]*np.sqrt(L**2 - epsilon)
        if abs(boundary_R) < tol:  # check for convergence
            # print(f"Epsilon {modes}:", epsilon)  # write out eigenvalue
            break  # get out of convergence loop

        if (-1) ** (modes) * (boundary_R)  > 0:  # adjust epsilon
            epsilon += d_epsilon
        else:
            epsilon -= d_epsilon / 2
            d_epsilon /= 2

    epsilon_start = epsilon + 2  # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xshoot)  # calculate the normalization
    A1[:, modes] = np.abs(y[:, 0] / np.sqrt(norm))  # store eigenfunction
    A2[modes] = epsilon  # store eigenvalue
    plt.plot(xshoot, A1[:, modes], col[modes], label=f'ϕ{modes}')  # plot modes

if graph:
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('ϕn(x)')
    plt.title('First Five Normalized Eigenfunctions')
    plt.show()

# part (b)
print("Starting part (b)")
L = 4
dx = 0.1
K = 1
x_span = np.arange(-L, L + dx, dx)
N = len(x_span)

A = np.zeros((N-2, N-2))
for i in range(N-2):
    A[i, i] = 2 + dx*dx*x_span[i+1]*x_span[i+1]
for i in range(N-3):
    A[i, i+1] = -1
    A[i+1, i] = -1
A[0,0] -= 4/3
A[0,1] += 1/3
A[-1,-1] -= 4/3
A[-1,-2] += 1/3

eigvals, eigfuncs = eigs(A, k=5, which='SM')

top = np.array(4*eigfuncs[0, :]/3 - eigfuncs[1, :]/3)
bottom = np.array(4*eigfuncs[-1, :]/3 - eigfuncs[-2, :]/3)
A3 = np.vstack([top, eigfuncs, bottom])
A4 = np.real(eigvals)/(dx*dx)

A3 = np.array([np.abs(A3[:, i]) / np.sqrt(np.trapz(np.abs(A3[:, i])**2)) for i in range(A3.shape[1])]).T
for n in range(5):
    norm = np.trapz(A3[:,n]*A3[:, n], x_span)
    normed = A3[:, n] / np.sqrt(norm)
    A3[:, n] = normed

plt.figure(figsize=(10, 6))
for i in range(A3.shape[1]):
    plt.plot(x_span, np.abs(A3[:, i]), label=f'Eigenfunction {i+1}')

if graph:
    plt.xlabel('x')
    plt.ylabel('Normalized Eigenfunction')
    plt.title('Normalized Eigenfunctions')
    plt.legend()
    plt.show()

# part (c)
print("Starting part (c)")

def shoot_c(x, y, epsilon, gamma):
    return [y[1], (gamma*np.abs(y[0])**2 + x*x - epsilon) * y[0]]

L = 2
K = 1
dx = 0.1
x_span = np.arange(-L, L + dx, dx)
N = len(x_span)
tol = 1e-4

esol_pos, esol_neg = np.zeros(2), np.zeros(2)
ysol_pos, ysol_neg = np.zeros((N, 2)), np.zeros((N, 2))

for gamma in [0.05, -0.05]:
    E0 = 0.1
    A = 1e-6
    for mode in range(1, 3):
        da = 0.01
        for i in range(1000):
            epsilon = E0
            depsilon = 0.2
            for j in range(1000):
                y0 = [A, A * np.sqrt(L**2 - epsilon)]
                sol = solve_ivp(shoot_c, [-L, L + dx], y0, t_eval=x_span, args=(epsilon, gamma))
                ys = sol.y.T
                xs = sol.t

                bc = ys[-1, 1] + np.sqrt(L**2 - epsilon) * ys[-1, 0]
                if abs(bc) < tol:
                    # print(f"Gamma {gamma}, Mode {mode} converged at {j} iterations")
                    break
                if (-1)**(mode + 1)*bc > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2
            area = simpson(ys[:, 0]**2, x=xs)
            if abs(area-1) < tol:
                # print(f"Gamma {gamma}, Mode {mode} area converged at {i} iterations")
                break
            if area < 1:
                A += da
            else:
                A -= da/2
                da /= 2
        E0 = epsilon + 0.2

        if gamma > 0:
            esol_pos[mode-1] = epsilon
            ysol_pos[:, mode-1] = abs(ys[:, 0])
        else:
            esol_neg[mode-1] = epsilon
            ysol_neg[:, mode-1] = abs(ys[:, 0])
        
if graph:
    plt.plot(x_span, ysol_pos[:, 0], 'r', label='ϕ1')
    plt.plot(x_span, ysol_pos[:, 1], 'b', label='ϕ2')
    plt.plot(x_span, ysol_neg[:, 0], 'g', label='ϕ1')
    plt.plot(x_span, ysol_neg[:, 1], 'c', label='ϕ2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('ϕn(x)')
    plt.title('First Two Normalized Eigenfunctions')
    plt.show()

A5 = ysol_pos
A6 = esol_pos
A7 = ysol_neg
A8 = esol_neg

# part (d)
print("Starting part (d)")

# Define parameters
L = 2
K = 1
E = 1
gamma = 0
x_span = [-L, L]
tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# Initial conditions
phi0 = 1
phi_x0 = np.sqrt(K * L**2 - E)
y0 = [phi0, phi_x0]

# Right-hand side of the ODE
def hw1_rhs_a(x, y, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

slopes = []
# Function to perform convergence study for each method
def run_convergence_study(method_name):
    avg_step_sizes = []
    for tol in tolerances:
        options = {'rtol': tol, 'atol': tol}
        sol = solve_ivp(hw1_rhs_a, x_span, y0, method=method_name, args=(E,), **options)
        
        # Calculate average step size
        step_sizes = np.diff(sol.t)
        avg_step_size = np.mean(step_sizes)
        avg_step_sizes.append(avg_step_size)

    # Log-log plot and slope calculation
    log_tolerances = np.log10(tolerances)
    log_avg_step_sizes = np.log10(avg_step_sizes)
    plt.plot(log_avg_step_sizes, log_tolerances, label=f'{method_name}')
    
    # Use polyfit to find the slope
    slope, _ = np.polyfit(log_avg_step_sizes, log_tolerances, 1)
    slopes.append(slope)
    # print(f"Slope for {method_name}: {slope:.2f} (Expected order: {order})")

if graph:
    plt.figure(figsize=(8, 6))
    plt.xlabel('log(Average Step Size)')
    plt.ylabel('log(Tolerance)')
    plt.title('Convergence Study for Different Tolerances')
    
run_convergence_study('RK45')  # RK45 method
run_convergence_study('RK23')  # RK23 method
run_convergence_study('Radau') # Radau method
run_convergence_study('BDF')   # BDF method

# Show the plot
if graph:
    plt.legend()
    plt.show()

# Print and save the slopes as a 4x1 vector
A9 = np.array(slopes).reshape((4,)).flatten()

# part (e)
print("Starting part (e)")

# Define the exact Gauss-Hermite polynomial solutions
def exact_eigenfunction(n, x):
    if n == 0:
        return np.exp(-x**2 / 2) / np.pi**0.25
    elif n == 1:
        return np.sqrt(2) * x * np.exp(-x**2 / 2) / np.pi**0.25
    elif n == 2:
        return (2 * x**2 - 1) * np.exp(-x**2 / 2) / (np.sqrt(2) * np.pi**0.25)
    elif n == 3:
        return (2 * x**3 - 3 * x) * np.exp(-x**2 / 2) / (np.sqrt(3) * np.pi**0.25)
    elif n == 4:
        return (4 * x**4 - 12 * x**2 + 3) * np.exp(-x**2 / 2) / (2 * np.sqrt(6) * np.pi**0.25)

# Define the exact eigenvalues
def exact_eigenvalue(n):
    return 2 * n + 1

# Compute the error norms for the eigenfunctions
def compute_eigenfunction_error(numerical, exact, x_span):
    error_func = lambda x: (np.abs(numerical(x)) - np.abs(exact(x)))**2
    error, _ = quad(error_func, x_span[0], x_span[-1])
    return error

# Compute the relative percent error for the eigenvalues
def compute_eigenvalue_error(numerical, exact):
    return 100 * np.abs((numerical - exact) / exact)

# Define the x_span for integration
x_span = np.linspace(-4, 4, 81)

# Compute errors for part (a)
A10 = np.zeros(5)
A11 = np.zeros(5)
for n in range(5):
    exact_func = lambda x: exact_eigenfunction(n, x)
    numerical_func_a = lambda x: np.interp(x, x_span, A1[:, n])
    A10[n] = compute_eigenfunction_error(numerical_func_a, exact_func, x_span)
    A11[n] = compute_eigenvalue_error(A2[n], exact_eigenvalue(n))

# Compute errors for part (b)
A12 = np.zeros(5)
A13 = np.zeros(5)
for n in range(5):
    exact_func = lambda x: exact_eigenfunction(n, x)
    numerical_func_b = lambda x: np.interp(x, x_span, A3[:, n])
    A12[n] = compute_eigenfunction_error(numerical_func_b, exact_func, x_span)
    A13[n] = compute_eigenvalue_error(A4[n], exact_eigenvalue(n))

# Display the results

# print("A1:", A1)
print("A2:", A2)
# print("A3:", A3)
print("A4:", A4)
# print("A5:", A5)
print("A6:", A6)
# print("A7:", A7)
print("A8:", A8)
print("A9:", A9)
print("A10:", A10)
print("A11:", A11)
print("A12:", A12)
print("A13:", A13)