import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shoot(y, x, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

tol = 1e-6  # define a tolerance level 
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
        y = odeint(shoot, y0, xshoot, args=(epsilon,)) 
        boundary_R = y[-1, 1] + y[-1, 0]*np.sqrt(L**2 - epsilon)
        if abs(boundary_R) < tol:  # check for convergence
            print(f"Epsilon {modes}:", epsilon)  # write out eigenvalue
            # print(np.shape(y))
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

plt.legend()
plt.xlabel('x')
plt.ylabel('ϕn(x)')
plt.title('First Five Normalized Eigenfunctions')
plt.show()

print(A1)
print(A2)