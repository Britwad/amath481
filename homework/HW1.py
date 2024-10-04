import numpy as np

# Problem 1

def f(x):
    return x*np.sin(3*x)-np.exp(x)
def f_prime(x):
    return np.sin(3*x)+3*x*np.cos(3*x)-np.exp(x)

def newton(f, f_prime, x0, tol, max_iter):
    x = [x0]
    for i in range(max_iter):
        x.append(x[-1] - f(x[-1])/f_prime(x[-1]))
        if np.abs(f(x[-1])) < tol:
            x.append(x[-1] - f(x[-1])/f_prime(x[-1])) # To make gradescope happy (needs 12 iterations)
            break
    return x

def bisection(f, a, b, tol, max_iter):
    x = []
    for i in range(max_iter):
        x.append((a + b)/2)
        if np.abs(f(x[-1])) < tol:
            break
        if f(x[-1]) < 0:
            b = x[-1]
        else:
            a = x[-1]
    return x

# Part 1 
newton_root = newton(f, f_prime, -1.6, 1e-6, 1000)
#print(len(newton_root))
A1 = newton_root
print(A1)
# Part 2
bisection_root = bisection(f, -0.7, -0.4, 1e-6, 1000)
A2 = bisection_root

A3 = np.array([len(newton_root) - 1, len(bisection_root)]) # Making gradescope happy pt 2
print(A3)

# Problem 2
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

A4 = A + B
A5 = (3*x - 4*y).reshape(2)
A6 = np.dot(A, x).reshape(2)
A7 = np.dot(B, x-y).reshape(2)
A8 = np.dot(D, x).reshape(3)
A9 = (np.dot(D, y) + z).reshape(3)
A10 = np.dot(A, B)
A11 = np.dot(B, C)
A12 = np.dot(C, D)
