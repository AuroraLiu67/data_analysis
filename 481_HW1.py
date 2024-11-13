import numpy as np
import matplotlib.pyplot as plt

x = np.array([-1.6])
iter = 0
for j in range(100):
    
    x = np.append(
        x, x[j] - ((x[j] * np.sin(3 * x[j]) - np.exp(x[j]))/ (np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j]))))
    fc = x[j] * np.sin(3 * x[j]) -np.exp(x[j])
    if (abs(fc) < 1e-6):
        break

A1 = x
nr_iterations = j + 1

xl = -0.7
xr = -0.4
xbi = []
iter = 0
for i in range(0, 100):
    iter = iter + 1
    
    x = (xl + xr)/2
    xbi.append(x)
    fc = x * np.sin(3 * x) - np.exp(x)
    if (fc > 0):
        xl = x
    else:
        xr = x
    
    if(abs(fc) < 1e-6):
        A2 = xbi
        bi_iterations = i+1
        break

A3 = np.array([nr_iterations, bi_iterations])

A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1 ,2, -1])

A4 = A + B
A5 = 3 * x - 4 * y
A6 = np.dot(A, x)
A7 = np.dot(B, x - y)
A8 = np.dot(D, x)
A9 = (np.dot(D, y)) + z
A10 = np.dot(A, B)
A11 = np.dot(B, C)
A12 = np.dot(C, D)

print(A3)