import numpy as np
import matplotlib.pyplot as plt

x = np.array([-1.6])
for j in range(100):
    x = np.append(
        x, x[j] - ((x[j] * np.sin(3 * x[j]) - np.exp(x[j]))/(np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j])))
    )
    fc = x[j+1] * np.sin(3 * x[j+1]) - np.exp(x[j+1])

    if abs(fc) < 1e-6:
        print(x[j])
        #print(fc)
        #print(j)
        A1 = x[j]
        nr_iterations = j
        break

xl = -0.7
xr = -0.4
for i in range(0, 100):
    x = (xl + xr)/2
    fc = x * np.sin(3 * x) - np.exp(x)
    if (fc > 0):
        xl = x
    else:
        xr = x
    
    if(abs(fc) < 1e-6):
        print(x)
        #print(fc)
        #print(i)
        A2 = x
        bi_iterations = i
        break

A3 = np.array([nr_iterations, bi_iterations])


A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

A4 = A + B
A5 = 3 * x - 4 * y
A6 = np.dot(A, x)
A7 = np.dot(B, x - y)
A8 = np.dot(D, x)
A9 = np.dot(D, y) + z
A10 = np.dot(A, B)
A11 = np.dot(B, C)
A12 = np.dot(C, D)



