import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.integrate import odeint
from scipy.sparse import linalg

########## For Part A #############
tol = 1e-4 ;col = ['r', 'b', 'g', 'c', 'm']
L = 4; 
K = 1; 
y_init = 1
dx = 0.1; 
xspan = np.arange(-L, L + dx, dx) 
A1 = np.zeros((len(xspan), 5))
A2 = np.zeros(5)         

def shoot(y, x, epsilon, K):
    return [y[1], (K * x**2 - epsilon) * y[0]]

eps_start = 0.1

for modes in range(1, 6):
    epsilon = eps_start
    deps = 0.2
    for _ in range(1000):    
        y_dash_init = np.sqrt(K * L**2 - epsilon)
        y0 = [1, y_dash_init]
        y = odeint(shoot, y0, xspan, args = (epsilon, K)) 

        if abs(y[-1, 1] + np.sqrt((L**2) - epsilon) * y[-1, 0]) < tol:  # check for convergence
            A2[modes - 1] = epsilon
            A1[: , modes - 1] = y[: ,0]
            break
        
        if ((-1)**(modes + 1)) * (y[-1, 1] + np.sqrt((L * L) - epsilon)*y[-1, 0]) > 0:
            epsilon += deps
        else:
            epsilon -= deps
            deps /= 2

    eps_start = epsilon + 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xspan)  # calculate the normalization
    plt.plot(xspan, abs(y[:, 0] / np.sqrt(norm)), col[modes - 1])  # plot modes

for i in range(5):
    norm = np.trapz(A1[:, i] * A1[:, i], xspan)
    A1[:, i] = abs(A1[:, i]/np.sqrt(norm))

########## For Part B #############

N = len(xspan) #81
n = N - 2 #79
A3 = np.zeros((N, 5))
A4 = np.zeros(5)         
A = np.zeros((n, n)) 

for i in range(n):
    A[i, i] = -2 - (K * dx**2 * xspan[i]**2)

for i in range(n-1):
    A[i, i+1] = 1
    A[i+1, i] = 1

A[0, 0] += 4/3 
A[0, 1] += -1/3
A[-1, -2] += -1/3
A[-1, -1] += 4/3

eigenvalues, eigenvectors = eig(A)

real_eigenvalues = eigenvalues.real
epsilon_2 = real_eigenvalues / (- dx**2)
epsilon_2 = np.sort(epsilon_2)
A4 = epsilon_2[:5]

A3[1:-1, :] = eigenvectors[:, :5].real

for i in range(5):
    norm = np.trapz(A3[:, i] * A3[:, i])
    A3[:, i] = abs(A3[:, i]/np.sqrt(norm))

print(eigenvectors.shape)
#print("This is A4: ", A4)
# print("This is A2: ", A2)
# print("This is A3: ", A3)
# print("This is A1: ", A1)

######## For Part C ############
# gamma1 = 0.05
# gamma2 = -0.05
# L1 = 2
# xspan_2 = np.arange(-L1, L1 + dx, dx) 
# A5 = np.zeros((len(xspan_2), 2))
# A6 = np.zeros(2)         
# A7 = np.zeros((len(xspan_2), 2))
# A8 = np.zeros(2)         

# def shoot_2(y, x, epsilon, K, gamma):
#     return [y[1], (K * x**2 - epsilon) * y[0]]

# eps_start = 0.1
# for modes in range(1, 3):
#     epsilon = eps_start
#     deps = 0.2
#     for _ in range(1000):    
#         y_dash_init = np.sqrt(K * L**2 - epsilon + gamma1)
#         y0 = [1, y_dash_init]
#         y = odeint(shoot_2, y0, xspan_2, args = (epsilon, K, gamma1)) 

#         if abs(y[-1, 1] + np.sqrt((L**2) - epsilon + gamma1 * abs(y[-1, 0] ** 2)) * y[-1, 0]) < tol:  # check for convergence
#             A6[modes - 1] = epsilon
#             A5[: , modes - 1] = y[: ,0]
#             break
        
#         if ((-1)**(modes + 1)) * (y[-1, 1] + np.sqrt((L**2) - epsilon + gamma1 * abs(y[-1, 0] ** 2))*y[-1, 0]) > 0:
#             epsilon += deps
#         else:
#             epsilon -= deps
#             deps /= 2
#     eps_start = epsilon + 0.1  # after finding eigenvalue, pick new start

# for i in range(2):
#     norm = np.trapz(A5[:, i] * A5[:, i], xspan_2)
#     A5[:, i] = abs(A5[:, i]/np.sqrt(norm))

# # print(A6)
# # print(A2)




