import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

tol = 1e-4 ;col = ['r', 'b', 'g', 'c', 'm']
L = 4; K = 1
dx = 0.1; xspan = np.arange(-L, L + dx, dx) 
A1 = np.zeros((len(xspan), 5))
A2 = np.zeros(5)         
y_init = 1

def shoot(y, x, K, epsilon):
    return [y[1], (K * x**2 - epsilon) * y[0]]

eps_start = 0.1
for modes in range(5):
    epsilon = eps_start
    deps = 0.1
    
    for _ in range(1000):    
        y_dash_init = np.sqrt(K*L**2 - epsilon) * y_init
        y0 = [y_init, y_dash_init]
        y = odeint(shoot, y0, xspan, args = (epsilon, K)) 

        if abs(y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) < tol:  # check for convergence
            A2[modes] = epsilon
            A1[: , modes] = y[: ,0]
            break
        
        if (-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) > 0:
            epsilon += deps
        else:
            epsilon -= deps / 2
            deps /= 2

    eps_start = epsilon - 0.1  # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xspan)  # calculate the normalization
    plt.plot(xspan, y[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes

# for i in range(5):
#     norm = np.trapz(A1[:, i] * A1[:, i], xspan)
#     A1[:, i] = abs(A1[:, i]/np.sqrt(norm))
plt.show()  # end mode loop
print(A2)



