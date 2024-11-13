import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

tol = 1e-6 ;col = ['r', 'b', 'g', 'c', 'm']
L = 4; K = 1; y_init = 1
dx = 0.1; xspan = np.arange(-L, L + dx, dx) 
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

#print(A1)
#plt.show()  # end mode loop

print(A2)
