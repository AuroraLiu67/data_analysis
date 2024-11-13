import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_ivp

############## part a #################
tol = 1e-4 ;col = ['r', 'b', 'g', 'c', 'm']
L = 4 
K = 1
y_init = 1
dx = 0.1
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
    #plt.plot(xspan, abs(y[:, 0] / np.sqrt(norm)), col[modes - 1])  # plot modes

for i in range(5):
    norm = np.trapz(A1[:, i] * A1[:, i], xspan)
    A1[:, i] = abs(A1[:, i]/np.sqrt(norm))

#print(A2)

########### part b ###############
L = 4
xspan = np.arange(-L, L + dx, dx) 
dx = xspan[1] - xspan[0]

N = len(xspan) #81
n = N - 2 #79
A3 = np.zeros((N, 5))
A4 = np.zeros(5)         
A = np.zeros((n, n)) 

for i in range(n):
    A[i, i] = -2 - (K * dx**2 * xspan[i + 1]**2)

    if i < (n-1):
        A[i, i+1] = 1
        A[i+1, i] = 1

A[0, 0] += 4/3 
A[0, 1] += -1/3
A[-1, -2] += -1/3
A[-1, -1] += 4/3

eigenvalues, eigenvectors = eigs(-A, k = 5, which = 'SM')
eigenvalues = eigenvalues.real
A4 = eigenvalues / (dx**2)

eigenvectors = eigenvectors.real
phi1 = eigenvectors[0, :]
phi2 = eigenvectors[1, :]
phi_n1 = eigenvectors[-1, :]
phi_n2 = eigenvectors[-2, :]

phi0 = 4/3 * phi1 - 1/3 * phi2
phin = 4/3 * phi_n1 - 1/3 * phi_n2

A3 = np.vstack([phi0, eigenvectors, phin])

for i in range(5):
    norm = np.trapz(abs(A3[:, i])**2, xspan)
    A3[:, i] = abs(A3[:, i]/np.sqrt(norm))
    plt.plot(xspan, A3[:, i], col[i]) 

#print(A4)

############ part c ############
L1 = 2
xspan_2 = np.arange(-L1, L1 + dx, dx) 
n = len(xspan_2)
tol = 1e-4

### for pos gamma 0.05
A5 = np.zeros((len(xspan_2), 2))
A6 = np.zeros(2)   
### for neg gamma -0.05    
A7 = np.zeros((len(xspan_2), 2))
A8 = np.zeros(2)         

def shoot_2(x, y, epsilon, gamma):
    return [y[1], (gamma * y[0]**2 + K * x**2 - epsilon) * y[0]]

for gamma in [0.05, -0.05]:
    e0 = 0.1
    A = 1e-6

    for modes in range(1, 3):
        dA = 0.01
        
        for j in range(100): #A loop
            epsilon = e0
            de = 0.2
            
            for i in range(100): #e loop
                y_dash_init = np.sqrt(L1**2 - epsilon)
                y0 = [A, y_dash_init * A]
                sol = solve_ivp(lambda x, y: shoot_2(x, y, epsilon, gamma), [xspan_2[0], xspan_2[-1]], y0, t_eval=xspan_2)
            
                y_arr = sol.y.T
                x_arr = sol.t

                bc = y_arr[-1, 1] + np.sqrt(K * (L1**2) - epsilon) * y_arr[-1, 0]
                if abs(bc) < tol:
                    break

                if (-1) ** (modes+1) * bc > 0:
                    epsilon += de
                else:
                    epsilon -= de
                    de /= 2

            area = np.trapz(y_arr[:, 0]**2, x = x_arr)
            if(abs(area - 1) < tol):
                break

            if area < 1:
                A += dA
            else:
                A -= dA
                dA /= 2

        e0 = epsilon + 0.2
        if (gamma > 0):
            A6[modes - 1] = epsilon
            A5[: , modes - 1] = abs(y_arr[: ,0])
            #plt.plot(xspan_2, A5[:, modes - 1]) 
        if (gamma < 0):
            A8[modes - 1] = epsilon
            A7[: , modes - 1] = abs(y_arr[: ,0])
            #plt.plot(xspan_2, A7[:, modes - 1])

# print('This is A6 ', A6)
# print('This is A8', A8)

########### part d ###########

def hw1_rhs_a(x, y, E):
    return [y[1], (x**2 - E) * y[0]]

L = 2
x_span = [-L, L]
E = 1
K = 1
y0 = [1, np.sqrt(K * L**2 - 1)]
TOL = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
dtRK45, dtRK23, dtRadau, dtBDF = [], [], [], []
A9 = np.zeros(4)

for tol in TOL:
    options = {'rtol': tol, 'atol': tol}
    solve45 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK45', args=(E,), **options)
    solve23 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK23', args=(E,), **options)
    solveRadau = solve_ivp(hw1_rhs_a, x_span, y0, method='Radau', args=(E,), **options)
    solveBDF = solve_ivp(hw1_rhs_a, x_span, y0, method='BDF', args=(E,), **options)

    t45 = solve45.t
    t23 = solve23.t
    tRadau = solveRadau.t
    tBDF = solveBDF.t

    dtRK45.append(np.mean(np.diff(t45)))
    dtRK23.append(np.mean(np.diff(t23)))
    dtRadau.append(np.mean(np.diff(tRadau)))
    dtBDF.append(np.mean(np.diff(tBDF)))
    # print(dtRK45)

fit45 = np.polyfit(np.log(dtRK45), np.log(TOL), 1)
fit23 = np.polyfit(np.log(dtRK23), np.log(TOL), 1)
fitRadau = np.polyfit(np.log(dtRadau), np.log(TOL), 1)
fitBDF = np.polyfit(np.log(dtBDF), np.log(TOL), 1)

A9[0] = fit45[0]
A9[1] = fit23[0]
A9[2] = fitRadau[0]
A9[3] = fitBDF[0]

A9 = A9.flatten()
# print(A9)


####### part e ################
### part a redo ###
tol = 1e-4 ;col = ['r', 'b', 'g', 'c', 'm']
L = 4 
y_init = 1
dx = 0.1
xspan = np.arange(-L, L + dx, dx) 
eigvec_numerical = np.zeros((len(xspan), 5))
eigval_numerical = np.zeros(5)         

def shoot(x, y, epsilon, K):
    return [y[1], (x**2 - epsilon) * y[0]]

eps_start = 0.1

for modes in range(1, 6):
    epsilon = eps_start
    deps = 0.2
    for _ in range(1000):    
        y_dash_init = np.sqrt(K * L**2 - epsilon)
        y0 = [1, y_dash_init]
        sol = solve_ivp(lambda x, y: shoot(x, y, epsilon, K), [xspan[0], xspan[-1]], y0, t_eval=xspan)
        y = sol.y.T
        x = sol.t

        if abs(y[-1, 1] + np.sqrt((L**2) - epsilon) * y[-1, 0]) < tol:  # check for convergence
            eigval_numerical[modes - 1] = epsilon
            eigvec_numerical[: , modes - 1] = y[: ,0]
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
    norm = np.trapz(eigvec_numerical[:, i] * eigvec_numerical[:, i], xspan)
    eigvec_numerical[:, i] = abs(eigvec_numerical[:, i]/np.sqrt(norm))

# print(eigenvalues)
x = np.arange(-L, L + dx, dx) 
h = np.array([np.ones_like(x), 2 * x, 4 * x**2 - 2, 8 * x**3 - 12*x, 16 * x**4 - 48 * x **2 + 12])
phi = np.zeros((len(h[0, :]), 5))

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

for j in range(5):
    phi[:, j] = (np.exp(-x**2 / 2) * (h[j, :] / np.sqrt(factorial(j) * (2**j) * np.sqrt(np.pi)))).T

er_psi_a = np.zeros(5)
er_psi_b = np.zeros(5)
er_a = np.zeros(5)
er_b = np.zeros(5)

for j in range(5):
    er_psi_a[j] = np.trapz((abs(eigvec_numerical[:, j]) - abs(phi[:, j])) **2, x)
    er_psi_b[j] = np.trapz((abs(A3[:, j]) - abs(phi[:, j])) **2, x)
    er_a[j] = 100 * abs(eigval_numerical[j] - (2 * (j+1) - 1)) / (2 * (j+1) - 1)
    er_b[j] = 100 * abs(A4[j] - (2 * (j+1) - 1)) / (2 * (j+1) - 1)

A10 = er_psi_a
A12 = er_psi_b

A11 = er_a
A13 = er_b
# print(h[4, :])
# print(A13)
