from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

tol = 1e-6
colors = ['r', 'b', 'g', 'c', 'm']
L = 4
K = 1
dx = 0.1
xspan = np.arange(0, L + dx, dx)  # Integrate from x = 0 to x = L because its symetrical
x_full = np.concatenate((-xspan[::-1], xspan[1:]))  # Full domain [-L, L]
A1 = np.zeros((len(x_full), 5))
A2 = np.zeros(5)


def shoot(y, x, epsilon):
    return [y[1], (K * x ** 2 - epsilon) * y[0]]


for modes in range(5):

    if modes % 2 == 0:
        # Even modes phi(0) = 1, phi'(0) = 0
        y0 = [1.0, 0.0]
    else:
        # Odd modes: phi(0) = 0, phi'(0) = 1
        y0 = [0.0, 1.0]

    epsilon_lower = 2 * modes + 0.5
    epsilon_upper = 2 * modes + 1.5

    for iteration in range(100):
        epsilon = (epsilon_lower + epsilon_upper) / 2.0
        y = odeint(shoot, y0, xspan, args=(epsilon,))
        phi_L = y[-1, 0]

        if abs(phi_L) < tol:
            A2[modes] = epsilon
            if modes % 2 == 0:
                phi_full = np.concatenate((y[::-1, 0], y[1:, 0]))
            else:
                phi_full = np.concatenate((-y[::-1, 0], y[1:, 0]))

            # Normalize the eigenfunction
            norm = np.trapz(phi_full ** 2, x_full)
            phi_normalized = phi_full / np.sqrt(norm)
            A1[:, modes] = np.abs(phi_normalized)

            plt.plot(x_full, np.abs(phi_normalized), colors[modes], label=f"Mode {modes + 1}")
            break  # Exit the eigenvalue adjustment loop


        y_lower = odeint(shoot, y0, xspan, args=(epsilon_lower,))
        phi_L_lower = y_lower[-1, 0]

        if phi_L * phi_L_lower < 0:
            epsilon_upper = epsilon
        else:
            epsilon_lower = epsilon

plt.xlabel('x')
plt.ylabel('Normalized Eigenfunction')
plt.title('Quantum Harmonic Oscillator Eigenfunctions')
plt.legend()
plt.grid(True)
plt.show()

print("Eigenvalues (A2):", A2)