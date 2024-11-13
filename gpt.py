import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Parameters
L = 4.0               # Boundary at x = L
num_eigen = 5         # Number of eigenvalues/eigenfunctions to find
tol = 1e-6            # Tolerance for root finding
x_points = 1000       # Number of points in spatial grid
x = np.linspace(0, L, x_points)  # Spatial grid from 0 to L

# Storage for eigenvalues and eigenfunctions
A1 = np.zeros((x_points, num_eigen))  # Each column will store an eigenfunction
A2 = np.zeros(num_eigen)              # Eigenvalues

def harmonic_oscillator_ode(x, y, epsilon):
    """
    Defines the ODE system for the harmonic oscillator.

    Parameters:
    - x: Independent variable
    - y: Dependent variables [phi, psi]
    - epsilon: Eigenvalue

    Returns:
    - [dphi/dx, dpsi/dx]
    """
    phi, psi = y
    dphi_dx = psi
    dpsi_dx = (x**2 + epsilon) * phi
    return [dphi_dx, dpsi_dx]

def shoot(epsilon, parity):
    """
    Integrates the ODE for a given epsilon and returns the boundary condition at x = L.

    Parameters:
    - epsilon: Eigenvalue guess
    - parity: 'even' or 'odd'

    Returns:
    - phi(L)
    """
    if parity == 'even':
        # Even eigenfunction: phi'(0) = 0
        y0 = [1.0, 0.0]  # phi(0) = 1 (arbitrary), phi'(0) = 0
    else:
        # Odd eigenfunction: phi(0) = 0
        y0 = [0.0, 1.0]  # phi(0) = 0, phi'(0) = 1 (arbitrary)
    
    # Integrate from x=0 to x=L
    sol = solve_ivp(harmonic_oscillator_ode, [0, L], y0, args=(epsilon,), 
                    t_eval=[L], method='RK45')
    
    phi_L = sol.y[0, -1]
    return phi_L

# Initial guesses for eigenvalues (these are approximate for the harmonic oscillator)
# The exact eigenvalues are (2n + 1), so we can start near these values
initial_guesses = [1, 3, 5, 7, 9]

for n in range(num_eigen):
    if n % 2 == 0:
        parity = 'even'  # Starting with n=0 as even
    else:
        parity = 'odd'   # n=1 as odd, etc.
    
    # Define the root-finding function
    def func(epsilon):
        return shoot(epsilon, parity)
    
    # Use root_scalar to find the eigenvalue where func(epsilon) = 0
    # We need to bracket the root; for harmonic oscillator, eigenvalues are approximately (2n +1)
    # So we can set a window around these values
    guess = initial_guesses[n]
    bracket = [guess - 2, guess + 2]
    
    sol = root_scalar(func, bracket=bracket, method='bisect', xtol=tol)
    
    if sol.converged:
        epsilon_n = sol.root
        A2[n] = epsilon_n
        
        # Now integrate the ODE with the found epsilon to get the eigenfunction
        if parity == 'even':
            y0 = [1.0, 0.0]
        else:
            y0 = [0.0, 1.0]
        
        sol_full = solve_ivp(harmonic_oscillator_ode, [0, L], y0, args=(epsilon_n,), 
                             t_eval=x, method='RK45')
        
        phi = sol_full.y[0]
        
        # Extend to negative x using parity
        if parity == 'even':
            phi_full = np.concatenate((phi[::-1], phi[1:]))
        else:
            phi_full = np.concatenate((-phi[::-1], phi[1:]))
        
        # Normalize the eigenfunction
        dx = (2*L) / (2 * x_points -1)
        norm = np.sqrt(np.trapz(phi_full**2, dx=dx))
        phi_normalized = phi_full / norm
        
        # Store the absolute value in A1
        A1[:, n] = np.abs(phi_normalized)
        
        # Plot the eigenfunction
        plt.plot(np.linspace(-L, L, 2 * x_points -1), A1[:, n], label=f'$\phi_{n+1}(x)$')
        
        print(f'Eigenvalue {n+1}: ε = {epsilon_n:.6f}')
    else:
        print(f'Failed to converge for eigenvalue {n+1}')

plt.title('First Five Normalized Eigenfunctions of the Harmonic Oscillator (Shooting Method)')
plt.xlabel('x')
plt.ylabel('$|\phi_n(x)|$')
plt.legend()
plt.grid(True)
plt.show()

# Convert A1 to have shape (2*x_points -1, 5) since we extended to [-L, L]
A1 = A1[:2*x_points -1, :]
