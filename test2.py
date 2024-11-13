import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# Parameters
L = 10               # Large value to approximate infinity
N = 1000             # Number of grid points
dx = 2 * L / (N - 1) # Spatial step size
x = np.linspace(-L, L, N)
n0 = 1               # Arbitrary constant

# Compute n(x)
n_x = np.zeros(N)
for i in range(N):
    xi = x[i]
    if abs(xi) <= 1:
        n_x[i] = n0 * (1 - xi**2)
    else:
        n_x[i] = 0

# Exclude boundary points (Dirichlet conditions ψ(-L) = ψ(L) = 0)
N_int = N - 2             # Number of interior points
x_int = x[1:-1]           # Interior spatial grid
n_x_int = n_x[1:-1]       # n(x) at interior points

# Set up the finite difference matrix A
main_diag = 2.0 / dx**2 - n_x_int
off_diag = -1.0 / dx**2 * np.ones(N_int - 1)

# Assemble the tridiagonal matrix
diagonals = [main_diag, off_diag, off_diag]
offsets = [0, -1, 1]
A = diags(diagonals, offsets, format='csr')

# Compute the eigenvalues and eigenvectors
num_eigenvalues = 5  # Number of eigenvalues/eigenfunctions to compute
eigenvalues, eigenvectors = eigsh(A, k=num_eigenvalues, which='SA')

# Convert eigenvalues to β_n (β_n = -λ_n)
beta_n = -eigenvalues

# Include boundary points in eigenfunctions
psi_n = np.zeros((N, num_eigenvalues))
for i in range(num_eigenvalues):
    # Eigenvectors correspond to ψ_n at interior points
    psi_n[1:-1, i] = eigenvectors[:, i]
    # ψ(-L) = ψ(L) = 0 due to Dirichlet boundary conditions

# Normalize the eigenfunctions
for i in range(num_eigenvalues):
    norm = np.sqrt(np.trapz(np.abs(psi_n[:, i])**2, x))
    psi_n[:, i] /= norm

# Save the absolute value of the eigenfunctions and eigenvalues
A1 = np.abs(psi_n)       # Matrix of absolute eigenfunctions (columns 0 to 4)
A2 = beta_n              # Vector of eigenvalues

# ANSWERS
A1  # Matrix containing the absolute eigenfunctions (columns 0 to 4)
A2  # Vector containing the eigenvalues
import matplotlib.pyplot as plt

for i in range(num_eigenvalues):
    plt.plot(x, psi_n[:, i], label=f'Eigenfunction {i+1}, β={beta_n[i]:.4f}')

plt.xlabel('x')
plt.ylabel('ψ_n(x)')
plt.title('First Five Eigenfunctions')
plt.legend()
plt.grid(True)
plt.show()
