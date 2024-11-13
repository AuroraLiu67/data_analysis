import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

L = 20;        
n = 8;         
dx = L / n;    
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)

m = 8    # N value in x and y directions
m_2 = m * m  # total size of matrix

e0 = np.zeros((m_2, 1))  # vector of zeros
e1 = np.ones((m_2, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:m_2] = e2[0:m_2-1]
e3[0] = e2[m_2-1]

e5 = np.zeros_like(e4)
e5[1:m_2] = e4[0:m_2-1]
e5[0] = e4[m_2-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(m_2-m), -m, -m+1, -1, 0, 1, m-1, m, (m_2-m)]
A1 = spdiags(diagonals, offsets, m_2, m_2).toarray() / (dx **2) #matrix A

#Plot matrix structure
plt.figure(5)
plt.spy(A1)
plt.title('Matrix Structure A')
plt.show()

diagonals_b = [e1.flatten(), -1 * e1.flatten(),e1.flatten(), -1 * e1.flatten()]
offsets_b = [-(m_2-m), -m, m, m_2-m]
A2 = 1/(2 * dx) * spdiags(diagonals_b, offsets_b, m_2, m_2).toarray() #matrix B
plt.figure(6)
plt.spy(A2)
plt.title('Matrix Structure B')
plt.show()

c0 = np.zeros((m_2, 1))  # vector of zeros
c1 = np.ones((m_2, 1))   # vector of ones
b1 = np.copy(c0)
b2 = np.copy(c1)
for j in range(1, m+1):
    b1[m*j-1] = 1  # overwirte every m^th value with one
    b2[m*j-1] = 0  # overwirte every m^th value with one

b3 = np.zeros_like(b1)
b3[1:m_2] = b1[0:m_2-1]
b3[0] = b1[m_2-1]

b4 = np.zeros_like(b2)
b4[1:m_2] = b2[0:m_2-1]
b4[0] = b4[m_2-1]


diagonals_c = [b3.flatten(), -1 * b2.flatten(), b4.flatten(), -1 * b1.flatten()]
offsets_c = [-(m - 1), -1, 1, (m - 1)]
A3 = 1/(2 * dx) * spdiags(diagonals_c, offsets_c, m_2, m_2).toarray() #matrix C
plt.figure(7)
plt.spy(A3)
plt.title('Matrix Structure C')
plt.show()
print(A3[7, 0])