import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags
from scipy.linalg import lu, solve_triangular
from scipy.sparse.linalg import gmres, bicgstab 

# ---------------------- Part a ----------------------
# Define parameters
tspan = np.arange(0, 4.5, 0.5)  # time range, step size 0.5
nu = 0.001  # Viscosity coefficient
Lx, Ly = 20, 20  # Define space domain 
nx, ny = 64, 64  # Define number of points
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx / 2, Lx / 2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly / 2, Ly / 2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)
w0 = np.exp(-X ** 2 - Y ** 2 / 20).flatten()  # Initial vorticity

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx / 2), np.arange(-nx / 2, 0)))
kx[0] = 1e-6  # Avoid zero problem
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny / 2), np.arange(-ny / 2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX ** 2 + KY ** 2

# Equalize variables
L = Lx  # L = 20
n = N   # n = 64 * 64
m = nx  # m = 64
delta = L / m  # Distance

# Construct initial vectors
e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))  # vector of ones
e2 = np.copy(e1)  # copy the one vector
e4 = np.copy(e0)  # copy the zero vector

# ====== Construct Martrix A ======
for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements of A
diagonals_A = [e1.flatten(), e1.flatten(), e5.flatten(),
               e2.flatten(), -4 * e1.flatten(), e3.flatten(),
               e4.flatten(), e1.flatten(), e1.flatten()]
offsets_A = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]

A = spdiags(diagonals_A, offsets_A, n, n) / (delta ** 2)

# ====== Contruct Matrix B ======
# Place diagonal elements of B
diagonals_B = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]
B = spdiags(diagonals_B, offsets_B, n, n) / (2 * delta)

# ====== Contruct Matrix C ======
# Adjust e1
for i in range(1, n):
    e1[i] = e4[i - 1]

# Place diagonal elements of C
diagonals_C = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m + 1, -1, 1,  m - 1]
C = spdiags(diagonals_C, offsets_C, n, n) / (2 * delta)

# ====== Convert Matrix A, B, C into a Dense Matrix ======
A = A.toarray()
B = B.toarray()
C = C.toarray()

A[0, 0] = 2 / (delta ** 2)

start_time1 = time.time()
def spc_rhs(t, w0, nx, ny, N, A, B, C, K, nu):
    w = w0.reshape((nx, ny))
    wt = fft2(w)
    psit = - wt / K
    psi = np.real(ifft2(psit)).reshape(N)
    rhs = (nu * np.dot(A, w0)
           - np.dot(B, psi) * np.dot(C, w0)
           + np.dot(B, w0) * np.dot(C, psi)
           )
    return rhs


end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1

w_sol = solve_ivp(spc_rhs, [0, 4], w0, t_eval=tspan,
                  args=(nx, ny, N, A, B, C, K, nu), method="RK45")
A1 = w_sol.y
print('A1: \n', A1)
print()

# ---------------------- Part b ----------------------
# Method A/b
def GE_rhs(t, w0, nx, ny, N, A, B, C, K, nu):
    psi = np.linalg.solve(A, w0)
    rhs = (nu * np.dot(A, w0)
           - np.dot(B, psi) * np.dot(C, w0)
           + np.dot(B, w0) * np.dot(C, psi)
           )
    return rhs


start_time2 = time.time()
w_sol2 = solve_ivp(GE_rhs, [0, 4], w0, t_eval=tspan,
                   args=(nx, ny, N, A, B, C, K, nu), method="RK45")
end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2
A2 = w_sol2.y
print('A2: \n', A2)
print()

# Method LU decomposition
def LU_rhs(t, w0, nx, ny, N, A, B, C, K, nu):
    Pb = np.dot(P, w0)
    y = solve_triangular(L, Pb, lower=True)
    psi = solve_triangular(U, y)
    rhs = (nu * np.dot(A, w0)
           - np.dot(B, psi) * np.dot(C, w0)
           + np.dot(B, w0) * np.dot(C, psi)
           )
    return rhs


P, L, U = lu(A)

start_time3 = time.time()
w_sol3 = solve_ivp(LU_rhs, [0, 4], w0, t_eval=tspan,
                   args=(nx, ny, N, A, B, C, K, nu), method="RK45")
end_time3 = time.time()
elapsed_time3 = end_time3 - start_time3
A3 = w_sol3.y
print('A3: \n', A3)
print()

# Method BICGSTAB
def BICGSTAB_rhs(t, w0, nx, ny, N, A, B, C, K, nu):
    psi, info = bicgstab(A, w0, rtol=1e-6)
    rhs = (nu * np.dot(A, w0)
           - np.dot(B, psi) * np.dot(C, w0)
           + np.dot(B, w0) * np.dot(C, psi)
           )
    return rhs


start_time4 = time.time()
w_sol4 = solve_ivp(BICGSTAB_rhs, [0, 4], w0, t_eval=tspan,
                   args=(nx, ny, N, A, B, C, K, nu), method="RK45")
end_time4 = time.time()
elapsed_time4 = end_time4 - start_time4
A4 = w_sol4.y
print('A4 (BICGSTAB): \n', A4)
print()

def GMRES_rhs(t, w0, nx, ny, N, A, B, C, K, nu):
    psi, info = gmres(A, w0, rtol=1e-6, maxiter=1000)
    rhs = (nu * np.dot(A, w0)
           - np.dot(B, psi) * np.dot(C, w0)
           + np.dot(B, w0) * np.dot(C, psi)
           )
    return rhs


start_time5 = time.time()
w_sol5 = solve_ivp(GMRES_rhs, [0, 4], w0, t_eval=tspan,
                   args=(nx, ny, N, A, B, C, K, nu), method="RK45")
end_time5 = time.time()
elapsed_time5 = end_time5 - start_time5
A5 = w_sol5.y
print('A5 (GMRES): \n', A5)
print()

# ---------------------- Comparison of Methods ----------------------
print(f"FFT method elapsed time: {elapsed_time1:.6f} seconds")
print(f"A/b method elapsed time: {elapsed_time2:.6f} seconds")
print(f"LU method elapsed time: {elapsed_time3:.6f} seconds")
print(f"BICGSTAB method elapsed time: {elapsed_time4:.6f} seconds")
print(f"GMRES method elapsed time: {elapsed_time5:.6f} seconds")
print()


# ---------------------- Part c ----------------------
# Experiment with different initial conditions using the fastest solver
def initialize_vortices(X, Y, case='positive_negative_pair'):
    if case == 'positive_negative_pair':
        w0 = np.exp(-((X + 3) ** 2 + Y ** 2)) - np.exp(-((X - 3) ** 2 + Y ** 2))
    elif case == 'positive_pair':
        w0 = np.exp(-((X + 3) ** 2 + Y ** 2)) + np.exp(-((X - 3) ** 2 + Y ** 2))
    elif case == 'colliding_pairs':
        w0 = (np.exp(-((X + 4) ** 2 + (Y + 4) ** 2)) -
              np.exp(-((X - 4) ** 2 + (Y + 4) ** 2)) +
              np.exp(-((X + 4) ** 2 + (Y - 4) ** 2)) -
              np.exp(-((X - 4) ** 2 + (Y - 4) ** 2)))
    elif case == 'random_vortices':
        np.random.seed(42)
        w0 = np.zeros_like(X)
        for _ in range(15):
            x0, y0 = np.random.uniform(-10, 10, 2)
            strength = np.random.uniform(-1, 1)
            ellipticity = np.random.uniform(0.5, 2.0)
            w0 += strength * np.exp(-((X - x0) ** 2 + ellipticity * (Y - y0) ** 2))
    return w0.flatten()


# Solve for each case
cases = ['positive_negative_pair', 'positive_pair', 'colliding_pairs', 'random_vortices']
solutions = {}
for case in cases:
    w0 = initialize_vortices(X, Y, case=case)
    w_sol_case = solve_ivp(spc_rhs, [0, 4], w0, t_eval=tspan,
                           args=(nx, ny, N, A, B, C, K, nu), method="RK45")
    solutions[case] = w_sol_case.y
    print(f'Solution for {case}: \n', w_sol_case.y)


# ---------------------- Part d ----------------------
# Create a 2-D movie of the dynamics
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(solutions['random_vortices'][:, 0].reshape((nx, ny)), extent=[-10, 10, -10, 10], cmap='plasma')
fig.colorbar(cax, ax=ax, label='Vorticity')
ax.set_title('Vorticity Field')
ax.set_xlabel('x')
ax.set_ylabel('y')


# Update function for the animation
def update(frame):
    ax.set_title(f'Vorticity Field at t = {tspan[frame]:.2f}')
    cax.set_data(solutions['random_vortices'][:, frame].reshape((nx, ny)))
    return cax,


ani = animation.FuncAnimation(fig, update, frames=solutions['random_vortices'].shape[1], blit=True)
plt.show()
