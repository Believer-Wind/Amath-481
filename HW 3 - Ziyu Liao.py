import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_ivp, simpson
import math


# HW 3 - Part a
def shoot_eq(x, phi, epsilon):
    return [phi[1], (x**2 - epsilon) * phi[0]]


tol = 1e-6  # Tolerance for convergence
col = ['r', 'b', 'g', 'c', 'm']  # Color map for plotting
L = 4  # Range limit
dx = 0.1  # X step size
xshoot = np.arange(-L, L + dx, dx)  # X range with 81 points
epsilon_start = 0.1  # Initial value of epsilon

# The Result Vectors for eigenfunctions & eigenvalues
A1 = np.zeros((len(xshoot), 5))
A2 = np.zeros(5)

for modes in range(1, 6):
    epsilon = epsilon_start  # Start of epsilon
    depsilon = 0.2  # Step size of epsilon

    for _ in range(1000):
        # Solve ODE using solve_ivp
        Y0 = [1, np.sqrt(L**2 - epsilon)]
        sol = solve_ivp(shoot_eq,
                        [xshoot[0], xshoot[-1]],
                        Y0,
                        args=(epsilon,),
                        t_eval=xshoot
                        )

        # Check convergence
        if abs(sol.y[1, -1] + np.sqrt(L**2 - epsilon) * sol.y[0, -1]) < tol:
            break

        if ((-1) ** (modes + 1) * (sol.y[1, -1] + np.sqrt(L**2 - epsilon) * sol.y[0, -1])) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon /= 2

    A2[modes - 1] = epsilon
    epsilon_start = epsilon + 0.1  # Pick a new start
    norm = np.trapz(sol.y[0] * sol.y[0], xshoot)  # Normalization
    eigenfuction = abs(sol.y[0] / np.sqrt(norm))
    A1[:, modes - 1] = eigenfuction
    plt.plot(xshoot, eigenfuction, col[modes - 1])

print("A1:")
print(A1)
print("A2:")
print(A2)
# plt.show()

# HW 3 - Part b
L = 4
dx = 0.1
x = np.arange(-L, L + dx, dx)
N = len(x) - 2

B = np.zeros((N, N))
for j in range(N):
    B[j, j] = -2 - (x[j + 1] ** 2) * (dx ** 2)

for j in range(N - 1):
    B[j, j + 1] = 1
    B[j + 1, j] = 1

B1 = B

B2 = np.zeros((N, N))
B2[0, 0] = 4 / 3
B2[0, 1] = - 1 / 3

B3 = np.zeros((N, N))
B3[N - 1, N - 2] = - 1 / 3
B3[N - 1, N - 1] = 4 / 3

B = B1 + B2 + B3
B = B / (dx ** 2)

# Compute eigenvalues and eigenvectors
D, V = eigs(- B, k=5, which='SM')

# calculate the bc's
phi_0 = (4 / 3) * V[0, :] - (1 / 3) * V[1, :]
phi_n = - (1 / 3) * V[-2, :] + (4 / 3) * V[-1, :]

# append to the side
V = np.vstack((phi_0, V, phi_n))

for i in range(5):
    norm = np.trapz(V[:, i] ** 2, x)  # calculate the normalization
    V[:, i] = abs(V[:, i] / np.sqrt(norm))
    plt.plot(x, V[:, i])

plt.legend(
    ["$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", "$\\phi_5$"],
    loc="upper right"
    )

A3 = V
A4 = D

print("A3:")
print(A3)
print("A4:")
print(A4)
# plt.show()


# HW 3 - Part c

# Define differential equation
def shoot_eq(x, phi, epsilon, gamma):
    # return phi', phi''
    return [phi[1],
            (gamma * phi[0] ** 2 + x**2 - epsilon) * phi[0]]


# Parameters
tol = 1e-6
L = 2
dx = 0.1
xshoot = np.arange(-L, L + dx, dx)  # range of x values
gamma_values = [0.05, - 0.05]

# results for eigenfunctions & eigenvalues
A5, A7 = np.zeros((len(xshoot), 2)), np.zeros((len(xshoot), 2))
A6, A8 = np.zeros(2), np.zeros(2)

# Main for loop for 2 gammas
for gamma in gamma_values:
    epsilon_start = 0.1
    A = 1e-6

    # Two column results
    for modes in range(1, 3):
        dA = 0.01

        # Iterations to adjust A
        for ii in range(100):
            epsilon = epsilon_start
            depsilon = 0.2

            # Iterations to adjust epsilon
            for i in range(100):
                # initial conditions
                phi0 = [A, np.sqrt(L**2 - epsilon) * A]

                # Solve the ODE
                ans = solve_ivp(
                    lambda x, phi: shoot_eq(x, phi, epsilon, gamma),
                    [xshoot[0], xshoot[-1]],
                    phi0,
                    t_eval=xshoot
                    )
                phi_sol = ans.y.T
                x_sol = ans.t

                # Check boundary condition
                bc = phi_sol[-1, 1] + np.sqrt(L**2 - epsilon) * phi_sol[-1, 0]
                if abs(bc) < tol:
                    break

                # Adjust to steps of epsilon
                if (-1) ** (modes + 1) * bc > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2

            # Check whether it is focused
            integral = simpson(phi_sol[:, 0]**2, x=x_sol)
            if abs(integral - 1) < tol:
                break

            # Adjust to steps of A
            if integral < 1:
                A += dA
            else:
                A -= dA
                dA /= 2

        # Adjust to epsilon_start
        epsilon_start = epsilon + 0.2

        # Input results of eigenfuncitons & eigenvalues
        if gamma > 0:
            A5[:, modes - 1] = np.abs(phi_sol[:, 0])
            A6[modes - 1] = epsilon

        else:
            A7[:, modes - 1] = np.abs(phi_sol[:, 0])
            A8[modes - 1] = epsilon

plt.plot(xshoot, A5)
plt.plot(xshoot, A7)
plt.legend(["$\\phi_1$", "$\\phi_2$"], loc="upper right")
print("A5:")
print(A5)
print("A7:")
print(A7)
print("A6:")
print(A6)
print("A8:")
print(A8)
plt.show()


# HW 3 - Part d
def hwl_rhs_a(x, phi, epsilon):
    return [phi[1], (x ** 2 - epsilon) * phi[0]]


L = 2
x_span = [-L, L]
epsilon = 1
A = 1
phi0 = [A, np.sqrt(L ** 2 - epsilon) * A]
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

dt45, dt23, dtRadua, dtBDF = [], [], [], []

for tol in tols:
    options = {'rtol': tol, 'atol': tol}

    sol45 = solve_ivp(hwl_rhs_a, x_span, phi0, method='RK45', args=(epsilon,), **options)
    sol23 = solve_ivp(hwl_rhs_a, x_span, phi0, method='RK23', args=(epsilon,), **options)
    solRadua = solve_ivp(hwl_rhs_a, x_span, phi0, method='Radau', args=(epsilon,), **options)
    solBDF = solve_ivp(hwl_rhs_a, x_span, phi0, method='BDF', args=(epsilon,), **options)

    # calculate average time steps
    dt45.append(np.mean(np.diff(sol45.t)))
    dt23.append(np.mean(np.diff(sol23.t)))
    dtRadua.append(np.mean(np.diff(solRadua.t)))
    dtBDF.append(np.mean(np.diff(solBDF.t)))

# Perform linear regression (log - log) to determine slopes
fit45 = np.polyfit(np.log(dt45), np.log(tols), 1)
fit23 = np.polyfit(np.log(dt23), np.log(tols), 1)
fitRadua = np.polyfit(np.log(dtRadua), np.log(tols), 1)
fitBDF = np.polyfit(np.log(dtBDF), np.log(tols), 1)

# Extract slopes
slope45 = fit45[0]
slope23 = fit23[0]
slopeRadua = fitRadua[0]
slopeBDF = fitBDF[0]

A9 = np.array([slope45, slope23, slopeRadua, slopeBDF])

print("A9:")
print(A9)


# HW 3 - Part e
# Define first five Gauss-Hermite polynomial
def H0(x):
    return np.ones_like(x)


def H1(x):
    return 2 * x


def H2(x):
    return 4 * (x ** 2) - 2


def H3(x):
    return 8 * (x ** 3) - 12 * x


def H4(x):
    return 16 * (x ** 4) - 48 * (x ** 2) + 12


# Define x range
L = 4
dx = 0.1
x = np.arange(-L, L + dx, dx)

# Create matrix h for the exact Gauss-Hermite polynomial solutions（81 * 5 matrix）
h = np.column_stack([H0(x), H1(x), H2(x), H3(x), H4(x)])

# Create a zeros matrix has same shape with h
phi = np.zeros(h.shape)

# Normalize solutions and append in phi
for j in range(5):
    phi[:, j] = ((np.exp(- (x ** 2) / 2) * h[:, j]) /
                 np.sqrt(math.factorial(j) * (2 ** j) * np.sqrt(np.pi))
                 )

# Build up results matrix for comparison
erps_a = np.zeros(5)
erps_b = np.zeros(5)

er_a = np.zeros(5)
er_b = np.zeros(5)

# Compare eigenfunctions and eigenvalues
for j in range(5):
    erps_a[j] = simpson(((abs(A1[:, j])) - abs(phi[:, j])) ** 2, x=x)
    erps_b[j] = simpson(((abs(A3[:, j])) - abs(phi[:, j])) ** 2, x=x)

    er_a[j] = 100 * (abs(A2[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1))
    er_b[j] = 100 * (abs(A4[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1))

A10 = erps_a
A11 = er_a

A12 = erps_b
A13 = er_b

print("A10:")
print(A10)
print("A11:")
print(A11)

print("A12:")
print(A12)
print("A13:")
print(A13)
