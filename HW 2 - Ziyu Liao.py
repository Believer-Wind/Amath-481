import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def shoot_eq(phi, x, epsilon):
    return [phi[1], (x**2 - epsilon) * phi[0]]


tol = 1e-6  # Tolerance for convergence
col = ['r', 'b', 'g', 'c', 'm']  # Color map for plotting
L = 4  # Range limit
dx = 0.1  # X step size
xshoot = np.arange(-L, L + dx, dx)  # X range with 81 points
epsilon_start = 0.1  # Initial value of epsilon

# The Result Vectors for eigenfunctions $ eigenvalues
A1 = np.zeros((len(xshoot), 5))
A2 = np.zeros(5)

for modes in range(1, 6):
    epsilon = epsilon_start  # Start of epsilon
    depsilon = 0.2  # Step size of epsilon

    for _ in range(1000):
        # Solve ODE
        Y0 = [1, np.sqrt(L**2 - epsilon)]
        y = odeint(shoot_eq, Y0, xshoot, args=(epsilon, ))

        # Check convergence
        if abs(y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) < tol:
            break

        if ((-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0])) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon /= 2

    A2[modes - 1] = epsilon
    epsilon_start = epsilon + 0.1  # Pick a new start
    norm = np.trapz(y[:, 0] * y[:, 0], xshoot)  # Normalization
    eigenfuction = abs(y[:, 0] / np.sqrt(norm))
    A1[:, modes - 1] = eigenfuction
    plt.plot(xshoot, eigenfuction, col[modes - 1])

print(A1)
print(A2)
plt.show()
