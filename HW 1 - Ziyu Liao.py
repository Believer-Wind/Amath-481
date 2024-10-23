# HW 1 - Ziyu Liao
import numpy as np
import matplotlib.pyplot as plt

# Part 1
# Newton-Raphson method
x = np.array([-1.6])
number1 = 0

for j in range(1000):
    x = np.append(
        x, x[j] - (x[j] * np.sin(3 * x[j]) - np.exp(x[j])) /
        (np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j]))
    )
    fc = x[j] * np.sin(3 * x[j]) - np.exp(x[j])
    number1 += 1

    if abs(fc) < 1e-6:
        break

A1 = x

print("A1:", A1)
print()

# Bisection Method
dx = 0.1
x2 = np.arange(-10, 10+dx, dx)
y = x2 * np.sin(3 * x2) - np.exp(x2)
plt.plot(x2, y)
plt.axhline(0, color='red', linestyle='--')
plt.axis([-1, 1, -1, 1])
# plt.show() # the curve is downward

xr = - 0.4
xl = - 0.7
number2 = 0
A2 = np.array([])

for j in range(1000):
    xc = (xr + xl)/2
    A2 = np.append(A2, xc)
    fc = xc * np.sin(3 * xc) - np.exp(xc)

    if fc > 0:
        xl = xc
    else:
        xr = xc

    number2 += 1

    if abs(fc) < 1e-6:
        print("A2:", A2)
        break

A3 = np.array([], dtype=int)
A3 = np.append(A3, number1)
A3 = np.append(A3, number2)
print()
print("A3:", A3)
print()

# Part 2

A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

A4 = A + B
print("A4:", A4)
print()

A5 = 3 * x - 4 * y
print("A5:", A5)
print()

A6 = np.dot(A, x)
print("A6:", A6)
print()

A7 = np.dot(B, (x - y))
print("A7:", A7)
print()

A8 = np.dot(D, x)
print("A8:", A8)
print()

A9 = np.dot(D, y) + z
print("A9:", A9)
print()

A10 = np.dot(A, B)
print("A10:", A10)
print()

A11 = np.dot(B, C)
print("A11:", A11)
print()

A12 = np.dot(C, D)
print("A12:", A12)
