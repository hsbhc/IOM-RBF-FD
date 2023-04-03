from matplotlib import pyplot as plt
from numpy import *
import numpy as np


def U_E(x, y):
    return sin(pi * x) * cos(2 * pi * y) * exp(-(x - 1 / 4) ** 2 - (y - 1 / 2) ** 2)


def F_E(x, y):
    return (4 * (x - 0.25) ** 2 * sin(pi * x) * cos(2 * pi * y) -
            2 * pi * (2 * x - 0.5) * cos(pi * x) * cos(2 * pi * y) +
            4 * (y - 0.5) ** 2 * sin(pi * x) * cos(2 * pi * y) +
            4 * pi * (2 * y - 1.0) * sin(pi * x) * sin(2 * pi * y) -
            5 * pi ** 2 * sin(pi * x) * cos(2 * pi * y) -
            4 * sin(pi * x) * cos(2 * pi * y)) * exp(-(x - 0.25) ** 2 - (y - 0.5) ** 2)


M, N = 10, 10
a, b = 1, 1
hx, hy = a / M, b / N
p, q = 1 / hx ** 2, 1 / hy ** 2
r = -2 * (p + q)

x1 = np.expand_dims(np.linspace(0, a, M + 1), axis=1)
x2 = np.expand_dims(np.linspace(0, b, N + 1), axis=1)
X1, X2 = np.meshgrid(x1, x2)

x_test_np = np.concatenate((np.vstack(np.expand_dims(X1, axis=2)), np.vstack(np.expand_dims(X2, axis=2))), axis=-1)
solution = np.reshape(U_E(x_test_np[:, [0]], x_test_np[:, [1]]), (X1.shape[0], X2.shape[1]))
F_solution = np.reshape(F_E(x_test_np[:, [0]], x_test_np[:, [1]]), (X1.shape[0], X2.shape[1]))

# plt.pcolor(X1, X2, solution, shading='auto', cmap='jet')
# plt.colorbar()
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.title('U')
# plt.tight_layout()
# plt.show()

U = zeros((M - 1, M - 1))
for i in range(M - 1):
    U[i, i] = r
    if i < M - 2: U[i, i + 1] = p
    if i > 0:   U[i, i - 1] = p
V = diag([q] * (M - 1))
Zero_mat = zeros((M - 1, M - 1))

A_blc = empty((N - 1, N - 1), dtype=object)
for i in range(N - 1):
    for j in range(N - 1):
        if i == j:
            A_blc[i, j] = U
        elif abs(i - j) == 1:
            A_blc[i, j] = V
        else:
            A_blc[i, j] = Zero_mat

A = vstack([hstack(A_i) for A_i in A_blc])

x_i = linspace(0, a, M + 1)
y_i = linspace(0, b, N + 1)
F = F_solution[1:M, 1:N]

F[[0], :] = F[[0], :] - q * solution[[0], 1:M]
F[[-1], :] = F[[-1], :] - q * solution[[-1], 1:M]
F[:, [0]] = F[:, [0]] - p * solution[1:N, [0]]
F[:, [-1]] = F[:, [-1]] - p * solution[1:N, [-1]]

F = np.reshape(F, ((M - 1) * (N - 1), 1))
u = dot(linalg.inv(A), F).reshape(M - 1, N - 1)
u_f = vstack([solution[[0], :],
              hstack([solution[1:N, [0]], u, solution[1:N, [-1]]]),
              solution[[-1], :]])

# plt.pcolor(X1, X2, u_f, shading='auto', cmap='jet')
# plt.colorbar()
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.title('U')
# plt.tight_layout()
# plt.show()

max_abs_error_u = np.max(abs(u_f - solution))
print('FDM MSE error = %e' % max_abs_error_u)

