# u_x - u_xx = f(x)
import numpy as np
from matplotlib import pyplot as plt


def U(x):
    return np.sin(np.pi * x) + (np.exp(x) - 1) / (np.e - 1)


def F(x):
    return np.pi ** 2 * np.sin(np.pi * x) + np.pi * np.cos(np.pi * x)


h = 1 / 80
N = int(1 / h) + 1
x = np.linspace(0, 1, N)

A = np.zeros([N - 2, N - 2])
for i in range(N - 2):
    if i != 0:
        A[i][i - 1] = - 1 / (2 * h) - 1 / h ** 2
    if i != N - 3:
        A[i][i + 1] = 1 / (2 * h) - 1 / h ** 2
    A[i][i] = 2 / h ** 2

b = F(x)[1:-1]
b[0] = b[0] - U(x[0]) * (- 1 / (2 * h) - 1 / h ** 2)
b[-1] = b[-1] - U(x[-1]) * (1 / (2 * h) - 1 / h ** 2)

u = np.linalg.solve(A, b)
max_abs_error_u = max(abs(U(x)[1:-1] - u))
print('FDM MSE error = %e' % max_abs_error_u)

# plt.title('FDM error = %e' % max_abs_error_u, fontsize=10)
# plt.plot(x[1:-1], u, '*', linewidth=1, label='Prediction')
# x = np.linspace(0, 1, 1001)
# plt.plot(x, U(x), 'r-', linewidth=1, label='Exact')
# plt.legend()
# plt.show()
