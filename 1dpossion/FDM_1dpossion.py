# u_xx = f(x)  #u(x) = 1 - sin(a x)
import numpy as np
from matplotlib import pyplot as plt

a = (5 * np.pi) / 4


def U(x):
    return 1 - np.sin(a * x)


def F(x):
    return a ** 2 * np.sin(a * x)


h = 1 / 80
N = int(1 / h) + 1
x = np.linspace(0, 1, N)

A = np.zeros([N - 2, N - 2])
for i in range(N - 2):
    if i != 0:
        A[i][i - 1] = 1 / h**2
    if i != N - 3:
        A[i][i + 1] = 1 / h**2
    A[i][i] = -2 / h**2

b = F(x)[1:-1]
b[0] = b[0] - U(x[0]) / h ** 2
b[-1] = b[-1] - U(x[-1]) / h ** 2

u = np.linalg.solve(A, b)
max_abs_error_u = max(abs(U(x)[1:-1] - u))
print('FDM MSE error = %e' % max_abs_error_u)

# plt.title('FDM MAE error = %e' % max_abs_error_u, fontsize=10)
# plt.plot(x[1:-1], u, '*', linewidth=1, label='Prediction')
# x = np.linspace(0, 1, 1001)
# plt.plot(x, U(x), 'r-', linewidth=1, label='Exact')
# plt.legend()
# plt.show()
