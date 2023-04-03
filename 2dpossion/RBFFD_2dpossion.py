import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim, Tensor
from torch.autograd import Variable

torch.set_default_dtype(torch.double)


def PHI(dis, c):
    if is_GA:
        return torch.exp(-c ** 2 * dis)
    return (c ** 2 + dis) ** d


def alpha(xy, c):
    points = xy
    x = points[:, [0]]
    y = points[:, [1]]
    A = PHI((x - x.T).T ** 2 + (y - y.T).T ** 2, c)
    center = torch.zeros([5, 2])
    for i in range(5):
        center[i] = points[2]
    center = Variable(center, requires_grad=True)
    phi = PHI((x - center[:, [0]]) ** 2 + (y - center[:, [1]]) ** 2, c)
    p1 = torch.autograd.grad(phi, center, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    phi_x = p1[:, [0]]
    phi_y = p1[:, [1]]
    phi_xx = torch.autograd.grad(phi_x, center, grad_outputs=torch.ones_like(phi_x), create_graph=True)[0][:, [0]]
    phi_yy = torch.autograd.grad(phi_y, center, grad_outputs=torch.ones_like(phi_y), create_graph=True)[0][:, [1]]
    b = phi_xx + phi_yy

    return torch.mm(torch.inverse(A), b)


def epsilon(points, points_u, c):
    x = points[:, [0]]
    y = points[:, [1]]
    A = PHI((x - x.T).T ** 2 + (y - y.T).T ** 2, c)
    center = torch.zeros([5, 2])
    for i in range(5):
        center[i] = points[2]
    center = Variable(center, requires_grad=True)
    alp = torch.mm(torch.inverse(A.T), points_u).T

    phi = PHI((x - center[:, [0]]) ** 2 + (y - center[:, [1]]) ** 2, c)
    p1 = torch.autograd.grad(phi, center, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    phi_x = p1[:, [0]]
    phi_y = p1[:, [1]]
    phi_xx = torch.autograd.grad(phi_x, center, grad_outputs=torch.ones_like(phi_x), create_graph=True)[0][:, [0]]
    phi_xxx = torch.autograd.grad(phi_xx, center, grad_outputs=torch.ones_like(phi_xx), create_graph=True)[0][:, [0]]
    phi_xxxx = torch.autograd.grad(phi_xxx, center, grad_outputs=torch.ones_like(phi_xxx), create_graph=True)[0][:, [0]]

    phi_yy = torch.autograd.grad(phi_y, center, grad_outputs=torch.ones_like(phi_y), create_graph=True)[0][:, [1]]
    phi_yyy = torch.autograd.grad(phi_yy, center, grad_outputs=torch.ones_like(phi_yy), create_graph=True)[0][:, [1]]
    phi_yyyy = torch.autograd.grad(phi_yyy, center, grad_outputs=torch.ones_like(phi_yyy), create_graph=True)[0][:, [1]]

    phi_xxy = torch.autograd.grad(phi_xx, center, grad_outputs=torch.ones_like(phi_xx), create_graph=True)[0][:, [1]]
    phi_xxyy = torch.autograd.grad(phi_xxy, center, grad_outputs=torch.ones_like(phi_xxy), create_graph=True)[0][:, [1]]

    phi_yyx = torch.autograd.grad(phi_yy, center, grad_outputs=torch.ones_like(phi_yy), create_graph=True)[0][:, [0]]
    phi_yyxx = torch.autograd.grad(phi_yyx, center, grad_outputs=torch.ones_like(phi_yyx), create_graph=True)[0][:, [0]]

    left = torch.mm(alp, phi_xxxx + phi_yyyy + 2 * phi_xxyy)

    center1 = Variable(points[[2]], requires_grad=True)
    f = F(center1[:, [0]], center1[:, [1]])
    f_p1 = torch.autograd.grad(f, center1, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    f_x = f_p1[:, [0]]
    f_y = f_p1[:, [1]]

    f_xx = torch.autograd.grad(f_x, center1, grad_outputs=torch.ones_like(f_x), create_graph=True)[0][:, [0]]
    f_yy = torch.autograd.grad(f_y, center1, grad_outputs=torch.ones_like(f_y), create_graph=True)[0][:, [1]]

    return f_xx + f_yy - left


def U(x, y):
    return torch.sin(torch.pi * x) * torch.cos(2 * torch.pi * y) * torch.exp(-(x - 1 / 4) ** 2 - (y - 1 / 2) ** 2)


def F(x, y):
    return (4 * (x - 0.25) ** 2 * torch.sin(torch.pi * x) * torch.cos(2 * torch.pi * y) -
            2 * torch.pi * (2 * x - 0.5) * torch.cos(torch.pi * x) * torch.cos(2 * torch.pi * y) +
            4 * (y - 0.5) ** 2 * torch.sin(torch.pi * x) * torch.cos(2 * torch.pi * y) +
            4 * torch.pi * (2 * y - 1.0) * torch.sin(torch.pi * x) * torch.sin(2 * torch.pi * y) -
            5 * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.cos(2 * torch.pi * y) -
            4 * torch.sin(torch.pi * x) * torch.cos(2 * torch.pi * y)) * torch.exp(-(x - 0.25) ** 2 - (y - 0.5) ** 2)


def train(nodes_num=10, is_GA_=True, d_=0.5, c_=2.0, epochs=100, lr=1):
    global d
    global is_GA

    h = 1 / nodes_num
    N = int(1 / h) + 1
    x1 = np.expand_dims(np.linspace(0, 1, N), axis=1)
    x2 = np.expand_dims(np.linspace(0, 1, N), axis=1)
    X1, X2 = np.meshgrid(x1, x2)
    x_test_np = np.concatenate((np.vstack(np.expand_dims(X1, axis=2)), np.vstack(np.expand_dims(X2, axis=2))), axis=-1)
    x_test_tensor = torch.from_numpy(x_test_np)
    solution = np.reshape(U(x_test_tensor[:, [0]], x_test_tensor[:, [1]]).numpy(), (X1.shape[0], X2.shape[1]))
    F_solution = np.reshape(F(x_test_tensor[:, [0]], x_test_tensor[:, [1]]).numpy(), (X1.shape[0], X2.shape[1]))

    x = torch.linspace(0, 1, N)
    y = torch.linspace(0, 1, N)
    xy = torch.zeros([5, 2])
    xy[0] = torch.tensor([x[0], y[1]])
    xy[1] = torch.tensor([x[2], y[1]])
    xy[2] = torch.tensor([x[1], y[1]])
    xy[3] = torch.tensor([x[1], y[0]])
    xy[4] = torch.tensor([x[1], y[2]])

    points = xy
    c = nn.Parameter(torch.tensor(c_), requires_grad=True)
    d = d_
    is_GA = is_GA_

    epochs = epochs
    optimizer = optim.Adagrad([c], lr=lr)

    iters = []
    loss_all = []
    c_all = []
    mserror_all = []
    l2error_all = []

    def solve(p, q, r):
        M = N - 1
        U = np.zeros((M - 1, M - 1))
        for i in range(M - 1):
            U[i, i] = r
            if i < M - 2: U[i, i + 1] = p
            if i > 0:   U[i, i - 1] = p
        V = np.diag([q] * (M - 1))
        Zero_mat = np.zeros((M - 1, M - 1))

        A_blc = np.empty((M - 1, M - 1), dtype=object)
        for i in range(M - 1):
            for j in range(M - 1):
                if i == j:
                    A_blc[i, j] = U
                elif abs(i - j) == 1:
                    A_blc[i, j] = V
                else:
                    A_blc[i, j] = Zero_mat

        A = np.vstack([np.hstack(A_i) for A_i in A_blc])

        F = F_solution.copy()[1:M, 1:M]
        F[0, :] = F[0, :] - q * solution[0, 1:M]
        F[-1, :] = F[-1, :] - q * solution[-1, 1:M]
        F[:, 0] = F[:, 0] - p * solution[1:M, 0]
        F[:, -1] = F[:, -1] - p * solution[1:M, -1]
        F = np.reshape(F, ((M - 1) * (M - 1), 1))

        u = np.dot(np.linalg.inv(A), F).reshape(M - 1, M - 1)
        u_f = np.vstack([solution[[0], :],
                         np.hstack([solution[1:M, [0]], u, solution[1:M, [-1]]]),
                         solution[[-1], :]])

        return u_f, A

    for s in range(epochs):
        al = alpha(points, c=c).squeeze(-1)
        current_u, A = solve(al[0].item(), al[1].item(), al[2].item())
        A = torch.from_numpy(A)

        # mserror_u = np.max(abs(current_u - solution))
        # l2error_u = np.linalg.norm(current_u - solution, 2) / np.linalg.norm(solution, 2)
        # print('%.2e %.2e'%(mserror_u,l2error_u))
        # exit(0)

        lle = torch.tensor([[0.]])
        for current_i in range(1, N - 1):
            for current_j in range(1, N - 1):
                xy = torch.zeros([5, 2])
                xy[0] = torch.tensor([x[current_i - 1], y[current_j]])
                xy[1] = torch.tensor([x[current_i + 1], y[current_j]])
                xy[2] = torch.tensor([x[current_i], y[current_j]])
                xy[3] = torch.tensor([x[current_i], y[current_j - 1]])
                xy[4] = torch.tensor([x[current_i], y[current_j + 1]])

                u = torch.zeros([5, 1])
                u[0] = torch.tensor([current_u[current_i][current_j - 1]])
                u[1] = torch.tensor([current_u[current_i][current_j + 1]])
                u[2] = torch.tensor([current_u[current_i][current_j]])
                u[3] = torch.tensor([current_u[current_i - 1][current_j]])
                u[4] = torch.tensor([current_u[current_i + 1][current_j]])

                ll_equation = epsilon(xy, u, c)
                lle = torch.cat((lle, ll_equation), dim=-1)

        los = torch.mm(torch.inverse(A), lle[:, 1:].T)
        loss = torch.nn.MSELoss()(los, torch.zeros_like(los))

        mserror_u = np.max(abs(current_u - solution))
        l2error_u = np.linalg.norm(current_u - solution, 2) / np.linalg.norm(solution, 2)
        print(s, loss.item(), c.item(), mserror_u, l2error_u)
        iters.append(s)
        loss_all.append(loss.item())
        c_all.append(c.item())
        mserror_all.append(mserror_u.item())
        l2error_all.append(l2error_u.item())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    return np.array([iters, loss_all, c_all, mserror_all, l2error_all])


def write_table(file, name, table):
    file.write(name + '\n')
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if i == 0:
                temp = str(int(table[i, j] + 1))
            elif i == 1:
                temp = '%.4f' % (table[i, j])
            else:
                temp = '%.2e' % (table[i, j])

            if j == table.shape[1] - 1:
                file.write(temp + '\n')
            else:
                file.write(temp + ' & ')

    file.write('\n\n\n')


if __name__ == '__main__':
    c_ = 1.5
    epochs = 50
    lr = 0.1
    # test
    info = train(nodes_num=10, is_GA_=True, d_=0.5, c_=c_, epochs=epochs, lr=lr)
    exit(0)

    # 绘制某个节点数目的迭代图
    nodes = 10
    GA_info = train(nodes_num=nodes, is_GA_=True, d_=0.5, c_=c_, epochs=epochs, lr=lr)
    MQ_info = train(nodes_num=nodes, is_GA_=False, d_=0.5, c_=c_, epochs=epochs, lr=lr)
    IQ_info = train(nodes_num=nodes, is_GA_=False, d_=-1, c_=c_, epochs=epochs, lr=lr)
    IMQ_info = train(nodes_num=nodes, is_GA_=False, d_=-0.5, c_=c_, epochs=epochs, lr=lr)

    plt.tick_params(labelsize=15)
    plt.rc('legend', fontsize=15)
    plt.yscale('log')
    plt.plot(GA_info[0, :], GA_info[1, :], 'k-', label='GA')
    plt.plot(MQ_info[0, :], MQ_info[1, :], 'b-', label='MQ')
    plt.plot(IQ_info[0, :], IQ_info[1, :], 'y-', label='IQ')
    plt.plot(IMQ_info[0, :], IMQ_info[1, :], 'r-', label='IMQ')
    plt.xlabel('$iters$', fontsize=15)
    plt.ylabel('$loss$', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('2dpossion_loss.pdf')
    plt.show()

    plt.tick_params(labelsize=15)
    plt.rc('legend', fontsize=15)
    plt.plot(GA_info[0, :], GA_info[2, :], 'k-', label='GA')
    plt.plot(MQ_info[0, :], MQ_info[2, :], 'b-', label='MQ')
    plt.plot(IQ_info[0, :], IQ_info[2, :], 'y-', label='IQ')
    plt.plot(IMQ_info[0, :], IMQ_info[2, :], 'r-', label='IMQ')
    plt.xlabel('$iters$', fontsize=15)
    plt.ylabel('$c$', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('2dpossion_c.pdf')
    plt.show()

    plt.tick_params(labelsize=15)
    plt.rc('legend', fontsize=15)
    plt.yscale('log')
    plt.plot(GA_info[0, :], GA_info[3, :], 'k-', label='GA')
    plt.plot(MQ_info[0, :], MQ_info[3, :], 'b-', label='MQ')
    plt.plot(IQ_info[0, :], IQ_info[3, :], 'y-', label='IQ')
    plt.plot(IMQ_info[0, :], IMQ_info[3, :], 'r-', label='IMQ')
    plt.xlabel('$iters$', fontsize=15)
    plt.ylabel('$||E(c^{opt})||_{\infty}$', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('2dpossion_mserror.pdf')
    plt.show()

    plt.tick_params(labelsize=15)
    plt.rc('legend', fontsize=15)
    plt.yscale('log')
    plt.plot(GA_info[0, :], GA_info[4, :], 'k-', label='GA')
    plt.plot(MQ_info[0, :], MQ_info[4, :], 'b-', label='MQ')
    plt.plot(IQ_info[0, :], IQ_info[4, :], 'y-', label='IQ')
    plt.plot(IMQ_info[0, :], IMQ_info[4, :], 'r-', label='IMQ')
    plt.xlabel('$iters$', fontsize=15)
    plt.ylabel('$||E(c^{opt})||_{L2}$', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('2dpossion_l2error.pdf')
    plt.show()

    # 记录基函数在不同节点个数时的性能
    # file = open('RBFFD_2dpossion_info.txt', 'w+')
    # nodes = [5, 10, 20, 30]
    #
    # # GA
    # GA_table = np.zeros([4, len(nodes)])
    # for i in range(len(nodes)):
    #     info = train(nodes_num=nodes[i], is_GA_=True, d_=0.5, c_=c_, epochs=epochs, lr=lr)
    #     GA_table[0, i] = nodes[i]
    #     GA_table[1, i] = info[2, -1]
    #     GA_table[2, i] = info[3, -1]
    #     GA_table[3, i] = info[4, -1]
    #
    # write_table(file, 'GA', GA_table)

    # # MQ
    # MQ_table = np.zeros([4, len(nodes)])
    # for i in range(len(nodes)):
    #     info = train(nodes_num=nodes[i], is_GA_=False, d_=0.5, c_=c_, epochs=epochs, lr=lr)
    #     MQ_table[0, i] = nodes[i]
    #     MQ_table[1, i] = info[2, -1]
    #     MQ_table[2, i] = info[3, -1]
    #     MQ_table[3, i] = info[4, -1]
    #
    # write_table(file, 'MQ', MQ_table)
    #
    # # IQ
    # IQ_table = np.zeros([4, len(nodes)])
    # for i in range(len(nodes)):
    #     info = train(nodes_num=nodes[i], is_GA_=False, d_=-1, c_=c_, epochs=epochs, lr=lr)
    #     IQ_table[0, i] = nodes[i]
    #     IQ_table[1, i] = info[2, -1]
    #     IQ_table[2, i] = info[3, -1]
    #     IQ_table[3, i] = info[4, -1]
    #
    # write_table(file, 'IQ', IQ_table)
    #
    # # IMQ
    # IMQ_table = np.zeros([4, len(nodes)])
    # for i in range(len(nodes)):
    #     info = train(nodes_num=nodes[i], is_GA_=False, d_=- 0.5, c_=c_, epochs=epochs, lr=lr)
    #     IMQ_table[0, i] = nodes[i]
    #     IMQ_table[1, i] = info[2, -1]
    #     IMQ_table[2, i] = info[3, -1]
    #     IMQ_table[3, i] = info[4, -1]
    #
    # write_table(file, 'IMQ', IMQ_table)
