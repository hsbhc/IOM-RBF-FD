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


def alpha(points, c):
    A = PHI((points - points.T) ** 2, c)
    center = Variable(torch.full([3, 1], points[1][0]), requires_grad=True)
    phi = PHI((points - center) ** 2, c)
    p1 = torch.autograd.grad(phi, center, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    b = torch.autograd.grad(p1, center, grad_outputs=torch.ones_like(p1), create_graph=True)[0]
    return torch.mm(torch.inverse(A), b)


def epsilon(points, points_u, c):
    A = PHI((points - points.T) ** 2, c)
    lambda_i = torch.mm(torch.inverse(A), points_u)

    center = Variable(points[[1]], requires_grad=True)

    u = torch.mm(PHI((points - center) ** 2, c).T, lambda_i)
    f = F(center)

    ux = torch.autograd.grad(u, center, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, center, grad_outputs=torch.ones_like(ux), create_graph=True)[0]

    equation = uxx - f
    l_equation = torch.autograd.grad(equation, center, grad_outputs=torch.ones_like(equation), create_graph=True)[0]
    ll_equation = torch.autograd.grad(l_equation, center, grad_outputs=torch.ones_like(l_equation), create_graph=True)[
        0]
    return ll_equation


a = (5 * np.pi) / 4


def U(x):
    return 1 - torch.sin(a * x)


def F(x):
    return a ** 2 * torch.sin(a * x)


def train(nodes_num=10, is_GA_=True, d_=0.5, c_=2.0, epochs=100, lr=1):
    global d
    global is_GA

    h = 1 / nodes_num
    N = int(1 / h) + 1
    x = torch.linspace(0, 1, N)
    points = x[0:3].unsqueeze(-1)

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

    for s in range(epochs):
        A = torch.zeros([N - 2, N - 2])
        for i in range(N - 2):
            al = alpha(points=points, c=c).squeeze(-1)
            if i != 0:
                A[i][i - 1] = al[0]
            if i != N - 3:
                A[i][i + 1] = al[2]
            A[i][i] = al[1]

        b = F(x)[1:-1]
        al = alpha(points=points, c=c).squeeze(-1)
        b[0] = b[0] - U(x[0]) * al[0]
        al = alpha(points=points, c=c).squeeze(-1)
        b[-1] = b[-1] - U(x[-1]) * al[-1]

        u = torch.linalg.solve(A, b)

        current_u = torch.cat((Tensor([U(x[0])]), u, Tensor([U(x[-1])])), dim=0)

        lle = torch.tensor([[0.]])
        for current_i in range(1, N - 1):
            current_points = x[current_i - 1:current_i + 2].unsqueeze(-1)
            current_points_u = current_u[current_i - 1:current_i + 2].unsqueeze(-1)

            ll_equation = epsilon(current_points, current_points_u, c)
            lle = torch.cat((lle, ll_equation), dim=-1)

        los = torch.mm(torch.inverse(A), lle[:, 1:].T)
        loss = torch.nn.MSELoss()(los, torch.zeros_like(los))

        mserror_u = max(abs(U(x)[1:-1] - u))
        l2error_u = torch.linalg.norm(U(x)[1:-1] - u, 2) / torch.linalg.norm(U(x)[1:-1], 2)
        print(s, loss.item(), c.item(), mserror_u.item(), l2error_u.item())
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
    c_ = 2.0
    epochs = 1000
    lr = 0.1
    # test
    info = train(nodes_num=20, is_GA_=True, d_=0.5, c_=c_, epochs=epochs, lr=lr)
    exit(0)

    # 绘制某个节点数目的迭代图
    nodes = 20
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
    plt.savefig('1dpossion_loss.pdf')
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
    plt.savefig('1dpossion_c.pdf')
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
    plt.savefig('1dpossion_mserror.pdf')
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
    plt.savefig('1dpossion_l2error.pdf')
    plt.show()

    # # 记录基函数在不同节点个数时的性能
    # file = open('RBFFD_1dpossion_info.txt', 'w+')
    # nodes = [10, 20, 40, 60, 80]
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
    #
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
