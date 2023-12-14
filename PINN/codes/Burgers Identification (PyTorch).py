import sys
sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from plotting import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('TkAgg')
import warnings
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings('ignore')

np.random.seed(1234)

# CUDA support
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')


# ## Physics-informed Neural Networks

# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1  # 隐藏层数（除了输入层和输出层）

        # set up layer order dict
        self.activation = torch.nn.Tanh  # 激活函数

        # layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):  # 正向传播，定义这个网络的流程 data
        out = self.layers(x)
        return out


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, u, layers, lb, ub):

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)  # 将一个 Numpy 数组 lb 转换为 PyTorch 张量，并将其存储在模型的属性 self.lb 中
        self.ub = torch.tensor(ub).float().to(device)

        # data (X,u)
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)  # 第一列取值
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)  # 第二列取值


        # true u
        self.u = torch.tensor(u).float().to(device)  # 下面定义了u

        # settings
        self.lambda_1 = torch.tensor([1.0], requires_grad=True).to(device)  # 标量转换为tenser
        self.lambda_2 = torch.tensor([-6.0], requires_grad=True).to(device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)  # 在模型里添加的参数lambda1和lambda2
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),  # 优化目标
            lr=1.0,  # 学习率
            max_iter=50000,  # 最大迭代次数
            max_eval=50000,  # 最大函数评估次数 max_eval
            history_size=50,  # 历史梯度存储大小
            tolerance_grad=1e-7,  # 梯度容差
            tolerance_change=1.0 * np.finfo(float).eps,  # 参数变化容差
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )  # 这些超参数可以根据具体问题进行调整

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

    def net_u(self, x, t):  # 控制初边界条件
        #predict u
        u = self.dnn(torch.cat([x, t], dim=1))  # self.dnn 表示当前实例（对象）的神经网络模型
        return u

    def net_f(self, x, t):  # 控制方程本身
        """ The pytorch autograd version of calculating residual """
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)

        # predict u
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f

    def loss_func(self):
        u_pred = self.net_u(self.x, self.t)  # true x ,t
        f_pred = self.net_f(self.x, self.t) # predict u

        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)  # 计算模型的预测值 u_pred 和真实值 self.u 之间的差距，
        # 并加上预测值 f_pred 的平方作为正则化项，最后返回总的损失值。

        self.optimizer.zero_grad()  # 优化器中的梯度缓存清零，

        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'iter:%d Loss: %e, l1: %.5f, l2: %.5f' %
                (   self.iter,
                    loss.item(),
                    self.lambda_1.item(),  # 表示将损失值转换为 Python 数值类型，
                    torch.exp(self.lambda_2.detach()).item()  # 为什么为指数形式
                )
            )
        return loss

    def train(self, nIter):

        self.dnn.train()
        for epoch in range(nIter):
            u_pred = self.net_u(self.x, self.t)
            f_pred = self.net_f(self.x, self.t)

            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)

            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            if epoch % 100 == 0:
                print(
                    'epoch: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' %
                    (
                        epoch,
                        loss.item(),
                        self.lambda_1.item(),
                        torch.exp(self.lambda_2).item()
                    )
                )

        # Backward and optimize
        self.optimizer.step(self.loss_func) #第一个

    # 经过训练的模型对新的输入数据(' X ')进行预测。它返回预测值(' u ')和残差(' f ')。
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


nu = 0.01 / np.pi

N_u = 2000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('../data/output.mat')


t = data['t'].flatten()[:, None]
# print('t',t.shape)
x = data['x'].flatten()[:, None]
# print('x',x)
Exact = np.real(data['usol']).T
# print('Exact',Exact.shape)
X, T = np.meshgrid(x, t)
# print('X',X,X.shape)
# print('T',T,T.shape)

print("X",X.shape)
print("T",T.shape)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

# Doman bounds
lb = X_star.min(0)  # 下界 x和t的下界
ub = X_star.max(0)  # 上界 x和t的上界

# ## Training on Non-noisy Data
noise = 0.0
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
# print(X_star.shape[0])
print("X_star shape:",X_star.shape)
print("idx:",idx)
X_u_train = X_star[idx, :]
u_train = u_star[idx, :]
# training
model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
print("------------------------------------no noise-----------------------------------")
model.train(0)


# evaluations
u_pred, f_pred = model.predict(X_star)

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

lambda_1_value = model.lambda_1.detach().cpu().numpy()
lambda_2_value = model.lambda_2.detach().cpu().numpy()
lambda_2_value = np.exp(lambda_2_value)

error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

print('Error u: %e' % (error_u))
print('Error l1: %.5f%%' % (error_lambda_1))
print('Error l2: %.5f%%' % (error_lambda_2))

# ## Training on Noisy Data

noise = 0.01
# create training set
u_train = u_train + noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])

# training
model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
print("------------------------------------noise-----------------------------------")
model.train(500)

"""========================================================================================="""

# """ The aesthetic setting has changed. """
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    'kx', label='Data (%d points)' % (u_train.shape[0]),
    markersize=4,  # marker size doubled
    clip_on=False,
    alpha=.5
)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize=20)  # font size doubled
ax.tick_params(labelsize=15)

plt.show()

""" The aesthetic setting has changed. """
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0.25$', fontsize=15)
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 0.50$', fontsize=15)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 0.75$', fontsize=15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()

# evaluations
u_pred, f_pred = model.predict(X_star)

lambda_1_value_noisy = model.lambda_1.detach().cpu().numpy()
lambda_2_value_noisy = model.lambda_2.detach().cpu().numpy()
lambda_2_value_noisy = np.exp(lambda_2_value_noisy)

error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu) / nu * 100

print('Error u: %e' % (error_u))
print('Error l1: %.5f%%' % (error_lambda_1_noisy))
print('Error l2: %.5f%%' % (error_lambda_2_noisy))

fig = plt.figure(figsize=(14, 10))

gs2 = gridspec.GridSpec(1, 3)
gs2.update(top=0.25, bottom=0, left=0.0, right=1.0, wspace=0.0)

ax = plt.subplot(gs2[:, :])
ax.axis('off')


# s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
# s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
# s3 = r'Identified PDE (1\% noise) & '
# s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
# s5 = r'\end{tabular}$'
# s = s1+s2+s3+s4+s5
# ax.text(0.1, 0.1, s, size=25)
plt.show()
