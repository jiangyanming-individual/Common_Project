"""
A scratch for PINN solving the following PDE
u_xx-u_yyyy=(2-x^2)*exp(-y)
Author: ST
Date: 2023/2/26
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from torch.autograd import Variable
import numpy as np
from matplotlib import cm
from matplotlib import cm
from scipy.special import gamma
from torch.distributions.beta import Beta
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.distributions.beta import Beta
# import differint.differint as df
import math
import sympy as sp
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



epochs = 9000    # 训练代数
h = 100    # 画图网格密度
N = 1000    # 内点配置点数
N1 = 200    # 边界点配置点数
N2 = 1000    # PDE数据点

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(12345)

# Domain and Sampling
def interior(n=N):
    # 内点
    x = torch.rand(n, 1)#n 行 1 列的随机张量 (tensor) x，张量元素的取值范围为 [0, 1) 之间的均匀分布
    y = torch.rand(n, 1)
    one=torch.ones_like(x)
    two=2*one
    cond = 0.7*torch.sin(torch.mul(x, torch.tensor(math.pi)))-torch.exp(-y)*(torch.sin(torch.mul(x, torch.tensor(math.pi)))-((math.pi)**2)*torch.sin(torch.mul(x, torch.tensor(math.pi)))) * torch.exp(x)#pde 等式右边
    #cond=(2*y**1.8/math.gamma(2.8))+2*x+0.8*x**2-two
    return x.requires_grad_(True), y.requires_grad_(True), cond


# def down_yy(n=N1):
#     # 边界 u_yy(x,0)=x^2
#     x = torch.rand(n, 1)
#     y = torch.zeros_like(x) #y=0
#     cond = x ** 2 #二阶导的初始条件 右边
#     return x.requires_grad_(True), y.requires_grad_(True), cond


# def up_yy(n=N1):
#     # 边界 u_yy(x,1)=x^2/e
#     x = torch.rand(n, 1)
#     y = torch.ones_like(x) #y=1
#     cond = x ** 2 / torch.e#二阶导的初始条件 右边
#     return x.requires_grad_(True), y.requires_grad_(True), cond


def down(n=N1):
    # 边界 u(x,0)=e^2
    x = torch.rand(n, 1)
    y = torch.zeros_like(x)#y=0
    #cond = torch.exp(x)  #初始条件 右边
    # cond=x**2
    cond=torch.sin(torch.mul(x, torch.tensor(math.pi)))
    return x.requires_grad_(True), y.requires_grad_(True), cond


# def up(n=N1):
#     # 边界 u(x,1)=x^2/e
#     x = torch.rand(n, 1)
#     y = torch.ones_like(x)#y=1
#     cond = x ** 2 / torch.e#初始条件 右边
#     return x.requires_grad_(True), y.requires_grad_(True), cond
#

def left(n=N1):
    # 边界 u(0,y)=y^2+1
    y = torch.rand(n, 1)
    x = torch.zeros_like(y)
    one = torch.ones_like(x)
    cond = x #x为边界条件，y为初始条件
    # cond=y**2
    return x.requires_grad_(True), y.requires_grad_(True), cond


def right(n=N1):
    # 边界 u(1,y)=e^(-y)
    y = torch.rand(n, 1)
    x = torch.ones_like(y)
    one = torch.ones_like(x)
    zero = torch.zeros_like(y)
    cond = zero
    # cond=one+y**2
    return x.requires_grad_(True), y.requires_grad_(True), cond

def data_interior(n=N2):
     # 内点
     x = torch.rand(n, 1)
     y = torch.rand(n, 1)
     cond = torch.exp(-y)*torch.sin(torch.mul(x, torch.tensor(math.pi)))#方程的真解
     return x.requires_grad_(True), y.requires_grad_(True), cond


# Neural Network
class MLP(torch.nn.Module):
    def __init__(self):

        super(MLP, self).__init__()
        self.register_parameter('lambda_1', nn.Parameter(torch.tensor([0.8]),requires_grad=True))
        self.register_parameter('lambda_2', nn.Parameter(torch.tensor([-6.0]),requires_grad=True))

        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),

            torch.nn.Linear(32, 1)
        )


    def forward(self, x):
        return self.net(x)


# Loss
loss = torch.nn.MSELoss()


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# 以下7个损失是PDE损失
def l_interior(u):
    # 损失函数L1
    # lambda_1 = self.lambda_1
    x, y, cond = interior()
    uxy = u(torch.cat([x, y], dim=1))#u(x,y)
    return loss(0.3*gradients(uxy, y, 1) -u.lambda_1 * gradients(uxy, x, 2)+0.7*uxy, cond)#PDE ：逗号前等式左边 都好后 等式右边
#0.2*gradients(uxy, y, 1) + gradients(uxy, x, 1)-gradients(uxy, x, 2)+0.8*uxy

# def l_down_yy(u):
#     # 损失函数L2
#     x, y, cond = down_yy()
#     uxy = u(torch.cat([x, y], dim=1))
#     return loss(gradients(uxy, y, 2), cond)


# def l_up_yy(u):
#     # 损失函数L3
#     x, y, cond = up_yy()
#     uxy = u(torch.cat([x, y], dim=1))
#     return loss(gradients(uxy, y, 2), cond)


def l_down(u):
    # 损失函数L4
    x, y, cond = down()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


# def l_up(u):
#     # 损失函数L5
#     x, y, cond = up()
#     uxy = u(torch.cat([x, y], dim=1))
#     return loss(uxy, cond)


def l_left(u):
    # 损失函数L6
    x, y, cond = left()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_right(u):
    # 损失函数L7
    x, y, cond = right()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)

#构造数据损失
def l_data(u):
    # 损失函数L8
    x, y, cond = data_interior()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


# Training

u = MLP()
opt = torch.optim.Adam(params=u.parameters(),lr=0.0001)

for i in range(epochs):
    opt.zero_grad()
    l = l_interior(u) \
        + 0.5*l_down(u) \
        + 0.5*l_left(u) \
        + 0.5*l_right(u) \
        +0.5*l_data(u)
    l.backward()
    opt.step()
    if i % 100 == 0:
        print("epoch is :",i,"loss is :",l.detach().cpu().numpy(), "lamda1:",u.lambda_1.detach().cpu().numpy())

# Inference
xc = torch.linspace(0, 1, h)#h表示网格密度
xm, ym = torch.meshgrid(xc, xc)#xm 和ym 的网格大小都为（h，h）
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
u_pred = u(xy)
one=torch.ones_like(yy)
u_real = torch.exp(-yy)*torch.sin(torch.mul(xx, torch.tensor(math.pi)))
u_error = torch.abs(u_pred-u_real)
u_pred_fig = u_pred.reshape(h,h)
u_real_fig = u_real.reshape(h,h)
u_error_fig = u_error.reshape(h,h)
error_sum=0
true_value_sum=0

for predict_value, true_value in zip(u_pred,u_real):

    # print(type(true_value))
    error_sum+=np.square(predict_value.detach().numpy() - true_value.detach().cpu().numpy())
    true_value_sum+=np.square(true_value.detach().cpu().numpy())

l2_normal_error=np.sqrt(error_sum) / np.sqrt(true_value_sum)
print("l2_normal_number :",l2_normal_error)


print("Min abs error is: ", float(torch.min(torch.abs(u_pred - (yy*yy*yy+one)*torch.exp(xx)))))
print("Max abs error is: ", float(torch.max(torch.abs(u_pred - (yy*yy*yy+one)*torch.exp(xx)))))
print("Max abs error is: ", float(torch.max(torch.max(torch.abs(u_pred - (yy*yy*yy+one)*torch.exp(xx))))/torch.max(torch.abs((yy*yy*yy+one)*torch.exp(xx)))))
# 仅有PDE损失    Max abs error:  0.004852950572967529
# 带有数据点损失  Max abs error:  0.0018916130065917969
#三维带边带的图
# pt_u = u(xy)
# u = pt_u.data.cpu().numpy()
# ms_u = u.reshape(yy.shape)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xc = torch.linspace(0, 1, h)#h表示网格密度
xm, ym = torch.meshgrid(xc, xc)#xm 和ym 的网格大小都为（h，h）
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
pt_u = u(xy)
u = pt_u.data.cpu().numpy()
ms_u = u.reshape(ym.shape)

surf = ax.plot_surface(xm, ym, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()















# 作PINN数值解图

fig = plt.figure()
#ax = Axes3D(fig)
ax =fig.add_subplot(projection='3d')
ax.invert_xaxis()
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_pred_fig.detach().numpy())
ax.text2D(0.5, 0.9, "PINN", transform=ax.transAxes)

fig.savefig("PINN solve.png")
plt.show()
# 作真解图
fig = plt.figure()
#ax = Axes3D(fig)
ax =fig.add_subplot(projection='3d')
ax.invert_xaxis()
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_real_fig.detach().numpy())
ax.text2D(0.5, 0.9, "real solve", transform=ax.transAxes)

fig.savefig("real solve.png")
plt.show()
# 误差图

fig = plt.figure()
#ax = Axes3D(fig)
ax =fig.add_subplot(projection='3d')
ax.invert_xaxis()
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_error_fig.detach().numpy())
ax.text2D(0.5, 0.9, "abs error", transform=ax.transAxes)

fig.savefig("abs error.png")
plt.show()


# x = np.arange(-1, 1, 0.02)
# t = np.arange(0, 1, 0.02)
# ms_x, ms_t = np.meshgrid(x, t)
# # Just because meshgrid is used, we need to do the following adjustment
# x = np.ravel(ms_x).reshape(-1, 1)
# t = np.ravel(ms_t).reshape(-1, 1)
#
# pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
# pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
# pt_u = u(pt_x, pt_t)
# u = pt_u.data.cpu().numpy()
# ms_u = u.reshape(ms_t.shape)
#
# surf = ax.plot_surface(ms_x, ms_t, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()
