import numpy as np
N,D_in,H,D_out=64,1000,100,10
x=np.random.randn(N,D_in)
y=np.random.randn(N,D_out)

w1=np.random.randn(D_in,H)
w2=np.random.randn(H,D_out)

learning_rate=0.000001
for t in range(500):
    h=x.dot(w1)
    h_relu=np.maximum(h,0)
    y_pred=h_relu.dot(w2)

    loss=np.square(y_pred-y).sum()
    print(t,loss)

    grad_y_pred=2.0*(y_pred-y)
    grad_w2=h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    w1-=learning_rate*grad_w1
    w2-=learning_rate*grad_w2

import torch

dtype=torch.float
device=torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

#创建随机输入和输出数据
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype,requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype,requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传递：计算预测y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算和打印损失
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop计算w1和w2相对于损耗的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 使用梯度下降更新权重
    w1 =w1 - learning_rate * grad_w1
    w2 =w2 - learning_rate * grad_w2

# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device（“cuda：0”）＃取消注释以在GPU上运行

# N是批量大小; D_in是输入维度;
# H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机Tensors以保持输入和输出。
# 设置requires_grad = False表示我们不需要计算渐变
# 在向后传球期间对于这些Tensors。
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 为权重创建随机Tensors。
# 设置requires_grad = True表示我们想要计算渐变
# 在向后传球期间尊重这些张贴。
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传播：使用tensors上的操作计算预测值y;
      # 由于w1和w2有requires_grad=True，涉及这些张量的操作将让PyTorch构建计算图，
    # 从而允许自动计算梯度。由于我们不再手工实现反向传播，所以不需要保留中间值的引用。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 使用Tensors上的操作计算和打印丢失。
    # loss是一个形状为()的张量
    # loss.item() 得到这个张量对应的python数值
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用autograd计算反向传播。这个调用将计算loss对所有requires_grad=True的tensor的梯度。
    # 这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
    loss.backward()

    # 使用梯度下降更新权重。对于这一步，我们只想对w1和w2的值进行原地改变；不想为更新阶段构建计算图，
    # 所以我们使用torch.no_grad()上下文管理器防止PyTorch为更新构建计算图
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 反向传播后手动将梯度设置为零
        w1.grad.zero_()
        w2.grad.zero_()

import torch
N, D_in, H, D_out = 64, 1000, 100, 10

#创建输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)

loss_fn=torch.nn.MSELoss(reduction="sum")
learning_rate=1e-4
for i in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(t,loss.item())
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param=param-learning_rate*param.grad

import torch
import torch.optim as optim
N, D_in, H, D_out = 64, 1000, 100, 10

#创建输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
model=torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.ReLU(),
    torch.nn.Linear(H,D_out),
)

loss_fn=torch.nn.MSELoss(reduction="sum")
learning_rate=1e-4
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
for i in range(500):
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print(t,loss.item())
    model.zero_grad()
    loss.backward()
    optimizer.step()

#自定义网络
import torch
class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__()
        self.linear1=torch.nn.Linear(D_in,H)
        self.linear2=torch.nn.Linear(H,D_out)
    def forward(self,x):
        h_relu=self.linear1(x).clamp(min=0)
        y_pred=self.linear2(h_relu)
        return y_pred
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出的随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 通过实例化上面定义的类来构建我们的模型。
model = TwoLayerNet(D_in, H, D_out)

# 构造损失函数和优化器。
# SGD构造函数中对model.parameters()的调用，
# 将包含模型的一部分，即两个nn.Linear模块的可学习参数。
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # 前向传播：通过向模型传递x计算预测值y
    y_pred = model(x)

    #计算并输出loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # 清零梯度，反向传播，更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们构造了三个nn.Linear实例，它们将在前向传播时被使用。
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        对于模型的前向传播，我们随机选择0、1、2、3，
        并重用了多次计算隐藏层的middle_linear模块。
        由于每个前向传播构建一个动态计算图，
        我们可以在定义模型的前向传播时使用常规Python控制流运算符，如循环或条件语句。
        在这里，我们还看到，在定义计算图形时多次重用同一个模块是完全安全的。
        这是Lua Torch的一大改进，因为Lua Torch中每个模块只能使用一次。
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 实例化上面定义的类来构造我们的模型
model = DynamicNet(D_in, H, D_out)

# 构造我们的损失函数（loss function）和优化器（Optimizer）。
# 用平凡的随机梯度下降训练这个奇怪的模型是困难的，所以我们使用了momentum方法。
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # 前向传播：通过向模型传入x计算预测的y。
    y_pred = model(x)

    # 计算并打印损失
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # 清零梯度，反向传播，更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()