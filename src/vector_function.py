import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from src.pcgrad import PCGrad


class NPVectorFunction:
    def __init__(self, *fs):
        self.fs = fs

    def __call__(self, *args):
        values = [f(*args) for f in self.fs]
        return np.stack(values)

    def __iter__(self):
        return iter(self.fs)

#     def grad(self):
#         return VectorFunction(*[grad(f) for f in self.fs])

#     def egrad(self):
#         return VectorFunction(*[elementwise_grad(f) for f in self.fs])


class Lambda(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class VectorFunction(nn.Module):
    def __init__(self, *funcs):
        super().__init__()
        self.layers = nn.ModuleList([Lambda(f) for f in funcs])

    def forward(self, x):
        ys = []
        for module in self.layers:
            y = module(x)
            ys.append(y)
        return ys


def optimize(f, x, opt, epoch):
    values = [[] for _ in f.layers]
    xs = []
    for e in tqdm(range(epoch)):
        opt.zero_grad()
        ys = f(x)
        loss = sum(ys)
        loss.backward()
        opt.step()
        xs.append(x.detach().numpy())
        for i, y in enumerate(ys):
            values[i].append(y.item())

    return xs, values


def optimize_pcgrad(f, x, opt, epoch, reduction='sum'):
    values = [[] for _ in f.layers]
    xs = []
    pc_grad = PCGrad(opt, reduction=reduction)
    for e in tqdm(range(epoch)):
        pc_grad.zero_grad()
        losses = f(x)
        pc_grad.pc_backward(losses)
        pc_grad.step()
        xs.append(x.detach().numpy())
        for i, y in enumerate(losses):
            values[i].append(y.item())

    return xs, values
