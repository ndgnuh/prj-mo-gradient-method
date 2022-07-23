import numpy as np
import torch
from torch import nn
from tqdm import tqdm


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


class VectorFunction(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        ys = [module(x) for module in self.modules]
        return torch.stack(ys)


def optimize(f, x, opt, epoch):
    values = []
    xs = []
    for e in tqdm(epoch):
        y = f(x)
