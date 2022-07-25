import torch
from src.vector_function import VectorFunction


def f1(x):
    return x[0]


def f2(x):
    n = x.shape[0]
    g = 1 + 9 / (n - 1) * x[1:].sum()
    h = 1 - torch.sqrt(x[0] / g) - (x[0] * g) * torch.sin(10 * torch.pi * x[0])
    return g * h


f = VectorFunction(f1, f2)
max_epoch = 500
lr = 1e-3
seed = 42
n = 2
