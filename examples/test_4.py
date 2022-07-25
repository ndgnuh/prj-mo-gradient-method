from src.vector_function import VectorFunction
import torch


def f1(x):
    return x[0]


def f2(x):
    n = x.shape[0]
    g = 1 + 10* (n - 1) +  (x[1:]**2- 10* torch.cos(4 * torch.pi * x[1:])).sum()
    h = 1 - torch.sqrt(x[0] / g)
    return g * h


f = VectorFunction(f1, f2)
max_epoch = 1000
lr = 5e-4
seed = 42
n = 2
