import torch
from src.vector_function import VectorFunction


def f(x):
    return (x[0]-1)**2 + 3 * (x[1] - 10)**2


def g(x):
    return 2 * (x[1] - 10)**4 + (x[0]-1)**2 + 1


f = VectorFunction(f, g)
max_epoch = 100
lr = 0.5
seed = 10
n = 2
