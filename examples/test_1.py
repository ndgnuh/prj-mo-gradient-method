from src.vector_function import VectorFunction


def f1(x):
    return x[0]**2 + 3 * (x[1] - 1)**2


def f2(x):
    return 2 * (x[0] - 1)**2 + x[1]**2


f = VectorFunction(f1, f2)
max_epoch = 100
lr = 5e-4
seed = 42
n = 2
