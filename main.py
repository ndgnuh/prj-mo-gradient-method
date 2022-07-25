import numpy as np
from examples.test_2 import (
    f,
    max_epoch,
    n,
    lr,
    seed,
)
import torch
from torch import nn
from src.vector_function import (
    VectorFunction,
    optimize,
    optimize_pcgrad,
    find_dominate_set,
)
from matplotlib import pyplot as plt

max_epoch = 100


def last_not_na(xs):
    for i, x in enumerate(reversed(xs)):
        if not np.all(np.isnan(x)):
            return x


torch.manual_seed(seed)
x = nn.Parameter(torch.rand(n))
opt = torch.optim.AdamW((x,), lr=lr)
xs, values = optimize(f, x, opt, max_epoch)

torch.manual_seed(seed)
x = nn.Parameter(torch.rand(n))
opt = torch.optim.AdamW((x,), lr=lr)
xspc, valuespc = optimize_pcgrad(f, x, opt, max_epoch)

# print("xs", xs[-1])
# print("xspc", xspc[-1])


def plot_per_epoch(max_epoch, y, ypc):
    for i, y in enumerate(y):
        plt.subplot(221 + i)
        plt.plot(range(max_epoch), y)
        plt.xlabel(f"f{i + 1} with normal update rule")

    for i, y in enumerate(ypc):
        plt.subplot(223 + i)
        plt.plot(range(max_epoch), y)
        plt.xlabel(f"f{i + 1} with PCGrad rule")
    plt.show()


def plot_pareto_like(y, ypc):
    plt.subplot(211)
    plt.scatter(y[0], y[1])
    plt.subplot(212)
    plt.scatter(ypc[0], ypc[1])
    plt.show()


# idx_pc = find_dominate_set(valuespc)
idx = find_dominate_set(values)
idxpc = find_dominate_set(valuespc)
values = np.array(values)
valuespc = np.array(valuespc)
pareto = values[:, idx]
paretopc = valuespc[:, idx]
print(values.shape, idx.shape, pareto.shape)

# PLOT
plot_per_epoch(max_epoch, values, valuespc)
plot_pareto_like(pareto, paretopc)
