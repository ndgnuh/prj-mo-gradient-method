import numpy as np
from examples.test_3 import (
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


def last_not_na(xs):
    for i, x in enumerate(reversed(xs)):
        if not np.all(np.isnan(x)):
            return x


torch.manual_seed(seed)
x = nn.Parameter(torch.rand(n))
xs, values = optimize(f, x, max_epoch)

torch.manual_seed(seed)
x = nn.Parameter(torch.rand(n))
xspc, valuespc = optimize_pcgrad(f, x, max_epoch)

print("xs", xs[-1])
print("xspc", xspc[-1])


def plot_per_epoch(max_epoch, y, ypc):
    for i, y in enumerate(y):
        ax = plt.subplot(221 + i)
        plt.plot(range(max_epoch), y)
        plt.xlabel(f"f{i + 1} with normal update rule")
        ax.ticklabel_format(useOffset=False, style='plain')

    for i, y in enumerate(ypc):
        ax = plt.subplot(223 + i)
        plt.plot(range(max_epoch), y)
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.xlabel(f"f{i + 1} with PCGrad rule")
    plt.show()


def plot_pareto_like(y, ypc):
    ax = plt.subplot(211)
    plt.scatter(y[0], y[1])
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel("Pareto values with normal update rule")
    ax = plt.subplot(212)
    plt.scatter(ypc[0], ypc[1])
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel("Pareto values with PCGrad update rule")
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
