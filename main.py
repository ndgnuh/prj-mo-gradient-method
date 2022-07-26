import numpy as np
from importlib import import_module
import torch
from torch import nn
from src.vector_function import (
    VectorFunction,
    optimize,
    optimize_pcgrad,
    find_dominate_set,
)
from matplotlib import pyplot as plt

import matplotlib
import sys
example_num = sys.argv[-1]
em = import_module(f"examples.test_{example_num}")
f = em.f
max_epoch = em.max_epoch
n = em.n
lr = em.lr
seed = em.seed

font = {'family': 'serif',
        'size': 16}

matplotlib.rc('font', **font)


def last_not_na(xs):
    for i, x in enumerate(reversed(xs)):
        if not np.all(np.isnan(x)):
            return x


torch.manual_seed(seed)
x = nn.Parameter(torch.rand(n) / 2)
xs, values = optimize(f, x, max_epoch)

torch.manual_seed(seed)
x = nn.Parameter(torch.rand(n) / 2)
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


def plot_pareto_like(y, ypc, type='values'):
    def r(x):
        return np.around(x, decimals=5)
    ax = plt.subplot(211)
    plt.scatter(r(y[0]), y[1])
    ax.ticklabel_format(useOffset=False, style='plain')
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    plt.xlabel(f"Pareto {type} with normal update rule")
    ax = plt.subplot(212)
    plt.scatter(r(ypc[0]), ypc[1])
    ax.ticklabel_format(useOffset=False, style='plain')
    # ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    plt.xlabel(f"Pareto {type} with PCGrad update rule")


# idx_pc = find_dominate_set(valuespc)
idx = find_dominate_set(values)
idxpc = find_dominate_set(valuespc)
values = np.array(values)
valuespc = np.array(valuespc)
pareto = values[:, idx]
paretopc = valuespc[:, idx]

# PLOT
plt.figure(figsize=(21, 10.8))
plot_per_epoch(max_epoch, values, valuespc)
plt.savefig(f"perepoch-{example_num}.png")
plt.figure(figsize=(10, 10))
plot_pareto_like(pareto, paretopc)
plt.savefig(f"pareto-{example_num}.png")
xs = np.array(xs)
xspc = np.array(xspc)
# print(xs.shape)
# if xs.shape[-1] == 2:
#     x_pareto = xs[idx, :]
#     xpc_pareto = xspc[idx, :]
#     plot_pareto_like(x_pareto, xpc_pareto, type='points')
