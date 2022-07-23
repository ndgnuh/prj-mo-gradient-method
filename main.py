from examples.test_1 import f
import torch
from torch import nn
from src.vector_function import VectorFunction, optimize, optimize_pcgrad
from matplotlib import pyplot as plt
max_epoch = 200

torch.manual_seed(42)
x = nn.Parameter(torch.rand(2))
opt = torch.optim.AdamW((x,), lr=1e-2)
xs, values = optimize(f, x, opt, max_epoch)

torch.manual_seed(42)
x = nn.Parameter(torch.rand(2))
opt = torch.optim.AdamW((x,), lr=5e-3)
xspc, valuespc = optimize_pcgrad(f, x, opt, max_epoch)

for i, y in enumerate(values):
    plt.subplot(221 + i)
    plt.plot(range(max_epoch), y)
    plt.xlabel(f"f{i + 1} with normal update rule")

for i, y in enumerate(valuespc):
    plt.subplot(223 + i)
    plt.plot(range(max_epoch), y)
    plt.xlabel(f"f{i + 1} with PCGrad rule")
plt.show()
