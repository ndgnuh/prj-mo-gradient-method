from examples.test_1 import f
import torch
from torch import nn

x = nn.Parameter(torch.rand(3))
print(f(x))
