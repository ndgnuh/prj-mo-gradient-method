import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from src.pcgrad import PCGrad


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


class Lambda(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class VectorFunction(nn.Module):
    def __init__(self, *funcs):
        super().__init__()
        self.layers = nn.ModuleList([Lambda(f) for f in funcs])

    def forward(self, x):
        ys = []
        for module in self.layers:
            y = module(x)
            ys.append(y)
        return torch.stack(ys)


@torch.no_grad()
def armijo_step_size(f, x, grad, lr=1):
    # Tìm alpha
    jacob = torch.autograd.functional.jacobian(f, (x,))
    # print(len(jacob))
    # print(jacob[0].shape)
    jacob = jacob[0]
    alpha = 1
    for i in range(150):
        alpha = alpha / 2
        c = f(x + alpha * grad) - f(x) - lr * alpha * (jacob @ grad)
        if c.all() < 0:
            return alpha
    return 1


def optimize(f, x, opt, epoch):
    values = [[] for _ in f.layers]
    xs = []
    lr = opt.param_groups[0]['lr']
    for e in tqdm(range(epoch)):
        opt.zero_grad()
        ys = f(x)
        loss = sum(ys)
        loss.backward()
        step_size = armijo_step_size(f, x, x.grad, lr=lr)
        x.grad = x.grad * step_size
        opt.step()
        xs.append(x.detach().numpy())
        for i, y in enumerate(ys):
            values[i].append(y.item())

    return xs, values


def optimize_pcgrad(f, x, opt, epoch, reduction='sum'):
    values = [[] for _ in f.layers]
    xs = []
    pc_grad = PCGrad(opt, reduction=reduction)
    lr = opt.param_groups[0]['lr']
    for e in tqdm(range(epoch)):
        pc_grad.zero_grad()
        losses = f(x)
        pc_grad.pc_backward(losses)
        step_size = armijo_step_size(f, x, x.grad, lr=lr)
        x.grad = x.grad * step_size
        pc_grad.step()
        xs.append(x.detach().numpy())
        for i, y in enumerate(losses):
            values[i].append(y.item())

    return xs, values


def find_dominate_set(xs):
    if not isinstance(xs, torch.Tensor):
        # Shape: n * d
        xs = torch.tensor(xs).transpose(0, 1)

    le = xs[None, :, :] <= xs[:, None, :]
    lt = xs[None, :, :] < xs[:, None, :]
    is_dominated = torch.all(le, dim=-1) & torch.any(lt, dim=-1)
    dominated = is_dominated.any(dim=1)
    idx = torch.where(torch.logical_not(dominated))
    return idx[0]
