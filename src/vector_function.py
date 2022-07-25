import numpy as np
import torch
from torch import nn
from tqdm import tqdm
# from src.pcgrad import PCGrad
from scipy.optimize import fmin, NonlinearConstraint, minimize
from torch.autograd.functional import jacobian
from copy import deepcopy
from random import shuffle


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
def phi(d, f, x):
    jf = jacobian(f, x)
    if isinstance(d, np.ndarray):
        d = torch.tensor(d, dtype=x.dtype)
    return torch.max(torch.matmul(jf, d)) + 0.5 * torch.norm(d) ** 2


# @torch.no_grad()
# def find_descend(f, x):
#     out = fmin(phi, x, args=(f, x), disp=False)
#     return torch.tensor(out, dtype=x.dtype)


@torch.no_grad()
def find_descend(f, x):
    m = x.shape[0]
    n = len(f.layers)
    d = torch.zeros(m + 1, dtype=x.dtype)
    jf = jacobian(f, x)

    def as_tensor(inp):
        if isinstance(inp, np.ndarray):
            inp = torch.tensor(inp, dtype=jf.dtype)
        return inp

    def obj(inp):
        inp = as_tensor(inp)
        d = inp[1:]
        beta = inp[0]
        return beta + torch.norm(d)**2 / 2

    def cond(inp):
        inp = as_tensor(inp)
        d = inp[1:]
        beta = inp[0]
        lhs = torch.matmul(jf, d) - beta
        lhs = torch.cat([lhs, -torch.ones(1)])
        return lhs

    ub = torch.zeros(n + 1)
    lb = - torch.ones(n + 1) * torch.inf
    constraints = NonlinearConstraint(fun=cond,
                                      ub=ub,
                                      lb=lb)
    # bnd = [(-x_i, None) for x_i in x]
    d = minimize(obj, d, constraints=constraints).x[1:]
    return as_tensor(d)


@torch.no_grad()
def armijo_step_size(f, x, grad, lr=1):
    jacob = jacobian(f, x)
    alpha = 1
    for i in range(100):
        alpha = alpha / 2
        c = f(x + alpha * grad) - f(x) - lr * alpha * torch.matmul(jacob, grad)
        if c.all() < 0:
            return alpha
    return alpha + 3e-5


@torch.no_grad()
def optimize(f, x, epoch):
    values = [[] for _ in f.layers]
    xs = []
    lr = 1
    for e in tqdm(range(epoch)):
        losses = f(x)
        dk = find_descend(f, x)
        step_size = armijo_step_size(f, x, dk, lr=lr)
        x = x + dk * step_size * lr
        phi_ = phi(dk, f, x)
        print('phi', phi_, 'step_size', step_size)

        if torch.abs(phi_) < 1e-6:
            break
        xs.append(x.detach().numpy())
        for i, y in enumerate(losses):
            values[i].append(y.item())

    return xs, values


# def optimize_pcgrad(f, x, opt, epoch, reduction='sum'):
#     values = [[] for _ in f.layers]
#     xs = []
#     pc_grad = PCGrad(opt, reduction=reduction)
#     lr = opt.param_groups[0]['lr']
#     for e in tqdm(range(epoch)):
#         pc_grad.zero_grad()
#         losses = f(x)
#         pc_grad.pc_backward(losses)
#         step_size = armijo_step_size(f, x, x.grad, lr=lr)
#         x.grad = x.grad * step_size
#         pc_grad.step()
#         xs.append(x.detach().numpy())
#         for i, y in enumerate(losses):
#             values[i].append(y.item())

#     return xs, values


def project_grad(jf):
    ljf = [grad for grad in jf]
    ljf_pc = deepcopy(ljf)
    for (i, g_i) in enumerate(ljf_pc):
        shuffle(ljf)
        for g_j in ljf:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                ljf_pc[i] = ljf_pc[i] - g_i_g_j / (torch.norm(g_j)**2) * g_j
    return torch.stack(ljf_pc)


@torch.no_grad()
def optimize_pcgrad(f, x, epoch, reduction='sum'):
    values = [[] for _ in f.layers]
    xs = []
    # epoch = 3
    for e in tqdm(range(epoch)):
        jf = jacobian(f, x)
        # print('jf', jf)
        jf_pc = project_grad(jf)
        # print('pc', jf_pc)
        dk = -jf_pc.sum(dim=0)
        losses = f(x)
        step_size = armijo_step_size(f, x, dk)
        phi_ = phi(dk, f, x)
        print('phi', phi_, 'step_size', step_size)
        # print('step_size', step_size)
        x = x + dk * step_size
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
