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
def armijo_step_size(f, x, d, control=0.5, start=1):
    jacob = jacobian(f, x)
    alpha = start
    m = torch.matmul(jacob, d)
    t = -control * m
    for i in range(20):
        alpha = alpha / 2
        lhs = f(x) - f(x + alpha * d)
        rhs = alpha * t
        if torch.all(lhs >= rhs):
            return alpha
    return alpha


@torch.no_grad()
def optimize(f, x, epoch):
    values = [[] for _ in f.layers]
    xs = []
    momentum = 0.05
    changed = 0
    for e in tqdm(range(epoch)):
        x = nn.Parameter(x)
        losses = f(x)
        dk = find_descend(f, x)
        step_size = armijo_step_size(f, x, dk)
        changed = dk * step_size + momentum * changed
        x = x + changed
        phi_ = phi(dk, f, x)
        print('phi', phi_, 'step_size', step_size)

        # if torch.abs(phi_) < 1e-6:
        #     break
        xs.append(x.detach().numpy())
        for i, y in enumerate(losses):
            values[i].append(y.item())

    return xs, values


def project_grad(jf):
    ljf = [grad for grad in jf]
    ljf_pc = deepcopy(ljf)
    for (i, _) in enumerate(ljf_pc):
        g_i = ljf_pc[i]
        perm = np.random.permutation(len(ljf))
        for j in perm:
            if i == j:
                continue
            g_j = ljf[j]
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                ljf_pc[i] = ljf_pc[i] - g_i_g_j / (torch.norm(g_j)**2) * g_j
    return torch.stack(ljf_pc)


@torch.no_grad()
def optimize_pcgrad(f, x, epoch, reduction='sum'):
    values = [[] for _ in f.layers]
    xs = []
    # epoch = 3
    momentum = 0.05
    changed = 0
    for e in tqdm(range(epoch)):
        jf = jacobian(f, x)
        jf_pc = project_grad(jf)
        dk = -getattr(jf_pc, reduction)(dim=0)
        losses = f(x)
        step_size = armijo_step_size(f, x, dk)
        phi_ = phi(dk, f, x)
        print('phi', phi_, 'step_size', step_size)
        changed = dk * step_size + momentum * changed
        x = x + changed
        xs.append(x.detach().numpy())
        for i, y in enumerate(losses):
            values[i].append(y.item())

    return xs, values


@torch.no_grad()
def optimize_pcgrad_multiarmijo(f, x, epoch, reduction='sum'):
    values = [[] for _ in f.layers]
    xs = []
    # epoch = 3
    changed = 0
    momentum = 0.05
    for e in tqdm(range(epoch)):
        jf = jacobian(f, x)
        step_sizes = torch.tensor([
            armijo_step_size(f_i, x, -dk_i)
            for (f_i, dk_i) in zip(f.layers, jf)])
        jf_pc = project_grad(jf * step_sizes[:, None])
        dk = -getattr(jf_pc, reduction)(dim=0)
        losses = f(x)
        step_size = armijo_step_size(f, x, dk, start=1)
        phi_ = phi(dk, f, x)
        print('phi', phi_, 'step_size', step_size)
        changed = dk * step_size + momentum * changed
        x = x + changed
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
