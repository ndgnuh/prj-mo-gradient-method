from autograd import numpy as np
from autograd.numpy.linalg import norm
from autograd import grad, jacobian, elementwise_grad
from scipy.optimize import minimize, NonlinearConstraint


class VectorFunction:
    def __init__(self, *fs):
        self.fs = fs

    def __call__(self, *args):
        values = [f(*args) for f in self.fs]
        return np.stack(values)

    def __iter__(self):
        return iter(self.fs)

    def grad(self):
        return VectorFunction(*[grad(f) for f in self.fs])

    def egrad(self):
        return VectorFunction(*[elementwise_grad(f) for f in self.fs])


def f1(x):
    a = [25, 1, 1, 1, 1, 0]
    b = [2, 2, 1, 4, 1, 0]
    return -((x - b)**2 * a).sum()


def f2(x):
    return (x**2).sum()


def f(x):
    return np.stack([f1(x), f2(x)])


def find_d_k(fs, dfs, x):
    m = dfs.shape[0] + 1
    n = x.shape[-1]

    def d_obj(inp):
        d = inp[1:]
        beta = inp[0]
        return beta + norm(d)**2 / 2

    def d_con(inp):
        d = inp[1:]
        beta = inp[0:1]
        lhs = dfs.dot(d) - beta
        lhs = lhs.tolist()
        lhs.append(-1)
        lhs = np.stack(lhs)
        # print(lhs.shape)
        return lhs

    inp = np.random.rand(n + 1)
    ub = np.zeros(m)
    lb = - np.ones(m) * np.inf
    constraints = NonlinearConstraint(fun=d_con,
                                      ub=ub,
                                      lb=lb)

    return minimize(d_obj, inp, constraints=constraints)


def msdm(fs, x, sigma=0.5, max_k=2):
    dfs = fs.egrad()
    jf = jacobian(fs)
    for k in range(max_k):
        print("Step: ", k)
        # Tìm d_k
        dfs_x = dfs(x)
        d_k_result = find_d_k(fs, dfs_x, x)
        d_k_result = find_d_k(f, dfs_x, x)
        assert d_k_result.success
        d_k = d_k_result.x[1:]

        # Check dk dừng
        theta_d_k = (dfs_x.dot(d_k) + norm(d_k)**2 / 2).max()
        if np.abs(theta_d_k) < 1e-3:
            print("Stop")
            return x, True

        print("Theta", theta_d_k)

        jf_x = jf(x)
        fx = f(x)
        print("f(x)", norm(fx), fx)

        # Tìm alpha
        alpha_k = 1
        for j in range(1000):
            alpha = 1.0 / (2**j)
            c = f(x + alpha * d_k) - fx - sigma * alpha * jf_x.dot(d_k)
            if c.all() < 0:
                alpha_k = alpha
                break

        x = x + alpha_k * d_k
    return x, False


np.random.seed(0)
x0 = np.random.randn(6)
f = VectorFunction(f1, f2)
x, success = msdm(f, x0, max_k=100)
print("x = ", x, ", success = ", success)
