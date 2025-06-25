import torch
import numpy as np
from scipy.optimize import root_scalar
import torch.nn.functional as F

import MGTomo.tomoprojection as mgproj
from MGTomo.utils import myexp, mylog, mydiv

def kl_distance(x: torch.tensor, proj: mgproj.TomoTorch, b: torch.tensor):
    ax = proj(x)
    ab = mydiv(ax,b)
    
    erg = ax * mylog(ab) + b - ax
    fx = torch.sum( erg[b > 0.] ) + 0.5*torch.sum(ax[b == 0.]**2)
    assert fx >= 0, 'kl distance error: output is negative.'
    return fx.requires_grad_(True)

def kl_distance_no_matrix(x: torch.tensor, y: torch.tensor):
    xy = mydiv(x,y)

    erg = x * mylog(xy) + y - x
    fx = torch.sum( erg[y > 0.] ) + 0.5*torch.sum(x[y == 0.]**2)
    assert fx >= 0, fx
    return fx.requires_grad_(True)

def kl_distance_rev(x: torch.tensor, b: torch.tensor, A: mgproj.TomoTorch):
    ax = A(x)
    ax.requires_grad_(True)
    ba = mydiv(b,ax)
    
    erg = b * mylog(ba) - b + ax
    fx = torch.sum( erg[ax > 0.]) + 0.5*torch.sum(b[ax == 0.]**2)
    assert fx >= 0, fx
    return fx.requires_grad_(True)

def kl_distance_rev_pointwise(x: torch.tensor, b: torch.tensor, A):
    if not torch.is_tensor(A):
        A = torch.tensor(A)
    ax = A*x
    ax.requires_grad_(True)
    ba = mydiv(b,ax)
    
    erg = b * mylog(ba) - b + ax
    fx = torch.sum( erg[ax > 0.]) + 0.5*torch.sum(b[ax == 0.]**2)
    assert fx >= 0, fx
    #assert fx >= 0, 'kl distance error: output is negative.'
    return fx.requires_grad_(True)

def SMART(f, x: torch.tensor, tau):
    fx = f(x)
    fx.backward(retain_graph = True)
    val = x * myexp(-tau * x.grad)
    
    if (f(val) - fx).abs() < 1e-2*5:
        return x
    return val

def BSMART(f, x: torch.tensor, tau):
    fx = f(x)
    fx.backward(retain_graph = True)
    ones = torch.ones_like(x)
    val = mydiv(x, ones - x) * myexp(-tau * x.grad)
    
    x_new = mydiv(val, ones + val)
    
    if (f(x_new) - fx).abs() < 1e-2*5:
        return x
    
    return x_new

def BSMART_general(f, x: torch.tensor, tau, l, u, logv=None):
    x = x.clone().detach().requires_grad_(True)
    if logv is None:
        logv = mylog(x - l) - mylog(u - x)
    fx = f(x)
    fx.backward(retain_graph = True)
    xgrad = x.grad

    assert not torch.any(torch.isnan(logv))
    assert not torch.any(torch.isnan(xgrad))

    with torch.no_grad():
        logv_new = logv - tau * xgrad
        assert not torch.any(torch.isnan(logv_new))
        xminl = (u-l) * myexp(logv_new) * myexp(-F.softplus(logv_new))
        assert not torch.any(torch.isnan(xminl)), torch.any(myexp(logv_new - F.softplus(logv_new)))
        x_new = xminl + l

        assert not torch.any(torch.isnan(x_new)), print(x_new)

        assert torch.all(x_new >= l), (x_new - l).max()
        assert torch.all(x_new <= u + 1e-7), torch.sum(x_new - u <= 0)

    if (f(x_new) - fx).abs() < 1e-2*5:
        return x, logv

    return x_new, logv_new

def BSMART_general_ml(f, x: torch.Tensor, tau, l, u, logv=None, **context):
    """
    Mirror descent with box constraints, compatible with MultiLevelOptimizer.run.
    Accepts arbitrary **context for generality.
    Always returns (x_new, logv_new) so it can be used with *args, **kwargs in the optimizer.
    """
    # Allow logv to be passed via context if not given directly
    if logv is None and 'logv' in context:
        logv = context['logv']

    x = x.clone().detach().requires_grad_(True)
    if logv is None:
        logv = mylog(x - l) - mylog(u - x)
    fx = f(x)
    fx.backward(retain_graph=True)
    xgrad = x.grad

    assert not torch.any(torch.isnan(logv))
    assert not torch.any(torch.isnan(xgrad))

    with torch.no_grad():
        logv_new = logv - tau * xgrad
        xminl = (u - l) * myexp(logv_new) * myexp(-F.softplus(logv_new))
        x_new = xminl + l

        assert not torch.any(torch.isnan(x_new)), print(x_new)
        assert torch.all(x_new >= l - 1e-3), (x_new - l).max()
        assert torch.all(x_new <= u + 1e-3), torch.sum(x_new - u <= 0)

    if (f(x_new) - fx).abs() < 1e-2*5:
        return x, logv
    return x_new, logv_new

def Poisson_LIP(f, x: torch.tensor, tau):
    fx = f(x)
    fx.backward(retain_graph=True)
    xgrad = x.grad

    with torch.no_grad():
        val = torch.reciprocal(torch.reciprocal(x) + tau * xgrad)
    return val

def mirror_descent_IS(f, x:torch.tensor, tau, l, **context):
    fx = f(x)
    fx.backward(retain_graph=True)
    xgrad = x.grad
    # xgrad = A(torch.ones_like(x) - b/A(x))

    with torch.no_grad():
        x_new = l + torch.reciprocal(torch.reciprocal(x-l) + tau * xgrad)

    if (f(x_new) - fx).abs() < 1e-2*5:
        return x
    
    return x_new

def RL(f, x, A, b):
    # fx = f(x)
    # fx.backward(retain_graph = True)
    # xgrad = x.grad
    
    x_new = x * A(b/A(x))

    return x_new

#finite differences in 2D
def nabla(X):
    #D = (-1, 1) represents the discrete difference operator.

    assert X.shape[0] == X.shape[1], "Input matrix must be quadratic (n x n)."
    n = X.shape[0]

    # Compute the difference along columns (vertical direction)
    col_diff = X[1:, :] - X[:-1, :]  # Difference between rows
    col_diff_padded = torch.cat((col_diff, torch.zeros(1, n)), dim=0)  # Pad last row with zeros

    # Compute the difference along rows (horizontal direction)
    row_diff = X[:, 1:] - X[:, :-1]  # Difference between columns
    row_diff_padded = torch.cat((row_diff, torch.zeros(n, 1)), dim=1)  # Pad last column with zeros

    nabla_X = torch.cat((col_diff_padded, row_diff_padded))

    return nabla_X

def tv_huber(X, rho):
    abs_X = torch.abs(X)
    mask = (abs_X < rho)
    result = torch.where(mask, abs_X**2 / (2 * rho), abs_X - rho / 2)

    return torch.sum(result)

def finite_diff_huber(X, rho):
    return tv_huber(nabla(X), rho)

def bpgd_newton_solver(f, x, tau, lh, uh=None, *args, **context):
    """
    General BPGD interface for Newton/root_scalar update.
    Expects context to provide:
        - 'x0':  (required)
        - 'theta': initial guess for root finding (optional, default=100)
        - 'root_equation': function for root_scalar (required)
        - 'root_derivative': derivative for root_scalar (required)
        - 'bracket': tuple for root_scalar (optional, default computed from c)
    """
    fx = f(x)
    fx.backward(retain_graph=True)
    xgrad = x.grad

    c = xgrad + 1/x
    S = context['x0'].sum().item()
    root_equation = context.get('root_equation', None)
    root_derivative = context.get('root_derivative', None)
    if root_equation is None or root_derivative is None:
        raise ValueError("root_equation and root_derivative must be provided in context.")
    theta = args[0] if len(args) > 0 else context.get('theta', 100)
    bracket = context.get('bracket', [-(torch.min(c)).item(), 1e8])

    # Detach and convert c to numpy for root_scalar
    c_np = c.detach().cpu().numpy()
    S_np = S  # already a float
    theta_np = float(theta)
    bracket_np = [float(bracket[0]), float(bracket[1])]

    # Solve for theta using root_scalar (no grad)
    res = root_scalar(
        root_equation,
        bracket=bracket_np,
        args=(c_np, S_np),
        x0=theta_np,
        fprime=root_derivative,
        method="bisect"
    )
    theta_new = res.root
    # x_new as torch tensor, same device/dtype as x
    x_new = 1 / (c + torch.tensor(theta_new, dtype=x.dtype, device=x.device))
    return x_new, theta_new

def Doptdesign(x: torch.Tensor, H: torch.Tensor):
    X = torch.diag(x)
    FIM = H @ X @ H.T
    sign, logdet = torch.linalg.slogdet(FIM)
    return -logdet.requires_grad_(True)