import torch
import torch.nn.functional as F

import MGTomo.tomoprojection as mgproj
from MGTomo.utils import myexp, mylog, mydiv

def kl_distance(x: torch.tensor, proj: mgproj.TomoTorch, b: torch.tensor):
    ax = proj(x)
    #ab = torch.divide(ax, b)
    ab = mydiv(ax,b)
    
    erg = ax * mylog(ab) + b - ax
    fx = torch.sum( erg[b > 0.] ) + 0.5*torch.sum(ax[b == 0.]**2)
    assert fx >= 0, fx
    #assert fx >= 0, 'kl distance error: output is negative.'
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
    
    #assert val.max() <= 1.
    
    if (f(val) - fx).abs() < 1e-2*5:
        return x
    
    #assert f(val) < fx, 'SMART iterations do not descend'
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

def BSMART_general(f, x: torch.tensor, logv, tau, l, u):
    x = x.clone().detach().requires_grad_(True)
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
        #(u - x_new).min()
        #assert torch.all(x_new <= u), x_new.flatten()[(u - x_new).argmin()]

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

def mirror_descent_IS(f, x:torch.tensor, tau, l):
    fx = f(x)
    fx.backward(retain_graph=True)
    xgrad = x.grad

    with torch.no_grad():
        x_new = l + torch.reciprocal(torch.reciprocal(x-l) + tau * xgrad)

    if (f(x_new) - fx).abs() < 1e-2*5:
        return x
    
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