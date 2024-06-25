import torch
import astra

import MGTomo.tomoprojection as mgproj
from MGTomo.utils import myexp, mylog, mydiv

def kl_distance(x: torch.tensor, proj: mgproj.TomoTorch, b: torch.tensor):
    ax = proj(x)
    #ab = torch.divide(ax, b)
    ab = mydiv(ax,b)
    
    erg = ax * mylog(ab) + b - ax
    fx = torch.sum( erg[b > 0.] ) + 0.5*torch.sum(ax[b == 0.]**2)
    #fx = torch.sum(erg[b > 0.])
    assert fx >= 0, fx
    #assert fx >= 0, 'kl distance error: output is negative.'
    return fx.requires_grad_(True)

def kl_distance_no_matrix(x: torch.tensor, y: torch.tensor):
    xy = mydiv(x,y)

    erg = x * mylog(xy) + y - x
    fx = torch.sum( erg[y > 0.] ) + 0.5*torch.sum(x[y == 0.]**2)
    assert fx >= 0, fx
    return fx.requires_grad_(True)

def kl_distance_v2(x: torch.tensor, proj: mgproj.TomoTorch, b: torch.tensor):
    ax = proj(x)
    erg = ax * (mylog(ax) - mylog(b)) + b - ax
    fx = torch.sum( erg[b > 0.] ) + 0.5*torch.sum(ax[b == 0.]**2)
    assert fx >= 0, 'kl distance error: output is negative.'
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