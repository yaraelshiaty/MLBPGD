import torch
import astra

import MGTomo.tomoprojection as mgproj
from MGTomo.utils import myexp, mylog, mydiv

def kl_distance_v2(x: torch.tensor, proj: mgproj.TomoTorch, b: torch.tensor):
    ax = proj(x)
    #ab = torch.divide(ax, b)
    ab = mydiv(ax,b)
    
    erg = ax * mylog(ab) + b - ax
    fx = torch.sum( erg[b > 0.] ) + 0.5*torch.sum(ax[b == 0.]**2)
    #fx = torch.sum(erg[b > 0.])
    return fx

def kl_distance(x: torch.tensor, proj: mgproj.TomoTorch, b: torch.tensor):
    ax = proj(x)
    erg = ax * (mylog(ax) - mylog(b)) + b - ax
    fx = torch.sum(erg)
    
    return fx.requires_grad_(True)

def SMART(f, x: torch.tensor, tau):
    fx = f(x)
    fx.backward(retain_graph = True)
    val = x * myexp(-tau * x.grad)
    #val.requires_grad = True
    
    return val

def armijo_linesearch(f,x: torch.tensor,d: torch.tensor,a=1.,r=0.5,c=1e-4):
    fx = f(x)
    fx.backward()
    dgk = torch.sum(x.grad*d)
    f_new = f(x + a * d)
    x_new = x
    
    assert dgk <= 0, 'd needs to be a descent direction (dkg = %.5e)' % dgk
    
    if dgk == 0.:
        return x
    
    while f_new > fx + a * c * dgk and a > 1e-7:
        #needed added x_new > 0
        x_new = x + a * d
        f_new = f(x_new)
        a *= r
    
    if f_new < fx :
        return x_new
    else:
        return x
