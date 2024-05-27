import torch
import astra

import MGTomo.tomoprojection as mgproj
from MGTomo.utils import myexp, mylog, mydiv

def kl_distance(x: torch.tensor, proj: mgproj.TomoTorch, b: torch.tensor):
    ax = proj(x)
    ab = torch.divide(ax, b)
    
    erg = ax * mylog(ab) + b - ax
    fx = torch.sum( erg[b > 0.] ) + 0.5*torch.sum(ax[b == 0.]**2)
    
    return fx

def SMART(f, x: torch.tensor, tau = 0.2):
    fx = f(x)
    #x.requires_grad = True
    #x.retain_grad()
    fx.backward()
    val = myexp(-tau * x.grad)
    
    return val

def armijo_linesearch(f,x: torch.tensor,d: torch.tensor,a=1.,r=0.5,c=1e-4):
    fx = f(x)
    fx.backward()
    dgk = torch.sum(x.grad*d)
    f_new,_ = fct(x + a * d)
    x_new = x.copy()
    
    assert dgk <= 0, 'd needs to be a descent direction (dkg = %.5e)' % dgk
    
    if dgk == 0.:
        return x
    
    while f_new > f + a * c * dgk and x_new > 0 and a > 1e-7:
        x_new = x + a * d
        f_new = f(x_new)
        a *= r
    
    if f_new < f :
        return x_new, a
    else:
        return x
