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

def SMART(f, x: torch.tensor, tau):
    fx = f(x)
    fx.backward()
    val = myexp(-tau * x.grad)
    
    return val