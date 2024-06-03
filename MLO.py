#import MGTomo.model as mgmodel
import MGTomo.functions as fcts
from MGTomo.optimize import armijo_linesearch
from scipy import interpolate
import numpy as np
import torch
from torch.func import grad

from skimage.transform import rescale, resize

def MLO(fh, y, l=0):
    x = R(y).detach().requires_grad_(True)
    y0, x0 = y, x.clone().detach().requires_grad_(True)
    
    fhy0 = fh(y0)
    fhy0.backward(retain_graph = True)
    grad_fhy0 = y0.grad.clone()
    y0.grad.zero_()
    
    fH = lambda x: fcts.kl_distance(x, A[l+1], b[l+1])
    fHx0 = fH(x0)
    fHx0.backward(retain_graph = True)
    grad_fHx0 = x0.grad.clone()
    x0.grad.zero_()
    
    kappa = R(grad_fhy0) - grad_fHx0
    
    psi = lambda x: fH(x) + torch.sum(kappa * (x-x0))
    
    for i in range(maxIter[l]):
        x.retain_grad()
        val = fcts.SMART(psi, x, tau[l+1])
        x = val.clone().detach().requires_grad_(True)
      
    if l < max_levels-1:
        x = MLO(psi, x, l+1)
    
    assert psi(x) < psi(x0), 'psi(x) < psi(x0) = fH(x0) does not hold'
    
    d = P(x-x0)
    z, a = fcts.armijo_linesearch(fh, y0, d)
    #print('stepsize length for armijo', a)
    
    assert z.min() >= 0
    
    for i in range(maxIter[l]):
        z.retain_grad()
        zval = fcts.SMART(fh, z, tau[l])
        y0.grad.zero_()
        z = zval.clone().detach().requires_grad_(True)
    return z