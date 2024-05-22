import MGTomo.model as mgmodel
import MGTomo.functions as fcts
from scipy import interpolate
import numpy as np
import torch
import torch.func import grad

import matplotlib.pyplot as plt
import h5py
from skimage.transform import rescale, resize

import pylops
import sys
import pandas as pd

def R(y):
    x = y[1:-1:2, 1:-1,2]
    return x

def iterative_method(f, x, tau):
    

def MLO(fh, y, l=0):
    x = R(y)
    y0, x0 = y, R(y)
    psi = lambda x: coarsen_fn(fh, x, y0)
    
    for i in range(maxIter[l]):
        x = iterative_method(psi, x, tau)
        
    if l < max_levels:
        x = MLO(psi, x, l+1)
        
    d = P(x-x0)
    y = linesearch(fh, y0, d)
    
    for i in range(maxIter[l]):
        y = iterative_method(psi, y, tau)
    
    return y

def coarse_f(fh, l):
    

def coarsen_fn(fh, x, y0, l):
    x0 = R(y0)
    fH = lambda x: coarse_f(fh, x, l)
    grad_fh = grad(fh)
    grad_fH = grad(fH)
    
    kappa = R(grad_fh(y0)) - grad_fH(x0)
    val = fH(x) + torch.inner(kappa, x)
    
    #val_grad = grad_fH(x) + kappa
    
    return val, val_grad