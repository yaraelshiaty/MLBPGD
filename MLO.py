import MGTomo.model as mgmodel
import MGTomo.functions as fcts
from scipy import interpolate
import numpy as np
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
    x0 = R(y)
    fH = lambda x: coarse_f(fh, x, l)
    grad_fh()
    
    val_f_y0, grad_f_y0 = fh(y0)
    val_fH_x0, grad_fH_x0 = fH(x0)
    val_fH_x, grad_fH_x = fH(x)
    
    kappa = R(grad_fh_y0) - grad_fH_x0
    
    val = val_fH_x + np.inner(kappa, x)
    val_grad = grad_fH_x + kappa
    
    return val, val_grad