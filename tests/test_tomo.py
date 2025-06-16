import MGTomo.model as mgmodel
import MGTomo.tomoprojection as mgproj
from MGTomo.utils import mylog
import multilevel.functions as fcts
from multilevel.optimize import armijo_linesearch, box_bounds
from multilevel.gridop import RBox as R, PBox as P
from multilevel import gridop

import time
import numpy as np
import torch
from torch.linalg import matrix_norm

import matplotlib.pyplot as plt 
from skimage import data
from skimage.transform import resize

max_levels = 1
maxIter = [1,2,5,5,5,5]

N = 255
x_orig = data.shepp_logan_phantom()
x_orig = resize(x_orig, (N,N), anti_aliasing = False)
x_torch = torch.tensor(x_orig, requires_grad = True)

model = mgmodel.astra_model(N,{'mode' : 'line', 'num_angles' : 50, 'level_decrease' : 1})
fine_dim = model.dim
A = [mgproj.TomoTorch(model.proj_factory(fine_dim))]
b = [A[0](x_torch)]
P_nonzero = []


for i in range(1,max_levels+1):
    coarse_dim = model.reduce_dim(fine_dim)
    A.append(mgproj.TomoTorch(model.proj_factory(coarse_dim)))
    b.append(torch.from_numpy(model.reduce_rhs(b[-1].detach().numpy(), fine_dim, coarse_dim)))
    P_nonzero.append(gridop.compute_nonzero_elements_of_P(coarse_dim))
    fine_dim=coarse_dim

fh = lambda x: fcts.kl_distance(x, A[0], b[0])

c0 = 100
tau0 = 0.5 * 1/c0
tau = [tau0]*(max_levels+1)

def coarse_condition_v2(y, grad_y, kappa, eta, y_last=None):
    grad_y_norm = matrix_norm(grad_y)
    if matrix_norm(R(grad_y)) >= kappa * grad_y_norm:
        if y_last is not None:
            y_diff_norm = matrix_norm(y_last - y)
            if y_diff_norm >= eta * matrix_norm(y):
                return True
        return y_last is None
    return False

def MLO_box(fh, y, lh, uh, last_pts: list, l=0, kappa = 0.49, eps = 0.001, verbose = True):
    x = R(y).detach().requires_grad_(True)
    
    fhy0 = fh(y)
    #fhy0.backward(retain_graph = True)
    fhy0.backward()
    grad_fhy0 = y.grad.clone()
    y.grad.zero_()
    
    if coarse_condition_v2(y, grad_fhy0, kappa, eps, last_pts[l]):
    #if coarse_condition_v3(grad_fhy0, kappa, eps):
        print(l, ' : coarse correction activated')
        last_pts[l] = y.clone().detach()
    
        x0 = x.clone().detach().requires_grad_(True)
        fH = lambda x: fcts.kl_distance(x, A[l+1], b[l+1])
        fHx0 = fH(x0)
        fHx0.backward(retain_graph = True)
        grad_fHx0 = x0.grad.clone()
        x0.grad.zero_()

        kappa = R(grad_fhy0) - grad_fHx0

        psi = lambda x: fH(x) + torch.sum(kappa * x)
        lH, uH = box_bounds(y, x, P_inf, lh, uh, P_nonzero[l])

        logvH_new = mylog(x - lH) - mylog(uH - x)
        for i in range(maxIter[l]):
            #x.retain_grad()
            val, logvH_new = fcts.BSMART_general(psi, x, logvH_new, tau[l+1], lH, uH)
            x = val.detach().requires_grad_(True)
            
        if l < max_levels-1:
            x, last_pts = MLO_box(psi, x,lH, uH, last_pts, l+1, verbose=verbose)

        d = P(x-x0)
        z, _ = armijo_linesearch(fh, y, d)
        y = z.detach().requires_grad_(True)
    else: 
        print(l, ' : coarse correction not activated')

    logvh_new = mylog(y - lh) - mylog(uh - y)
    
    print(l)
    for i in range(maxIter[l]):
        #y.retain_grad()
        yval, logvh_new = fcts.BSMART_general(fh, y, logvh_new, tau[l], lh, uh)
        y = yval.detach().requires_grad_(True)
    print("postsmoothing done")
    return y, last_pts

if __name__ == "__main__":
    P_inf = 1
    a = []
    z0 = torch.ones(N, N) * 0.5
    z0.requires_grad_(True)
    last_pts = [None]*(max_levels+1)

    lh = torch.zeros_like(z0)
    uh = torch.ones_like(z0)

    rel_f_err = []
    rel_f_err.append((matrix_norm(z0 - x_torch)/matrix_norm(z0)).item())

    iteration_times_ML = []
    iteration_times_ML.append(0)
    overall_start_time_ML = time.time()

    for i in range(100):
        iteration_start_time_ML = time.time()
        
        val, ylast = MLO_box(fh, z0, lh, uh, last_pts, verbose=False)
        print("one step done")

        iteration_end_time_ML = time.time()
        iteration_time_ML = iteration_end_time_ML - iteration_start_time_ML

        iteration_times_ML.append(iteration_time_ML)
        z0 = val.clone().detach().requires_grad_(True)
        rel_f_err.append((matrix_norm(z0-x_torch)/matrix_norm(z0)).item())
        
        print(f"Iteration {i}: {fh(z0)} - Time: {iteration_time_ML:.6f} seconds")

    overall_end_time_ML = time.time()  # End overall timing
    overall_time_ML = overall_end_time_ML - overall_start_time_ML  # Calculate overall elapsed time

    print(f"Overall time for all iterations: {overall_time_ML:.6f} seconds")

    cumaltive_times_ML = [sum(iteration_times_ML[:i+1]) for i in range(len(iteration_times_ML))]


    w0 = torch.ones(N, N) * 0.5
    w0.requires_grad_(True)

    rel_f_err_SL = []
    rel_f_err_SL.append((matrix_norm(w0 - x_torch)/matrix_norm(w0)).item())

    iteration_times_SL = []
    iteration_times_SL.append(0)
    overall_start_time_SL = time.time()  # Start overall timing

    logv_new = (w0 - lh) / (uh - w0)

    for i in range(100):
        iteration_start_time_SL = time.time()  # Start timing for this iteration
        
        w0, logv_new = fcts.BSMART_general(fh, w0, logv_new, tau0, lh, uh)
        
        iteration_end_time_SL = time.time()  # End timing for this iteration
        iteration_time_SL = iteration_end_time_SL - iteration_start_time_SL  # Calculate elapsed time for this iteration
        
        iteration_times_SL.append(iteration_time_SL)
        w0 = val.clone().detach().requires_grad_(True)
        rel_f_err_SL.append((matrix_norm(w0-x_torch)/matrix_norm(w0)).item())
        
        print(f"Iteration {i}: {fh(w0)} - Time: {iteration_time_SL:.6f} seconds")

    overall_end_time_SL = time.time()  # End overall timing
    overall_time_SL = overall_end_time_SL - overall_start_time_SL  # Calculate overall elapsed time

    print(f"Overall time for all iterations: {overall_time_SL:.6f} seconds")
    cumaltive_times_SL = [sum(iteration_times_SL[:i+1]) for i in range(len(iteration_times_SL))]