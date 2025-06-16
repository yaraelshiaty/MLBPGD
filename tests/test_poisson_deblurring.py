import MGBlurr.blurring as blur
import multilevel.functions as fcts
from MGTomo.utils import mylog, mydiv
from multilevel.optimize import armijo_linesearch, box_bounds, orthant_bounds
from multilevel.gridop import RBox as R, PBox as P
from multilevel import gridop

from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt

import torch
from torch.linalg import matrix_norm

import time

N = 1023
max_levels = 6
maxIter = [1,2,4,32,64,128]
kernel_size = 15
sigma = 10

# load image
x_orig = data.camera()
x_orig = resize(x_orig, (N,N), anti_aliasing = False)

x_torch = torch.tensor(x_orig, requires_grad = True)

A = [blur.GaussianBlurOperator(N, kernel_size, sigma)]
b = [torch.poisson(A[0](x_torch)*50)/50]
P_nonzero = []

fine_dim = N
for i in range(1, max_levels+1):
    coarse_dim = blur.reduce_dim(fine_dim)
    A.append(blur.GaussianBlurOperator(coarse_dim, kernel_size, sigma))
    rhs = resize(b[-1].detach().numpy(), (coarse_dim, coarse_dim), anti_aliasing=False)
    b.append(torch.tensor(rhs, requires_grad=True)) #maybe use a different way to define bH
    P_nonzero.append(gridop.compute_nonzero_elements_of_P(coarse_dim))
    fine_dim = coarse_dim

def kl_distance_rev(x: torch.tensor, b: torch.tensor, A):
    ax = A(x)
    ax.requires_grad_(True)
    ba = mydiv(b,ax)
    
    erg = b * mylog(ba) - b + ax
    #fx = torch.sum(erg[b > 0.])
    fx = torch.sum( erg[ax > 0.]) + 0.5*torch.sum(b[ax == 0.]**2)
    assert fx >= 0, fx
    #assert fx >= 0, 'kl distance error: output is negative.'
    return fx.requires_grad_(True)

def coarse_condition_v2(y, grad_y, kappa, eta, y_last = None):
    with torch.no_grad():
        gcond = (matrix_norm(R(grad_y)) >= kappa * matrix_norm(grad_y))
        if gcond:
            if y_last is not None:
                y_diff_norm = matrix_norm(y_last - y)
                y_norm = matrix_norm(y)
                return (y_diff_norm >= eta * y_norm)
            return True
        else:
            return False
        
def MLO_orthant(fh, y, lh, last_pts: list, l=0, kappa = 0.49, eps = 0.001):
    x = R(y).detach().requires_grad_(True)
    y0 = y.detach().requires_grad_(True)
    fhy0 = fh(y)
    fhy0.backward(retain_graph=True)
    grad_fhy0 = y.grad.clone()
    y.grad = None
    
    if coarse_condition_v2(y, grad_fhy0, kappa, eps, last_pts[l]):
        print(l, ' : coarse correction activated')
        last_pts[l] = y.detach()
    
        x0 = x.detach().requires_grad_(True)
        fH = lambda x: kl_distance_rev(x, b[l+1], A[l+1])
        fHx0 = fH(x0)
        fHx0.backward(retain_graph = True)
        grad_fHx0 = x0.grad.clone()
        x0.grad = None

        kappa = R(grad_fhy0) - grad_fHx0

        del grad_fHx0

        with torch.no_grad():
            psi = lambda x: fH(x) + torch.sum(kappa * x)
            lH = orthant_bounds(y, x, P_inf, lh, P_nonzero[l])
        
        for i in range(maxIter[l]):
            #x.retain_grad()
            val = fcts.mirror_descent_IS(psi, x, tau[l+1], lH)
            x = val.detach().requires_grad_(True)
            del val
            x.grad = None
            
        if l < max_levels-1:
            x, last_pts = MLO_orthant(psi, x, lH, last_pts, l+1)

        d = P(x-x0)
        z, _ = armijo_linesearch(fh, y0, d)
        y = z.detach().requires_grad_(True)
    else: 
        print(l, ' : coarse correction not activated')
    
    for i in range(maxIter[l]):
        #y.retain_grad()
        yval = fcts.mirror_descent_IS(fh, y, tau[l], lh)
        y = yval.detach().requires_grad_(True)
        del yval
        y.grad = None
    return y, last_pts


fh = lambda x: kl_distance_rev(x, b[0], A[0])
tau = [0.5 * torch.reciprocal(matrix_norm(bi, ord = 1)) for bi in b]
P_inf = 1

if __name__ == "__main__":
    w0 = torch.ones(N, N, requires_grad = True)*0.5
    lh = torch.zeros_like(w0)

    rel_f_err_SL = []
    rel_f_err_SL.append((matrix_norm(w0 - x_torch)/matrix_norm(w0)).item())

    norm_fval_SL = []
    norm_fval_SL.append(torch.tensor(1.))

    fhw = fh(w0)

    iteration_times_SL = []
    iteration_times_SL.append(0)
    overall_start_time_SL = time.time()  # Start overall timing

    w0.retain_grad()

    i=0
    while fh(w0) >= 100:
        iteration_start_time_SL = time.time()  # Start timing for this iteration
        
        val = fcts.mirror_descent_IS(fh, w0, tau[0], lh)
        
        iteration_end_time_SL = time.time()  # End timing for this iteration
        iteration_time_SL = iteration_end_time_SL - iteration_start_time_SL  # Calculate elapsed time for this iteration
        
        iteration_times_SL.append(iteration_time_SL)
        w0 = val.clone().detach().requires_grad_(True)
        rel_f_err_SL.append((matrix_norm(w0-x_torch)/matrix_norm(w0)).item())
        norm_fval_SL.append((fh(w0)/fhw).item())
        
        print(f"Iteration {i}: {fh(w0)} - Time: {iteration_time_SL:.6f} seconds")
        i+=1

    overall_end_time_SL = time.time()  # End overall timing
    overall_time_SL = overall_end_time_SL - overall_start_time_SL  # Calculate overall elapsed time

    print(f"Overall time for all iterations: {overall_time_SL:.6f} seconds")
    cumaltive_times_SL = [sum(iteration_times_SL[:i+1]) for i in range(len(iteration_times_SL))]

    
    z0 = torch.ones(N, N) * 0.5
    z0.requires_grad_(True)
    last_pts = [None]*(max_levels+1)

    lh = torch.zeros_like(z0)

    rel_f_err = []
    rel_f_err.append((matrix_norm(z0 - x_torch)/matrix_norm(z0)).item())

    norm_fval = []
    norm_fval.append(torch.tensor(1.))

    fhz = fh(z0)

    iteration_times_ML = []
    iteration_times_ML.append(0)
    overall_start_time_ML = time.time()

    for i in range(100):
        iteration_start_time_ML = time.time()
        
        val, ylast = MLO_orthant(fh, z0, lh, last_pts)
        iteration_end_time_ML = time.time()
        iteration_time_ML = iteration_end_time_ML - iteration_start_time_ML

        iteration_times_ML.append(iteration_time_ML)
        z0 = val.clone().detach().requires_grad_(True)
        rel_f_err.append((matrix_norm(z0-x_torch)/matrix_norm(z0)).item())
        norm_fval.append((fh(z0)/fhz).item())
        
        print(f"Iteration {i}: {fh(z0)} - Time: {iteration_time_ML:.6f} seconds")

    overall_end_time_ML = time.time()  # End overall timing
    overall_time_ML = overall_end_time_ML - overall_start_time_ML  # Calculate overall elapsed time

    print(f"Overall time for all iterations: {overall_time_ML:.6f} seconds")

    cumaltive_times_ML = [sum(iteration_times_ML[:i+1]) for i in range(len(iteration_times_ML))]


    plt.figure(figsize=(10, 6))
    plt.plot(cumaltive_times_ML, rel_f_err, marker='o', linestyle='-', label = 'ML')
    plt.plot(cumaltive_times_SL, rel_f_err_SL, marker='o', linestyle='-', label = 'SL')
    plt.xlabel('Cumulative CPU Time (seconds)')
    plt.ylabel('relative forward error')
    plt.title('relative forward error vs. CPU Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(cumaltive_times_ML, norm_fval, marker='o', linestyle='-', label = 'ML')
    plt.plot(cumaltive_times_SL, norm_fval_SL, marker='o', linestyle='-', label = 'SL')
    plt.xlabel('Cumulative CPU Time (seconds)')
    plt.ylabel('normalised function value')
    plt.title('normalised function value vs. CPU Time')
    plt.grid(True)
    plt.legend()
    plt.show()