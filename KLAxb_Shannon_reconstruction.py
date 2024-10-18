import MGTomo.model as mgmodel
import MGTomo.tomoprojection as mgproj
from MGTomo.utils import mylog
import MGTomo.functions as fcts
from MGTomo.optimize import armijo_linesearch, box_bounds

from MGTomo.gridop import RBox as R, PBox as P

from MGTomo import gridop

import time
import numpy as np
import torch
from torch.linalg import matrix_norm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt 
from skimage import data
from skimage.transform import resize

import datetime


hparams = {
    "image": "shepp_logan",
    "CC": "v2",
    "N": 1023,
    "max_levels": 3,
    "maxIter": [1,2,4,8,16,32],
    "tau" : [0.0005, 0.001, 0.001, 0.0045, 0.0091, 0.0185],
    "num_angels0": 200,
    "P_inf" : 1,
    "SL_iterate_count": 100,
    "ML_iterate_count": 20,
    "kappa": 0.45,
    "eps": 0.001
}

x_orig = data.shepp_logan_phantom()
x_orig = resize(x_orig, (hparams["N"],hparams["N"]), anti_aliasing = False)

x_torch = torch.tensor(x_orig, requires_grad = True)


model = mgmodel.astra_model(hparams["N"],{'mode' : 'line', 'num_angles' : hparams["num_angels0"], 'level_decrease' : 1})
fine_dim = model.dim
A = [mgproj.TomoTorch(model.proj_factory(fine_dim))]
b = [A[0](x_torch)]
level = {int(np.sqrt(A[0].shape[1])): 0}
P_nonzero = []


for i in range(1,hparams["max_levels"]+1):
    coarse_dim = model.reduce_dim(fine_dim)
    model_coarse = mgmodel.astra_model(coarse_dim, {'mode' : 'line', 'num_angles' : min(int(coarse_dim*np.pi/4),100), 'level_decrease' : 1})
    A.append(mgproj.TomoTorch(model_coarse.proj_factory(coarse_dim)))
    x_resized = resize(x_orig, (coarse_dim, coarse_dim), anti_aliasing=False)
    xT_resized = torch.tensor(x_resized, requires_grad = True)
    b.append(A[-1](xT_resized))
    P_nonzero.append(gridop.compute_nonzero_elements_of_P(coarse_dim))
    level.update({int(np.sqrt(A[i].shape[1])): i})
    fine_dim=coarse_dim

for i in range(hparams["max_levels"]+1):
    assert b[i].shape[0]*b[i].shape[1] == A[i].shape[0], 'dimension mismatch'
    print(f'level {i}:', b[i].shape[0], np.sqrt(A[i].shape[1]))


fh = lambda x: fcts.kl_distance(x, A[0], b[0])

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

def MLO_box(fh, y, lh, uh, last_pts: list, l=0, kappa = hparams["kappa"], eps = hparams["eps"]):
    x = R(y).detach().requires_grad_(True)
    fhy0 = fh(y)
    #fhy0.backward(retain_graph = True)
    fhy0.backward()
    grad_fhy0 = y.grad.clone()
    y.grad = None
    
    if coarse_condition_v2(y, grad_fhy0, kappa, eps, last_pts[l]):
    #if True:
        print(l, ' : coarse correction activated')
        last_pts[l] = y.clone().detach()
    
        x0 = x.clone().detach().requires_grad_(True)
        fH = lambda x: fcts.kl_distance(x, A[l+1], b[l+1]) 
        fHx0 = fH(x0)
        fHx0.backward(retain_graph = True)
        grad_fHx0 = x0.grad.clone()
        x0.grad = None

        kappa = R(grad_fhy0) - grad_fHx0
        del grad_fHx0

        with torch.no_grad():
            psi = lambda x: fH(x) + torch.sum(kappa * x)
            lH, uH = box_bounds(y, x, hparams["P_inf"], lh, uh, P_nonzero[l])

        logvH_new = mylog(x - lH) - mylog(uH - x)
        for i in range(hparams["maxIter"][l+1]):
            #x.retain_grad()
            val, logvH_new = fcts.BSMART_general(psi, x, logvH_new, hparams["tau"][l+1], lH, uH)
            x = val.detach().requires_grad_(True)
            del val
            x.grad = None
            
        if l < hparams["max_levels"]-1:
            x, last_pts = MLO_box(psi, x,lH, uH, last_pts, l+1)

        d = P(x-x0)
        z, _ = armijo_linesearch(fh, y, d)
        y = z.detach().requires_grad_(True)
    else: 
        print(l, ' : coarse correction not activated')

    logvh_new = mylog(y - lh) - mylog(uh - y)
    
    for i in range(hparams["maxIter"][l]):
        #y.retain_grad()
        yval, logvh_new = fcts.BSMART_general(fh, y, logvh_new, hparams["tau"][l], lh, uh)
        y = yval.detach().requires_grad_(True)
        del yval
        y.grad = None
    return y, last_pts

################
# setup logging
log_dir = "runs/KLAxb_reconstruction/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = SummaryWriter(log_dir)
log_writer.add_text("hparams", str(hparams))
ckpt_path_SL = f"{log_writer.log_dir}/SL"
ckpt_path_ML = f"{log_writer.log_dir}/ML"
################


z0 = torch.ones(hparams["N"], hparams["N"]) * 0.5
z0.requires_grad_(True)
last_pts = [None]*(hparams["max_levels"]+1)

lh = torch.zeros_like(z0)
uh = torch.ones_like(z0)

rel_f_err = []
rel_f_err.append((matrix_norm(z0 - x_torch)/matrix_norm(z0)).item())

norm_fval = []
norm_fval.append(torch.tensor(1.))

fhz = fh(z0)

fhz.backward(retain_graph=True)
Gz0 = matrix_norm(z0.grad)
z0.grad = None

norm_grad = []
norm_grad.append(torch.tensor(1.))

iteration_times_ML = []
iteration_times_ML.append(0)

for i in range(hparams['ML_iterate_count']):
    iteration_start_time_ML = time.time()
    
    val, ylast = MLO_box(fh, z0, lh, uh, last_pts)
    iteration_end_time_ML = time.time()
    iteration_time_ML = iteration_end_time_ML - iteration_start_time_ML

    iteration_times_ML.append(iteration_time_ML)
    z0 = val.clone().detach().requires_grad_(True)
    rel_f_err.append((matrix_norm(z0-x_torch)/matrix_norm(z0)).item())
    fval = fh(z0)
    norm_fval.append((fval/fhz).item())
    log_writer.add_scalar("ML_normalised_value", norm_fval[-1], i)
    fval.backward(retain_graph=True)
    norm_grad.append((matrix_norm(z0.grad)/Gz0).item())
    log_writer.add_scalar("ML_normalised_gradient", norm_grad[-1], i)
    z0.grad = None
    print(f"Iteration {i}: {fh(z0)} - Time: {iteration_time_ML:.6f} seconds")

print(f"Overall time for all iterations: {sum(iteration_times_ML):.6f} seconds")
cumaltive_times_ML = [sum(iteration_times_ML[:i+1]) for i in range(len(iteration_times_ML))]

##########
np.savez(ckpt_path_ML, iteration_times_ML = iteration_times_ML, norm_fval_ML = norm_fval, norm_grad_ML = norm_grad, rel_f_err_ML = rel_f_err)
##########


w0 = torch.ones(hparams["N"], hparams["N"], requires_grad = True)*0.5
fhw = fh(w0)
w0.retain_grad()
fhw.backward(retain_graph=True)
Gw0 = matrix_norm(w0.grad)
logv_new = (w0 - lh) / (uh - w0)

rel_f_err_SL = []
rel_f_err_SL.append((matrix_norm(w0 - x_torch)/matrix_norm(w0)).item())

norm_fval_SL = []
norm_fval_SL.append(torch.tensor(1.))

iteration_times_SL = []
iteration_times_SL.append(0)

norm_grad_SL = []
norm_grad_SL.append(torch.tensor(1.))

for i in range(hparams['SL_iterate_count']):
    iteration_start_time_SL = time.time()  # Start timing for this iteration
    
    val, logv_new = fcts.BSMART_general(fh, w0, logv_new, hparams["tau"][0], lh, uh)
    
    iteration_end_time_SL = time.time()  # End timing for this iteration
    iteration_time_SL = iteration_end_time_SL - iteration_start_time_SL  # Calculate elapsed time for this iteration
    
    iteration_times_SL.append(iteration_time_SL)
    w0 = val.clone().detach().requires_grad_(True)

    rel_f_err_SL.append((matrix_norm(w0-x_torch)/matrix_norm(w0)).item())
    fval = fh(w0)
    norm_fval_SL.append((fval/fhw).item())
    log_writer.add_scalar("SL_normalised_value", norm_fval_SL[-1], i)
    fval.backward(retain_graph=True)
    norm_grad_SL.append((matrix_norm(w0.grad)/Gw0).item())
    log_writer.add_scalar("SL_normalised_gradient", norm_grad_SL[-1], i)
    print(f"Iteration {i}: {fh(w0)} - Time: {iteration_time_SL:.6f} seconds")

print(f"Overall time for all iterations: {sum(iteration_times_SL):.6f} seconds")
cumaltive_times_SL = [sum(iteration_times_SL[:i+1]) for i in range(len(iteration_times_SL))]

##########
np.savez(ckpt_path_SL, iteration_times_SL = iteration_times_SL, norm_fval_SL = norm_fval_SL, norm_grad_SL = norm_grad_SL, rel_f_err_SL = rel_f_err_SL)
##########

plt.figure(figsize=(10, 6))
plt.plot(cumaltive_times_ML, norm_fval, marker='o', linestyle='-', label = 'ML')
plt.plot(cumaltive_times_SL, norm_fval_SL, marker='o', linestyle='-', label = 'SL')
plt.yscale('log')
plt.xlabel('Cumulative CPU Time (seconds)')
plt.ylabel('normalised function value')
plt.title('normalised function value vs. CPU Time')
plt.grid(True)
plt.legend()

log_writer.add_figure("normalised function value vs. CPU Time", plt.gcf())
log_writer.close()