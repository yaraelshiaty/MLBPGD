import MGTomo.model as mgmodel
import MGTomo.tomoprojection as mgproj
from MGTomo.utils import mylog
import multilevel.functions as fcts
from multilevel.optimize import armijo_linesearch, box_bounds_optimized
import multilevel.coarse_corrections as CC
from multilevel.gridop import RBox as R, PBox as P

from multilevel import gridop

import time
import numpy as np
import torch
from torch.linalg import norm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt 
from skimage import data
from skimage.transform import resize

import datetime

from PIL import Image


hparams = {
    "image": "walnut",
    "CC": "Bregman",
    "N": 1023,
    "max_levels": 2,
    "maxIter": [1,10,10,16,32,128],
    "num_angels0": 200,
    "P_inf" : 1.0,
    "SL_iterate_count": 150,
    "ML_iterate_count": 50,
    "kappa": 0.49,
    "eps": 0.001,
    "SL_image_indices": range(0,150,10),
    "ML_image_indices": range(0,50,5)
}

# x_orig = data.shepp_logan_phantom()
# x_orig = resize(x_orig, (hparams["N"],hparams["N"]), anti_aliasing = False)

# x_torch = torch.tensor(x_orig, requires_grad = True)

image = Image.open('walnut.png').convert('L')
image_np = np.array(image)
x_orig = np.array(image.resize((hparams["N"],hparams["N"])))/255
x_torch = torch.tensor(x_orig, requires_grad=True)

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
tau = [torch.reciprocal(Ai.sumnorm_opt()) for Ai in A]

def MLO_box(fh, y, lh, uh, last_pts: list, y_diff:list, l=0, kappa = hparams["kappa"], eps = hparams["eps"]):
    x = R(y).detach().requires_grad_(True)
    fhy0 = fh(y)
    #fhy0.backward(retain_graph = True)
    fhy0.backward()
    grad_fhy0 = y.grad.clone()
    y.grad = None

    CC_bool, y_diff[l] = CC.coarse_condition_bregman(y, grad_fhy0, kappa, eps, last_pts[l])

    #CC_bool, y_diff[l] = CC.coarse_condition_v2(y, grad_fhy0, kappa, eps, last_pts[l])
    
    if CC_bool:
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
            lH, uH = box_bounds_optimized(y, x, hparams["P_inf"], lh, uh, P_nonzero[l])

        logvH_new = mylog(x - lH) - mylog(uH - x)
        for i in range(hparams["maxIter"][l+1]):
            #x.retain_grad()
            val, logvH_new = fcts.BSMART_general(psi, x, tau[l+1], lH, uH, logvH_new)
            x = val.detach().requires_grad_(True)
            del val
            x.grad = None
            
        if l < hparams["max_levels"]-1:
            x, last_pts, y_diff = MLO_box(psi, x,lH, uH, last_pts, y_diff, l+1)

        d = P(x-x0)
        z, _ = armijo_linesearch(fh, y, d)
        y = z.detach().requires_grad_(True)
    else: 
        print(l, ' : coarse correction not activated')

    logvh_new = mylog(y - lh) - mylog(uh - y)
    
    for i in range(hparams["maxIter"][l]):
        #y.retain_grad()
        yval, logvh_new = fcts.BSMART_general(fh, y, tau[l], lh, uh, logvh_new)
        y = yval.detach().requires_grad_(True)
        del yval
        y.grad = None
    return y, last_pts, y_diff

################
# setup logging
log_dir = "runs/KLAxb_reconstruction/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = SummaryWriter(log_dir)
hparams["tau"] = tau
log_writer.add_text("hparams", str(hparams))
ckpt_path_SL = f"{log_writer.log_dir}/SL"
ckpt_path_ML = f"{log_writer.log_dir}/ML"
################


z0 = torch.ones(hparams["N"], hparams["N"]) * 0.5
z0.requires_grad_(True)
last_pts = [None]*(hparams["max_levels"]+1)
y_diff = torch.zeros(hparams['max_levels'])

lh = torch.zeros_like(z0)
uh = torch.ones_like(z0)

rel_f_err = []
rel_f_err.append((norm(z0 - x_torch, 'fro')/norm(z0, 'fro')).item())

norm_fval = []
norm_fval.append(torch.tensor(1.))

fhz = fh(z0)
# norm_fval.append(fhz.item())

fhz.backward(retain_graph=True)
Gz0 = norm(z0.grad, 'fro')
z0.grad = None

norm_grad = []
norm_grad.append(torch.tensor(1.))
#norm_grad.append(Gz0)

iteration_times_ML = []
iteration_times_ML.append(0)

for i in range(hparams['ML_iterate_count']):
    iteration_start_time_ML = time.time()
    
    val, ylast, ydiff = MLO_box(fh, z0, lh, uh, last_pts, y_diff)
    iteration_end_time_ML = time.time()
    iteration_time_ML = iteration_end_time_ML - iteration_start_time_ML

    iteration_times_ML.append(iteration_time_ML)
    z0 = val.clone().detach().requires_grad_(True)
    rel_f_err.append((norm(z0-x_torch, 'fro')/norm(z0, 'fro')).item())
    fval = fh(z0)
    norm_fval.append((fval/fhz).item())
    #norm_fval.append((fval).item())
    log_writer.add_scalar("ML_normalised_value", norm_fval[-1], i)
    fval.backward(retain_graph=True)
    norm_grad.append((norm(z0.grad, 'fro')/Gz0).item())
    #norm_grad.append((norm(z0.grad, 'fro')).item())
    log_writer.add_scalar("ML_normalised_gradient", norm_grad[-1], i)
    z0.grad = None

    if i in hparams["ML_image_indices"]:
        log_writer.add_image(f'ML_iter', z0, global_step=i, dataformats='HW')
    for j in range(len(y_diff)):
        log_writer.add_scalar(f'CC {j}', y_diff[j], i)

    print(f"Iteration {i}: {fh(z0)} - Time: {iteration_time_ML:.6f} seconds")

print(f"Overall time for all iterations: {sum(iteration_times_ML):.6f} seconds")
cumaltive_times_ML = [sum(iteration_times_ML[:i+1]) for i in range(len(iteration_times_ML))]

##########
np.savez(ckpt_path_ML, iteration_times_ML = iteration_times_ML, norm_fval_ML = norm_fval, norm_grad_ML = norm_grad, rel_f_err_ML = rel_f_err, last_iterate_ML = z0.detach().numpy())
##########


w0 = torch.ones(hparams["N"], hparams["N"], requires_grad = True)*0.5
fhw = fh(w0)
w0.retain_grad()
fhw.backward(retain_graph=True)
Gw0 = norm(w0.grad, 'fro')
logv_new = mylog((w0 - lh)) - mylog((uh - w0))

rel_f_err_SL = []
rel_f_err_SL.append((norm(w0 - x_torch, 'fro')/norm(w0, 'fro')).item())

norm_fval_SL = []
norm_fval_SL.append(torch.tensor(1.))

iteration_times_SL = []
iteration_times_SL.append(0)

norm_grad_SL = []
norm_grad_SL.append(torch.tensor(1.))

for i in range(hparams['SL_iterate_count']):
    iteration_start_time_SL = time.time()  # Start timing for this iteration
    
    val, logv_new = fcts.BSMART_general(fh, w0, tau[0], lh, uh, logv_new)
    
    iteration_end_time_SL = time.time()  # End timing for this iteration
    iteration_time_SL = iteration_end_time_SL - iteration_start_time_SL  # Calculate elapsed time for this iteration
    
    iteration_times_SL.append(iteration_time_SL)
    w0 = val.clone().detach().requires_grad_(True)

    rel_f_err_SL.append((norm(w0-x_torch, 'fro')/norm(w0, 'fro')).item())
    fval = fh(w0)
    norm_fval_SL.append((fval/fhw).item())
    #norm_fval_SL.append((fval).item())
    log_writer.add_scalar("SL_normalised_value", norm_fval_SL[-1], i)
    fval.backward(retain_graph=True)
    norm_grad_SL.append((norm(w0.grad, 'fro')/Gw0).item())
    #norm_grad_SL.append((norm(w0.grad, 'fro')).item())
    log_writer.add_scalar("SL_normalised_gradient", norm_grad_SL[-1], i)

    if i in hparams["SL_image_indices"]:
        log_writer.add_image(f'SL_iter', w0, global_step=i, dataformats='HW')

    print(f"Iteration {i}: {fh(w0)} - Time: {iteration_time_SL:.6f} seconds")

print(f"Overall time for all iterations: {sum(iteration_times_SL):.6f} seconds")
cumaltive_times_SL = [sum(iteration_times_SL[:i+1]) for i in range(len(iteration_times_SL))]

##########
np.savez(ckpt_path_SL, iteration_times_SL = iteration_times_SL, norm_fval_SL = norm_fval_SL, norm_grad_SL = norm_grad_SL, rel_f_err_SL = rel_f_err_SL, last_iterate_SL = w0.detach().numpy())
##########

# plt.figure(figsize=(10, 6))
# plt.plot(cumaltive_times_SL, norm_fval_SL, marker='o', linestyle='-', label = 'SL')
# plt.plot(cumaltive_times_ML, norm_fval, marker='o', linestyle='-', label = 'ML')
# plt.yscale('log')
# plt.xlabel('Cumulative CPU Time (seconds)')
# plt.ylabel('normalised function value')
# plt.title('normalised function value vs. CPU Time')
# plt.grid(True)
# plt.legend()

# log_writer.add_figure("normalised function value vs. CPU Time", plt.gcf())
# log_writer.close()