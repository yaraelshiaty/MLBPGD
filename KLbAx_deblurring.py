import MGBlurr.blurring as blur
import multilevel.functions as fcts
from multilevel.coarse_corrections import coarse_condition
from multilevel.optimize import armijo_linesearch, orthant_bounds_optimized
from multilevel.gridop import RBox as R, PBox as P
from multilevel import gridop
import multilevel.coarse_corrections as CC

from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.linalg import norm

import time
import datetime
import numpy as np

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

hparams = {
    "image": "crater",
    "CC": "Bregman",
    "N": 511,
    "max_levels": 2,
    "maxIter": [1,10,10,32,64,128],
    "kernel_size": 15,
    "sigma": 1.5,
    "poisson_lbd": 1000,
    "P_inf" : 0.5,
    "SL_iterate_count": 10,
    "ML_iterate_count": 10,
    "kappa": 0.49,
    "eps": 0.001,
    "SL_image_indices": range(0,60,59),
    "ML_image_indices": range(0,60,59)
}

# load image
# x_orig = data.camera()
# x_orig = resize(x_orig, (hparams["N"],hparams["N"]), anti_aliasing = False)

# filename = get_pkg_data_filename('tutorials/FITS-images/HorseHead.fits')
# hdul = fits.open(filename)
# image_data = hdul[0].data
from PIL import Image
image = Image.open('tycho-crater.png').convert('L')
image_data = np.array(image)

x_orig = resize(image_data, (hparams["N"],hparams["N"]), anti_aliasing = False)
x_torch = torch.tensor(x_orig, requires_grad = True)


A = [blur.GaussianBlurOperator(hparams["N"], hparams["kernel_size"], hparams["sigma"])]
b = [torch.poisson(A[0](x_torch)*hparams["poisson_lbd"])/hparams["poisson_lbd"]]
assert torch.all(b[-1] >= 0)
P_nonzero = []

fine_dim = hparams["N"]
for i in range(1, hparams["max_levels"]+1):
    coarse_dim = blur.reduce_dim(fine_dim)
    A.append(blur.GaussianBlurOperator(coarse_dim, hparams["kernel_size"], hparams["sigma"]))
    rhs = resize(b[-1].detach().numpy(), (coarse_dim, coarse_dim), anti_aliasing=False)
    b.append(torch.tensor(rhs, requires_grad=True)) #maybe use a different way to define bH#
    assert torch.all(b[-1] >= 0)
    P_nonzero.append(gridop.compute_nonzero_elements_of_P(coarse_dim))
    fine_dim = coarse_dim

fh = lambda x: fcts.kl_distance_rev(x, b[0], A[0])
tau = [torch.reciprocal(norm(bi, ord = 1)) for bi in b]

def MLO_orthant(fh, y, lh, last_pts: list, y_diff:list, l=0, kappa = hparams["kappa"], eps = hparams["eps"]):
    x = R(y).detach().requires_grad_(True)
    y0 = y.detach().requires_grad_(True)
    fhy0 = fh(y)
    fhy0.backward(retain_graph=True)
    grad_fhy0 = y.grad.clone()
    y.grad = None
    
    CC_bool, y_diff[l] = CC.coarse_condition_bregman(y, grad_fhy0, kappa, eps, last_pts[l])

    if CC_bool:
    #if True:
        print(l, ' : coarse correction activated')
        last_pts[l] = y.detach()
    
        x0 = x.detach().requires_grad_(True)
        fH = lambda x: fcts.kl_distance_rev(x, b[l+1], A[l+1]) 
        fHx0 = fH(x0)
        fHx0.backward(retain_graph = True)
        grad_fHx0 = x0.grad.clone()
        x0.grad = None

        kappa = R(grad_fhy0) - grad_fHx0

        del grad_fHx0

        with torch.no_grad():
            psi = lambda x: fH(x) + torch.sum(kappa * x)
            lH, _ = orthant_bounds_optimized(y, x, hparams["P_inf"], lh, P_nonzero[l])
        
        for i in range(hparams["maxIter"][l+1]):
            #x.retain_grad()
            val = fcts.mirror_descent_IS(psi, x, tau[l+1], lH)
            x = val.detach().requires_grad_(True)
            del val
            x.grad = None
            
        if l < hparams["max_levels"]-1:
            x, last_pts, y_diff = MLO_orthant(psi, x, lH, last_pts, y_diff, l+1)

        d = P(x-x0)
        z, _ = armijo_linesearch(fh, y0, d)
        y = z.detach().requires_grad_(True)
    else: 
        print(l, ' : coarse correction not activated')
    
    for i in range(hparams["maxIter"][l]):
        #y.retain_grad()
        yval = fcts.mirror_descent_IS(fh, y, tau[l], lh)
        y = yval.detach().requires_grad_(True)
        del yval
        y.grad = None
    return y, last_pts, y_diff


################
# setup logging
log_dir = "runs/KLbAx_deblurring/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = SummaryWriter(log_dir)
hparams["tau"] = tau
log_writer.add_text("hparams", str(hparams))
ckpt_path_SL = f"{log_writer.log_dir}/SL"
ckpt_path_ML = f"{log_writer.log_dir}/ML"
ckpt_path_RL = f"{log_writer.log_dir}/RL"
################


z0 = torch.ones(hparams["N"], hparams["N"]) * 0.5
z0.requires_grad_(True)
last_pts = [None]*(hparams["max_levels"]+1)
y_diff = torch.zeros(hparams['max_levels'])

lh = torch.zeros_like(z0)

rel_f_err = []
rel_f_err.append((norm(z0 - x_torch, 'fro')/norm(z0, 'fro')).item())

norm_fval = []
norm_fval.append(torch.tensor(1.))

fhz = fh(z0)

fhz.backward(retain_graph=True)
Gz0 = norm(z0.grad, 'fro')
z0.grad = None

norm_grad = []
norm_grad.append(torch.tensor(1.))

iteration_times_ML = []
iteration_times_ML.append(0)

for i in range(hparams['ML_iterate_count']):
    iteration_start_time_ML = time.time()
    
    val, ylast, y_diff = MLO_orthant(fh, z0, lh, last_pts, y_diff)
    iteration_end_time_ML = time.time()
    iteration_time_ML = iteration_end_time_ML - iteration_start_time_ML

    iteration_times_ML.append(iteration_time_ML)
    z0 = val.clone().detach().requires_grad_(True)
    rel_f_err.append((norm(z0-x_torch, 'fro')/norm(z0, 'fro')).item())
    fval = fh(z0)
    norm_fval.append((fval/fhz).item())
    log_writer.add_scalar("ML_normalised_value", norm_fval[-1], i)
    fval.backward(retain_graph=True)
    norm_grad.append((norm(z0.grad, 'fro')/Gz0).item())
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
# w0.grad = None

iteration_times_SL = []
iteration_times_SL.append(0)

norm_grad_SL = []
norm_grad_SL.append(torch.tensor(1.))

rel_f_err_SL = []
rel_f_err_SL.append((norm(w0 - x_torch, 'fro')/norm(w0, 'fro')).item())

norm_fval_SL = []
norm_fval_SL.append(torch.tensor(1.))
for i in range(hparams['SL_iterate_count']):
    iteration_start_time_SL = time.time()  # Start timing for this iteration
    
    val = fcts.mirror_descent_IS(fh, w0, tau[0], lh)
    
    iteration_end_time_SL = time.time()  # End timing for this iteration
    iteration_time_SL = iteration_end_time_SL - iteration_start_time_SL  # Calculate elapsed time for this iteration
    
    iteration_times_SL.append(iteration_time_SL)
    w0 = val.clone().detach().requires_grad_(True)

    fval = fh(w0)
    norm_fval_SL.append((fval/fhw).item())
    log_writer.add_scalar("SL_normalised_value", norm_fval_SL[-1], i)
    fval.backward(retain_graph=True)
    norm_grad_SL.append((norm(w0.grad, 'fro')/Gw0).item())
    log_writer.add_scalar("SL_normalised_gradient", norm_grad_SL[-1], i)
    w0.grad = None
    rel_f_err_SL.append((norm(w0 - x_torch, 'fro')/norm(w0, 'fro')).item())

    if i in hparams["SL_image_indices"]:
        log_writer.add_image(f'SL_iter', w0, global_step=i, dataformats='HW')

    print(f"Iteration {i}: {fh(w0)} - Time: {iteration_time_SL:.6f} seconds")

print(f"Overall time for all iterations: {sum(iteration_times_SL):.6f} seconds")
cumaltive_times_SL = [sum(iteration_times_SL[:i+1]) for i in range(len(iteration_times_SL))]

##########
np.savez(ckpt_path_SL, iteration_times_SL = iteration_times_SL, norm_fval_SL = norm_fval_SL, norm_grad_SL = norm_grad_SL, rel_f_err_SL = rel_f_err_SL, last_iterate_SL = w0.detach().numpy())
##########


w0 = torch.ones(hparams["N"], hparams["N"], requires_grad = True)*0.5
fhw = fh(w0)
w0.retain_grad()
fhw.backward(retain_graph=True)
Gw0 = norm(w0.grad, 'fro')
# w0.grad = None

iteration_times_RL = []
iteration_times_RL.append(0)

norm_grad_RL = []
norm_grad_RL.append(torch.tensor(1.))

rel_f_err_RL = []
rel_f_err_RL.append((norm(w0 - x_torch, 'fro')/norm(w0, 'fro')).item())

norm_fval_RL = []
norm_fval_RL.append(torch.tensor(1.))
for i in range(hparams['SL_iterate_count']):
    iteration_start_time_RL = time.time()  # Start timing for this iteration
    
    val = fcts.RL(fh, w0, A[0], b[0])
    
    iteration_end_time_RL = time.time()  # End timing for this iteration
    iteration_time_RL = iteration_end_time_RL - iteration_start_time_RL  # Calculate elapsed time for this iteration
    
    iteration_times_RL.append(iteration_time_RL)
    w0 = val.clone().detach().requires_grad_(True)

    fval = fh(w0)
    norm_fval_RL.append((fval/fhw).item())
    log_writer.add_scalar("RL_normalised_value", norm_fval_RL[-1], i)
    fval.backward(retain_graph=True)
    norm_grad_RL.append((norm(w0.grad, 'fro')/Gw0).item())
    log_writer.add_scalar("RL_normalised_gradient", norm_grad_RL[-1], i)
    w0.grad = None
    rel_f_err_RL.append((norm(w0 - x_torch, 'fro')/norm(w0, 'fro')).item())

    if i in hparams["SL_image_indices"]:
        log_writer.add_image(f'RL_iter', w0, global_step=i, dataformats='HW')

    print(f"Iteration {i}: {fh(w0)} - Time: {iteration_time_RL:.6f} seconds")

print(f"Overall time for all iterations: {sum(iteration_times_RL):.6f} seconds")
cumaltive_times_RL = [sum(iteration_times_RL[:i+1]) for i in range(len(iteration_times_RL))]

##########
np.savez(ckpt_path_RL, iteration_times_RL = iteration_times_RL, norm_fval_RL = norm_fval_RL, norm_grad_RL = norm_grad_RL, rel_f_err_RL = rel_f_err_RL, last_iterate_RL = w0.detach().numpy())
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