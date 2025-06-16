import MGTomo.model as mgmodel
import MGTomo.tomoprojection as mgproj
from MGTomo.utils import mylog
import multilevel.functions as fcts
from multilevel.optimize import armijo_linesearch, box_bounds_optimized
import multilevel.coarse_corrections as CC
from multilevel.MultiGridOperator import MultigridOperator2D
from multilevel.multilevel import MultiLevelOptimizer
from multilevel.results import extract_ml_metrics_with_cc

import time
import numpy as np
import torch
from torch.linalg import norm
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt 
from skimage.transform import resize

import datetime
from PIL import Image

# --- Hyperparameters ---
hparams = {
    "image": "walnut",
    "CC": "Bregman",
    "N": 1023,
    "max_levels": 2,
    "maxIter": [1,10,10,16,32,128],
    "num_angels0": 200,
    "ML_iterate_count": 50,
    "kappa": 0.49,
    "eps": 0.001,
    "ML_image_indices": range(0,50,5),
    "bounds": "box"
}

# --- Load image ---
image = Image.open('walnut.png').convert('L')
image_np = np.array(image)
x_orig = np.array(image.resize((hparams["N"],hparams["N"])))/255
x_torch = torch.tensor(x_orig, requires_grad=True)

# --- Build operators and right-hand sides ---
model = mgmodel.astra_model(hparams["N"],{'mode' : 'line', 'num_angles' : hparams["num_angels0"], 'level_decrease' : 1})
fine_dim = model.dim
A = [mgproj.TomoTorch(model.proj_factory(fine_dim))]
b = [A[0](x_torch)]
input_sizes = [fine_dim]

for i in range(1,hparams["max_levels"]+1):
    coarse_dim = model.reduce_dim(fine_dim)
    model_coarse = mgmodel.astra_model(coarse_dim, {'mode' : 'line', 'num_angles' : min(int(coarse_dim*np.pi/4),100), 'level_decrease' : 1})
    A.append(mgproj.TomoTorch(model_coarse.proj_factory(coarse_dim)))
    x_resized = resize(x_orig, (coarse_dim, coarse_dim), anti_aliasing=False)
    xT_resized = torch.tensor(x_resized, requires_grad = True)
    b.append(A[-1](xT_resized))
    input_sizes.append(coarse_dim)
    fine_dim=coarse_dim
hparams["input_sizes"] = input_sizes
print(f"Input sizes for each level: {hparams['input_sizes']}")

# --- Objective and step sizes ---
fh_list = [lambda x, bi=bi, Ai=Ai: fcts.kl_distance(x, Ai, bi) for bi, Ai in zip(b, A)]
tau = [torch.reciprocal(Ai.sumnorm_opt()) for Ai in A]

# --- Multigrid kernel ---
kernel = torch.tensor([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=torch.float32)/16.0

optimizer = MultiLevelOptimizer(
    fh_list=fh_list,
    tau=tau,
    kernel=kernel,
    hparams=hparams,
    BPGD=fcts.BSMART_general_ml,
    linesearch=armijo_linesearch,
    bounds=hparams["bounds"],
    CC=None
)

# --- Logging setup ---
log_dir = "runs/KLAxb_reconstruction_ML/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = SummaryWriter(log_dir)
hparams["tau"] = tau
log_writer.add_text("hparams", str(hparams))
ckpt_path_ML = f"{log_writer.log_dir}/ML"

# --- Initialization ---
z0 = torch.ones(hparams["N"], hparams["N"]) * 0.5
z0.requires_grad_(True)
last_pts = [None]*(hparams["max_levels"]+1)
y_diff = torch.zeros(hparams['max_levels'])

lh = torch.zeros_like(z0)
uh = torch.ones_like(z0)
logv = None  # Let BSMART_general handle initialization

rel_f_err = []
rel_f_err.append((norm(z0 - x_torch, 'fro')/norm(z0, 'fro')).item())

norm_fval = []
norm_fval.append(torch.tensor(1.))

fhz = fh_list[0](z0)
fhz.backward(retain_graph=True)
Gz0 = norm(z0.grad, 'fro')
z0.grad = None

norm_grad = []
norm_grad.append(torch.tensor(1.))

iteration_times_ML = []
iteration_times_ML.append(0)

# --- Multilevel optimization loop ---
for i in range(hparams['ML_iterate_count']):
    iteration_start_time_ML = time.time()
    
    prev_last_pts = [p.clone() if p is not None else None for p in last_pts]

    val, last_pts, y_diff = optimizer.run(z0, lh, uh=uh, last_pts=last_pts, y_diff=y_diff, l=0, logv=logv)
    iteration_end_time_ML = time.time()
    iteration_time_ML = iteration_end_time_ML - iteration_start_time_ML

    iteration_times_ML.append(iteration_time_ML)
    z0 = val.clone().detach().requires_grad_(True)
    results = extract_ml_metrics_with_cc(
        z0, x_torch, fh_list, fhz, Gz0, norm,
        iteration_times_ML, rel_f_err, norm_fval, norm_grad,
        last_pts, prev_last_pts
    )

    if i in hparams["ML_image_indices"]:
        log_writer.add_image(f'ML_iter', z0, global_step=i, dataformats='HW')
    for j in range(len(y_diff)):
        log_writer.add_scalar(f'CC {j}', y_diff[j], i)

    print(f"Iteration {i}: {fh_list[0](z0)} - Time: {iteration_time_ML:.6f} seconds")

print(f"Overall time for all iterations: {sum(iteration_times_ML):.6f} seconds")
cumaltive_times_ML = [sum(iteration_times_ML[:i+1]) for i in range(len(iteration_times_ML))]

np.savez(ckpt_path_ML, **results)