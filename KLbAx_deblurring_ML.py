import MGBlurr.blurring as blur
import multilevel.functions as fcts
from multilevel.coarse_corrections import coarse_condition_bregman
from multilevel.optimize import armijo_linesearch
from multilevel.MultiGridOperator import MultigridOperator2D
from multilevel.multilevel import MultiLevelOptimizer
from multilevel.results import extract_ml_metrics_with_cc

from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.linalg import norm
import time
import datetime
import numpy as np
from PIL import Image

# --- Hyperparameters ---
hparams = {
    "image": "crater",
    "CC": "Bregman",
    "N": 511,
    "max_levels": 2,
    "maxIter": [1,10,10,32,64,128],
    "kernel_size": 27,
    "sigma": 5,
    "poisson_lbd": 15,
    "ML_iterate_count": 100,
    "ML_image_indices": range(0,105,5),
    "kappa": 0.49,
    "eps": 0.001,
    "bounds": "orthant",
    "log": True  # or False to disable logging
}

# --- Load image ---
image = Image.open('tycho-crater.png').convert('L')
image_data = np.array(image)
x_orig = resize(image_data, (hparams["N"],hparams["N"]), anti_aliasing = False)
x_torch = torch.tensor(x_orig, requires_grad = True)

# --- Build operators and right-hand sides ---
A = [blur.GaussianBlurOperator(hparams["N"], hparams["kernel_size"], hparams["sigma"])]
b = [torch.poisson(A[0](x_torch)*hparams["poisson_lbd"])/hparams["poisson_lbd"]]
assert torch.all(b[-1] >= 0)

fine_dim = hparams["N"]
input_sizes = [fine_dim]
for i in range(1, hparams["max_levels"]+1):
    coarse_dim = blur.reduce_dim(fine_dim)
    A.append(blur.GaussianBlurOperator(coarse_dim, hparams["kernel_size"], hparams["sigma"]))
    rhs = resize(b[-1].detach().numpy(), (coarse_dim, coarse_dim), anti_aliasing=False)
    b.append(torch.tensor(rhs, requires_grad=True))
    assert torch.all(b[-1] >= 0)
    input_sizes.append(coarse_dim)
    fine_dim = coarse_dim
hparams["input_sizes"] = input_sizes
print(f"Input sizes for each level: {hparams['input_sizes']}")

# --- Objective and step sizes ---
fh_list = [lambda x, bi=bi, Ai=Ai: fcts.kl_distance_rev(x, bi, Ai) for bi, Ai in zip(b, A)]
tau = [torch.reciprocal(norm(bi, ord = 1)) for bi in b]

# --- Multigrid kernel ---
kernel = torch.tensor([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=torch.float32)/16.0

# --- Multilevel optimizer setup ---
optimizer = MultiLevelOptimizer(
    fh_list=fh_list,
    tau=tau,
    kernel=kernel,
    hparams=hparams,
    BPGD=fcts.mirror_descent_IS,
    linesearch=armijo_linesearch,
    bounds=hparams["bounds"],
    CC=coarse_condition_bregman
)

# --- Logging setup ---
if hparams["log"]:
    log_dir = "runs/KLbAx_deblurring_ML/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
all_cc_activated = []

for i in range(hparams['ML_iterate_count']):
    iteration_start_time_ML = time.time()
    
    prev_last_pts = [p.clone() if p is not None else None for p in last_pts]  # <-- Add this line

    val, last_pts, y_diff = optimizer.run(z0, lh, last_pts=last_pts, y_diff=y_diff, l=0)
    iteration_end_time_ML = time.time()
    iteration_time_ML = iteration_end_time_ML - iteration_start_time_ML

    iteration_times_ML.append(iteration_time_ML)
    z0 = val.clone().detach().requires_grad_(True)
    results = extract_ml_metrics_with_cc(
        z0, x_torch, fh_list, fhz, Gz0, norm,
        iteration_times_ML, rel_f_err, norm_fval, norm_grad,
        last_pts, prev_last_pts, cc_levels=[0, 1]
    )

    all_cc_activated.append(results['cc_activated'])

    if hparams["log"]:
        if i in hparams["ML_image_indices"]:
            log_writer.add_image(f'ML_iter', z0, global_step=i, dataformats='HW')
        for j in range(len(y_diff)):
            log_writer.add_scalar(f'CC {j}', y_diff[j], i)

    print(f"Iteration {i}: {fh_list[0](z0)} - Time: {iteration_time_ML:.6f} seconds")

print(f"Overall time for all iterations: {sum(iteration_times_ML):.6f} seconds")
cumaltive_times_ML = [sum(iteration_times_ML[:i+1]) for i in range(len(iteration_times_ML))]

##########
results['cc_activated'] = np.array(all_cc_activated)
if hparams["log"]:
    np.savez(ckpt_path_ML, **results)
    log_writer.add_figure("normalised function value vs. CPU Time", plt.gcf())
    log_writer.close()
##########