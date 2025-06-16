import MGBlurr.blurring as blur
import multilevel.functions as fcts
from multilevel.gridop import RBox as R, PBox as P
from torch.utils.tensorboard import SummaryWriter

from skimage.transform import resize
import torch
from torch.linalg import norm
import numpy as np
from PIL import Image
import time
import datetime

# --- Hyperparameters ---
hparams = {
    "image": "crater",
    "N": 511,
    "kernel_size": 15,
    "sigma": 1.5,
    "poisson_lbd": 1000,
    "SL_iterate_count": 60,
    "SL_image_indices": range(0,60,10),
}

# --- Load image ---
image = Image.open('tycho-crater.png').convert('L')
image_data = np.array(image)
x_orig = resize(image_data, (hparams["N"], hparams["N"]), anti_aliasing=False)
x_torch = torch.tensor(x_orig, requires_grad=True)

# --- Build blur operator and right-hand side ---
A = blur.GaussianBlurOperator(hparams["N"], hparams["kernel_size"], hparams["sigma"])
b = torch.poisson(A(x_torch) * hparams["poisson_lbd"]) / hparams["poisson_lbd"]
assert torch.all(b >= 0)

fh = lambda x: fcts.kl_distance_rev(x, b, A)
tau = torch.reciprocal(norm(b, ord=1))

# --- Logging setup ---
log_dir = "runs/KLbAx_deblurring_SL/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = SummaryWriter(log_dir)
hparams["tau"] = tau
log_writer.add_text("hparams", str(hparams))
ckpt_path_SL = f"{log_writer.log_dir}/SL"

# --- Initialization ---
w0 = torch.ones(hparams["N"], hparams["N"], requires_grad=True) * 0.5
lh = torch.zeros_like(w0)

fhw = fh(w0)
w0.retain_grad()
fhw.backward(retain_graph=True)
Gw0 = norm(w0.grad, 'fro')

rel_f_err_SL = [(norm(w0 - x_torch, 'fro') / norm(w0, 'fro')).item()]
norm_fval_SL = [torch.tensor(1.)]
iteration_times_SL = [0]
norm_grad_SL = [torch.tensor(1.)]

# --- Single-level optimization loop ---
for i in range(hparams['SL_iterate_count']):
    iteration_start_time_SL = time.time()
    val = fcts.mirror_descent_IS(fh, w0, tau, lh)
    iteration_end_time_SL = time.time()
    iteration_time_SL = iteration_end_time_SL - iteration_start_time_SL

    iteration_times_SL.append(iteration_time_SL)
    w0 = val.clone().detach().requires_grad_(True)

    fval = fh(w0)
    norm_fval_SL.append((fval / fhw).item())
    log_writer.add_scalar("SL_normalised_value", norm_fval_SL[-1], i)
    fval.backward(retain_graph=True)
    norm_grad_SL.append((norm(w0.grad, 'fro') / Gw0).item())
    log_writer.add_scalar("SL_normalised_gradient", norm_grad_SL[-1], i)
    w0.grad = None
    rel_f_err_SL.append((norm(w0 - x_torch, 'fro') / norm(w0, 'fro')).item())

    if i in hparams["SL_image_indices"]:
        log_writer.add_image(f'SL_iter', w0, global_step=i, dataformats='HW')

    print(f"Iteration {i}: {fh(w0)} - Time: {iteration_time_SL:.6f} seconds")

print(f"Overall time for all iterations: {sum(iteration_times_SL):.6f} seconds")
cumaltive_times_SL = [sum(iteration_times_SL[:i+1]) for i in range(len(iteration_times_SL))]

# --- Save results ---
np.savez(
    ckpt_path_SL,
    iteration_times_SL=iteration_times_SL,
    norm_fval_SL=norm_fval_SL,
    norm_grad_SL=norm_grad_SL,
    rel_f_err_SL=rel_f_err_SL,
    last_iterate_SL=w0.detach().numpy()
)