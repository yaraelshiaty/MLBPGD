import MGTomo.model as mgmodel
import MGTomo.tomoprojection as mgproj
from MGTomo.utils import mylog
import multilevel.functions as fcts

import time
import numpy as np
import torch
from torch.linalg import norm
from torch.utils.tensorboard import SummaryWriter

import datetime
from PIL import Image

# --- Hyperparameters ---
hparams = {
    "image": "walnut",
    "N": 1023,
    "num_angels0": 200,
    "SL_iterate_count": 150,
    "SL_image_indices": range(0,150,10),
}

# --- Load image ---
image = Image.open('walnut.png').convert('L')
image_np = np.array(image)
x_orig = np.array(image.resize((hparams["N"],hparams["N"])))/255
x_torch = torch.tensor(x_orig, requires_grad=True)

# --- Build operators and right-hand sides ---
model = mgmodel.astra_model(hparams["N"],{'mode' : 'line', 'num_angles' : hparams["num_angels0"], 'level_decrease' : 1})
fine_dim = model.dim
A = mgproj.TomoTorch(model.proj_factory(fine_dim))
b = A(x_torch)

fh = lambda x: fcts.kl_distance(x, A, b)
tau = torch.reciprocal(A.sumnorm_opt())

# --- Logging setup ---
log_dir = "runs/KLAxb_reconstruction_SL/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = SummaryWriter(log_dir)
hparams["tau"] = tau
log_writer.add_text("hparams", str(hparams))
ckpt_path_SL = f"{log_writer.log_dir}/SL"

# --- Initialization ---
w0 = torch.ones(hparams["N"], hparams["N"], requires_grad=True) * 0.5
lh = torch.zeros_like(w0)
uh = torch.ones_like(w0)
logv_new = mylog((w0 - lh)) - mylog((uh - w0))

fhw = fh(w0)
w0.retain_grad()
fhw.backward(retain_graph=True)
Gw0 = norm(w0.grad, 'fro')

rel_f_err_SL = [(norm(w0 - x_torch, 'fro')/norm(w0, 'fro')).item()]
norm_fval_SL = [torch.tensor(1.)]
iteration_times_SL = [0]
norm_grad_SL = [torch.tensor(1.)]

# --- Single-level optimization loop ---
for i in range(hparams['SL_iterate_count']):
    iteration_start_time_SL = time.time()
    val, logv_new = fcts.BSMART_general(fh, w0, tau, lh, uh, logv_new)
    iteration_end_time_SL = time.time()
    iteration_time_SL = iteration_end_time_SL - iteration_start_time_SL

    iteration_times_SL.append(iteration_time_SL)
    w0 = val.clone().detach().requires_grad_(True)

    rel_f_err_SL.append((norm(w0-x_torch, 'fro')/norm(w0, 'fro')).item())
    fval = fh(w0)
    norm_fval_SL.append((fval/fhw).item())
    log_writer.add_scalar("SL_normalised_value", norm_fval_SL[-1], i)
    fval.backward(retain_graph=True)
    norm_grad_SL.append((norm(w0.grad, 'fro')/Gw0).item())
    log_writer.add_scalar("SL_normalised_gradient", norm_grad_SL[-1], i)
    w0.grad = None

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