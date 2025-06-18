import multilevel.functions as fcts
from multilevel.optimize import armijo_linesearch
from multilevel.multilevel import MultiLevelOptimizer

import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import datetime

import astra
import scipy.sparse as sp
from functools import partial

np.int = np.int32

def coarse_condition_torch(y, grad_y=None, kappa=None, eta=0.49, y_last=None):
    """
    Torch-compatible coarse correction condition.
    Returns (bool, y_diff_norm) for compatibility with CoarseCorrectionHandler.
    """
    if y_last is not None:
        y_diff_norm = torch.norm(y_last - y)
        y_norm = torch.norm(y)
        cond = (y_diff_norm >= eta * y_norm)
        return bool(cond.item()), y_diff_norm.item()
    else:
        return True, float('nan')

def box_bounds_flat(xh, xH, P_inf, lh, uh, P_nonzero=None):
    """
    Flat version: xh, xH, lh, uh are 1D tensors; P_nonzero maps input flat indices to lists of output flat indices.
    """
    if P_nonzero is None:
        coarse_dim = xH.shape[0]
        raise ValueError("P_nonzero must be provided for flat version.")

    lH = torch.zeros_like(xH)
    uH = torch.zeros_like(xH)

    for input_idx, output_indices in P_nonzero.items():
        rows = torch.tensor(output_indices)
        diffs = xh[rows]
        lmax = torch.max(lh[rows] - diffs)
        umin = torch.min(uh[rows] - diffs)
        lH[input_idx] = xH[input_idx] + lmax / P_inf
        uH[input_idx] = xH[input_idx] + umin / P_inf
    return lH, uH

def stable_newton_equation_torch(c, theta, eq_constraint):
    safe_c_theta = torch.clamp(torch.abs(c + theta), min=1e-10)
    return torch.sum(torch.exp(-torch.log(safe_c_theta))) - eq_constraint

def stable_newton_derivative_torch(c, theta, eq_constraint):
    theta = theta.clone().detach().requires_grad_(True)
    val = stable_newton_equation_torch(c, theta, eq_constraint)
    val.backward()
    return theta.grad.item()  # Returns the derivative as a Python float

def stable_netwon_equation(c, theta, eq_constraint):
    safe_c_theta = np.clip(np.abs(c + theta), 1e-10, None)  # Avoid log(0)
    return np.sum(np.exp(-np.log(safe_c_theta))) - eq_constraint  # Uses log instead of division

def stable_netwon_derivative(c, theta, eq_constraint):
    safe_c_theta = np.clip(np.abs(c + theta), 1e-10, None)  # Avoid log(0)
    return -np.sum(np.exp(-2 * np.log(safe_c_theta)))  # Uses log instead of division

# --- Hyperparameters ---
fine_dim = 63
max_levels = 1
maxIter = [1, 10, 10]
num_proj = 120
angles = np.linspace(0, np.pi, num_proj, endpoint=False)  # Projection angles
det_count = fine_dim

# --- Create fine-level geometry and operator ---
geometry = astra.create_proj_geom('parallel', 1.0, det_count, angles)
vol_geom = astra.create_vol_geom(fine_dim, fine_dim)
proj_id = astra.create_projector('line', geometry, vol_geom)
matrix_id = astra.projector.matrix(proj_id)
H = [sp.csr_matrix(astra.matrix.get(matrix_id))]

# --- Multilevel operator setup ---
input_sizes = [fine_dim]
for i in range(1, max_levels + 1):
    coarse_dim = fine_dim // 2
    det_count = coarse_dim
    angles = np.linspace(0, np.pi, num_proj, endpoint=False)
    geometry = astra.create_proj_geom('parallel', 1.0, det_count, angles)
    vol_geom = astra.create_vol_geom(coarse_dim, coarse_dim)
    proj_id = astra.create_projector('line', geometry, vol_geom)
    matrix_id = astra.projector.matrix(proj_id)
    Hnew = astra.matrix.get(matrix_id)
    H.append(sp.csr_matrix(Hnew))
    input_sizes.append(coarse_dim)
    fine_dim=coarse_dim
print(f"Input sizes for each level: {input_sizes}")

# Prepare fh_list for each level
H_torch = [torch.tensor((Hl.T).toarray(), dtype=torch.float32) for Hl in H]
fh_list = [lambda x, Ht=Ht: fcts.Doptdesign(x, Ht) for Ht in H_torch]

# Step sizes
tau = [torch.tensor(1.0) for _ in H]

# --- Multigrid kernel ---
kernel = torch.tensor([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=torch.float32)/16.0

# --- Hyperparameters dict ---
hparams = {
    "max_levels": max_levels,
    "maxIter": maxIter,
    "input_sizes": input_sizes,
    "bounds": "box"
}

# --- MultiLevelOptimizer instance ---
# Define your root equation and derivative
root_eq = stable_netwon_equation
root_deriv = stable_netwon_derivative

custom_linesearch = partial(armijo_linesearch, a=1.0, r=0.25, c=1e-3)

# Create a partial function with context pre-filled
custom_bpgd = partial(
    fcts.bpgd_newton_solver,
    context={
        "root_equation": root_eq,
        "root_derivative": root_deriv,
    }
)

def custom_bpgd_with_context(f, x, tau, lh, uh=None, *args, **context):
    context.setdefault("root_equation", root_eq)
    context.setdefault("root_derivative", root_deriv)
    return fcts.bpgd_newton_solver(f, x, tau, lh, uh, *args, **context)

optimizer = MultiLevelOptimizer(
    fh_list=fh_list,
    tau=tau,
    kernel=kernel,
    hparams=hparams,
    CC=coarse_condition_torch,
    linesearch=custom_linesearch,
    BPGD=custom_bpgd_with_context,
    bounds="box"
)

from multilevel.MultiGridOperator import MultigridOperatorRowsOnly2D

mgop = MultigridOperatorRowsOnly2D(width=120)
optimizer.P = mgop.P
optimizer.R = mgop.R
optimizer.mgop = mgop
optimizer.P_inf = mgop.norm_infty_P()

# Overwrite P_nonzero with the 1D version for each level
optimizer.P_nonzero = [
    mgop.compute_nonzero_elements_of_P_rows_only_flat(input_sizes[l+1])
    for l in range(len(input_sizes) - 1)
]
optimizer.compute_bounds = box_bounds_flat

# --- Initialization ---
n = H[0].shape[0]
# # Perturbed uniform initialization
x = torch.ones(n, dtype=torch.float32) / n
# x += 0.001 * torch.rand(n, dtype=torch.float32)
# x = torch.clamp(x, min=1e-8)
# x = x / x.sum()
x.requires_grad_(True)
lh = torch.zeros_like(x)
uh = torch.ones_like(x)
last_pts = [None] * (max_levels + 1)
y_diff = torch.zeros(max_levels + 1)

fval = []
fhz = fh_list[0](x)
fval.append(fhz.item())
fhz.backward(retain_graph=True)
Gz0 = abs(x.grad)
x.grad = None

norm_grad = []
norm_grad.append(torch.tensor(1.))

iteration_times_ML = []
iteration_times_ML.append(0)

solution_vec_ML = [x.detach().numpy()]

# --- TensorBoard Logging Setup ---
log_dir = "runs/doptimaldesign_ML/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = SummaryWriter(log_dir)
hparams["tau"] = tau
log_writer.add_text("hparams", str(hparams))
ckpt_path_ML = f"{log_writer.log_dir}/ML"

for i in range(10):
    iteration_start_time_ML = time.time()
    x, last_pts, y_diff = optimizer.run(x, lh, uh, last_pts=last_pts, y_diff=y_diff, l=0)
    iteration_end_time_ML = time.time()
    iteration_time_ML = iteration_end_time_ML - iteration_start_time_ML
    iteration_times_ML.append(iteration_time_ML)
    solution_vec_ML.append(x.detach().numpy())
    fx = fh_list[0](x).item()
    fval.append(fx)

    print(f"Iteration {i}: {fx} - Time: {iteration_time_ML:.6f} seconds")

    # --- Logging ---
    log_writer.add_scalar("ML_normalised_value", fx, i)

overall_time_ML = sum(iteration_times_ML)
print(f"Overall time for all iterations: {overall_time_ML:.6f} seconds")
cumaltive_times_ML = [sum(iteration_times_ML[:i+1]) for i in range(len(iteration_times_ML))]


np.savez(
    ckpt_path_ML,
    solution_vec_ML=np.array(solution_vec_ML, dtype=object),
    iteration_times_ML=np.array(iteration_times_ML),
    fval=np.array(fval, dtype=object)
)

log_writer.flush()
log_writer.close()