import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import astra
import scipy.sparse as sp

np.int = np.int32

# --- D-optimal design objective ---
def dopt_objective(x, Ht):
    X = torch.diag(x)
    FIM = Ht @ X @ Ht.T
    sign, logdet = torch.linalg.slogdet(FIM)
    return -logdet

def stable_netwon_equation(c, theta, eq_constraint):
    safe_c_theta = np.clip(np.abs(c + theta), 1e-10, None)
    return np.sum(np.exp(-np.log(safe_c_theta))) - eq_constraint

def stable_netwon_derivative(c, theta, eq_constraint):
    safe_c_theta = np.clip(np.abs(c + theta), 1e-10, None)
    return -np.sum(np.exp(-2 * np.log(safe_c_theta)))

# --- Hyperparameters ---
fine_dim = 63
num_proj = 120
angles = np.linspace(0, np.pi, num_proj, endpoint=False)
det_count = fine_dim

# --- Create fine-level geometry and operator ---
geometry = astra.create_proj_geom('parallel', 1.0, det_count, angles)
vol_geom = astra.create_vol_geom(fine_dim, fine_dim)
proj_id = astra.create_projector('line', geometry, vol_geom)
matrix_id = astra.projector.matrix(proj_id)
H = sp.csr_matrix(astra.matrix.get(matrix_id))
Ht = torch.tensor(H.T.toarray(), dtype=torch.float32)

# --- Initialization ---
torch.manual_seed(0)

n = H.shape[0]
# Perturbed uniform initialization
x = torch.ones(n, dtype=torch.float32) / n
x += 0.001 * torch.rand(n, dtype=torch.float32)
x = torch.clamp(x, min=1e-8)
x = x / x.sum()
x.requires_grad_(True)
lh = torch.zeros_like(x)
uh = torch.ones_like(x)

# --- TensorBoard Logging Setup ---
log_dir = f"runs/doptimaldesign_single/{datetime.datetime.now():%Y%m%d-%H%M%S}"
log_writer = SummaryWriter(log_dir)
ckpt_path_SL = f"{log_writer.log_dir}/SL"

# --- Optimization loop ---
from scipy.optimize import root_scalar

max_iter = 50
solution_vec = [x.detach().numpy()]
fx0 = dopt_objective(x, Ht).item()
iteration_times = [0]
norm_fval_SL = [1.]
last_iterate_SL = None

for i in range(max_iter):
    start_time = time.time()
    # Compute gradient
    loss = dopt_objective(x, Ht)
    loss.backward(retain_graph=True)
    grad = x.grad.detach().clone()
    x.grad = None

    c = grad + 1 / x
    S = x.sum().item()
    theta0 = 100
    bracket = [-(torch.min(c)).item(), 1e5]
    c_np = c.detach().cpu().numpy()

    # Root finding (no grad)
    res = root_scalar(
        stable_netwon_equation,
        bracket=bracket,
        args=(c_np, S),
        x0=theta0,
        fprime=stable_netwon_derivative,
        method="bisect"
    )
    theta_new = res.root
    # Update x
    x = 1 / (c + theta_new)
    x = torch.clamp(x, min=1e-8)
    x = x / x.sum()
    x = x.detach().clone().requires_grad_()

    end_time = time.time()
    iteration_times.append(end_time - start_time)
    solution_vec.append(x.detach().numpy())
    fx = dopt_objective(x, Ht).item()
    # Save normalized function value (relative to first value)
    norm_fval_SL.append(fx / fx0)
    print(f"Iteration {i}: {fx} - Time: {iteration_times[-1]:.6f} seconds")
    log_writer.add_scalar("Doptimal_value", fx, i)

# Save last iterate
last_iterate_SL = x.detach().numpy()

print(f"Overall time for all iterations: {sum(iteration_times):.6f} seconds")
cumaltive_times = [sum(iteration_times[:i+1]) for i in range(len(iteration_times))]

# --- Save results ---
np.savez(
    ckpt_path_SL,
    iteration_times_SL=np.array(iteration_times),
    norm_fval_SL=np.array(norm_fval_SL),
    last_iterate_SL=last_iterate_SL
)