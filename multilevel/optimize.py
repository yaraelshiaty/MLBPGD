import torch
import torch_scatter
from multilevel import gridop

def armijo_linesearch(f, x: torch.Tensor, d: torch.Tensor, dfx=None, a=1.0, r=0.5, c=1e-3, grad_f=None):
    """
    Armijo linesearch for function f at point x along direction d.
    If grad_f is provided, uses grad_f(x) to compute the gradient; otherwise, computes it via autograd.
    dfx: directional derivative at x along d (optional, will be computed if not provided)
    """
    fx = f(x)
    if dfx is None:
        if grad_f is not None:
            grad = grad_f(x)
        else:
            x = x.clone().detach().requires_grad_(True)
            fx_temp = f(x)
            fx_temp.backward()
            grad = x.grad.detach()
        dfx = torch.sum(grad * d)
        # If grad is numpy, convert to torch
        if not isinstance(dfx, torch.Tensor):
            dfx = torch.tensor(dfx)
        dfx = dfx.item() if hasattr(dfx, 'item') else float(dfx)

    if torch.all(dfx == 0.):
        return x, 0.

    while True:
        x_new = x + a * d
        f_new = f(x_new)
        if torch.all(f_new <= fx + a * c * dfx):
            break
        a *= r
        if a <= 1e-7:
            print('Armijo step too small, a = 0')
            return x, 0.
    return x_new,a

def orthant_bounds_optimized(xh, xH, P_inf, lh, uh=None, P_nonzero=None):
    coarse_dim = xH.shape[0]
    if P_nonzero is None:
        P_nonzero = gridop.compute_nonzero_elements_of_P(coarse_dim)
    lH = torch.zeros_like(xH)

    all_rows = []
    all_cols = []
    col_coords_flat = []
    col_coords = []

    for (x,y), indices in P_nonzero.items():
        rows, cols = zip(*indices)
        all_rows.extend(rows)
        all_cols.extend(cols)
        col_coords_flat.extend([x*coarse_dim + y]*len(rows))
        col_coords.append((x,y))

    all_rows_tensor = torch.tensor(all_rows)
    all_cols_tensor = torch.tensor(all_cols)
    all_col_coords = torch.tensor(col_coords_flat)

    rowsH_tensor, colsH_tensor = torch.tensor(col_coords).unbind(dim=1)

    diffs = xh[all_rows_tensor, all_cols_tensor]
  
    lmax = torch_scatter.scatter_max(lh[all_rows_tensor, all_cols_tensor] - diffs, all_col_coords, dim = 0)[0]
    lH[rowsH_tensor, colsH_tensor] = xH[rowsH_tensor, colsH_tensor] + lmax / P_inf

    return lH, None

def box_bounds_optimized(xh, xH, P_inf, lh, uh, P_nonzero=None):
    coarse_dim = xH.shape[0]
    if P_nonzero is None:
        P_nonzero = gridop.compute_nonzero_elements_of_P(coarse_dim)
    lH = torch.zeros_like(xH)
    uH = torch.zeros_like(xH)

    all_rows = []
    all_cols = []
    col_coords_flat = []
    col_coords = []

    for (x,y), indices in P_nonzero.items():
        rows, cols = zip(*indices)
        all_rows.extend(rows)
        all_cols.extend(cols)
        col_coords_flat.extend([x*coarse_dim + y]*len(rows))
        col_coords.append((x,y))

    all_rows_tensor = torch.tensor(all_rows)
    all_cols_tensor = torch.tensor(all_cols)
    all_col_coords = torch.tensor(col_coords_flat)

    rowsH_tensor, colsH_tensor = torch.tensor(col_coords).unbind(dim=1)

    diffs = xh[all_rows_tensor, all_cols_tensor]
  
    lmax = torch_scatter.scatter_max(lh[all_rows_tensor, all_cols_tensor] - diffs, all_col_coords, dim = 0)[0]
    umin = torch_scatter.scatter_min(uh[all_rows_tensor, all_cols_tensor] - diffs, all_col_coords, dim = 0)[0]
    lH[rowsH_tensor, colsH_tensor] = xH[rowsH_tensor, colsH_tensor] + lmax / P_inf
    uH[rowsH_tensor, colsH_tensor] = xH[rowsH_tensor, colsH_tensor] + umin / P_inf

    return lH, uH