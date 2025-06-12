import torch
import torch_scatter
from MGTomo import gridop

def armijo_linesearch_box(f, x: torch.tensor, d: torch.tensor, a=1., r=0.5, c=1e-3, verbose = True):
    fx = f(x)
    fx.backward()
    dgk = torch.sum(x.grad * d)
    
    assert dgk <= 0, 'd needs to be a descent direction (dgk = %.5e)' % dgk
    
    if dgk == 0.:
        return x, 0.
    
    while True:
        x_new = x + a * d
        
        mask0 = (torch.abs(x_new) <= 1e-5)
        x_new[mask0] = 0
        
        mask1 = torch.logical_and(x_new > 1, x_new <= 1+1e-5)
        x_new[mask1] = 1
        
        
        f_new = f(x_new)
        
        if f_new <= fx + a * c * dgk:
            if verbose:
                print('at a = ', a, 'f_new is <= and x_new.min() = ', x_new.min(), 'with #<0: ', sum(sum(i < 0 for i in x_new)), 'and x_new.max() = ', x_new.max(), 'with #>1: ', sum(sum(i > 1 for i in x_new)))
            if x_new.min() >= 0 and x_new.max() <= 1:
                break
        
        a *= r
        if a <= 1e-3:
            if verbose:
                print('Armijo step too small, a = 0', 'x_new.min() = ', x_new.min(), ' x_new.argmin()' , (x_new==torch.min(x_new)).nonzero() ,sum(sum(i < 0 for i in x_new)), 'indices < 0')
                print('Armijo step too small, a = 0', 'x_new.max() = ', x_new.max(), ' x_new.argmax()' , (x_new==torch.max(x_new)).nonzero() ,sum(sum(i > 0 for i in x_new)), 'indices > 1')
            return x, 0.
    
    return x_new, a


def armijo_linesearch(f, x: torch.tensor, d: torch.tensor, a=1., r = 0.5, c = 1e-3):
    fx = f(x)
    fx.backward(retain_graph=True)
    dgk = torch.sum(x.grad * d)
    
    assert dgk <= 0, 'd needs to be a descent direction (dgk = %.5e)' % dgk
    
    if dgk == 0.:
        return x, 0.
    
    while True:
        x_new = x + a * d
        
        f_new = f(x_new)
        
        if f_new <= fx + a * c * dgk:
            break
        
        a *= r
        if a <= 1e-7:
            print('Armijo step too small, a = 0')
            return x, 0.
    
    return x_new, a

def box_bounds(xh, xH, P_inf, lh, uh, P_nonzero= None):
    if P_nonzero == None:
        coarse_dim = xH.shape[0]
        P_nonzero = gridop.compute_nonzero_elements_of_P(coarse_dim)
    
    lH = torch.zeros_like(xH)
    uH = torch.zeros_like(xH)

    for col_coord, indices in P_nonzero.items():
        rows, cols = zip(*indices)
        
        rows = torch.tensor(rows)
        cols = torch.tensor(cols)
        
        diffs = xh[rows, cols]
        lmax = torch.max(lh[rows, cols] - diffs)
        umin = torch.min(uh[rows, cols] - diffs)
        
        lH[col_coord] = xH[col_coord] + lmax / P_inf
        uH[col_coord] = xH[col_coord] + umin / P_inf
    return lH, uH

def orthant_bounds(xh, xH, P_inf, lh, P_nonzero = None):
    if P_nonzero is None:
        coarse_dim = xH.shape[0]
        P_nonzero = gridop.compute_nonzero_elements_of_P(coarse_dim)
    
    lH = torch.zeros_like(xH)

    for col_coord, indices in P_nonzero.items():
        rows, cols = zip(*indices)
        
        rows = torch.tensor(rows)
        cols = torch.tensor(cols)
        
        diffs = xh[rows, cols]
        lmax = torch.max(lh[rows, cols] - diffs)
        
        lH[col_coord] = xH[col_coord] + lmax / P_inf
    return lH

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