import torch
from torch.linalg import norm

def extract_ml_metrics(z0, x_torch, fh_list, fhz, Gz0, norm, 
                      iteration_times_ML, rel_f_err, norm_fval, norm_grad):
    """
    Updates and returns a dictionary of ML metrics for logging and plotting.
    """
    rel_f_err.append((norm(z0 - x_torch, 'fro') / norm(z0, 'fro')).item())
    fval = fh_list[0](z0)
    norm_fval.append((fval / fhz).item())
    fval.backward(retain_graph=True)
    norm_grad.append((norm(z0.grad, 'fro') / Gz0).item())
    z0.grad = None
    return {
        "iteration_times_ML": iteration_times_ML,
        "norm_fval_ML": norm_fval,
        "norm_grad_ML": norm_grad,
        "rel_f_err_ML": rel_f_err,
        "last_iterate_ML": z0.detach().cpu().numpy()
    }

def extract_ml_metrics_with_cc(z0, x_torch, fh_list, fhz, Gz0, norm, 
                              iteration_times_ML, rel_f_err, norm_fval, norm_grad,
                              last_pts, prev_last_pts, cc_levels=None):
    """
    Updates and returns a dictionary of ML metrics, including CC activation status.
    Only checks CC activation for levels in cc_levels (default: all except coarsest).
    """
    rel_f_err.append((norm(z0 - x_torch, 'fro') / norm(z0, 'fro')).item())
    fval = fh_list[0](z0)
    norm_fval.append((fval / fhz).item())
    fval.backward(retain_graph=True)
    norm_grad.append((norm(z0.grad, 'fro') / Gz0).item())
    z0.grad = None

    # By default, skip the coarsest level (last index)
    if cc_levels is None:
        cc_levels = range(len(last_pts) - 1)

    cc_activated = []
    for l in cc_levels:
        prev = prev_last_pts[l]
        curr = last_pts[l]
        if prev is None and curr is not None:
            cc_activated.append(True)
        elif prev is not None and curr is not None:
            cc_activated.append(not torch.allclose(prev, curr))
        else:
            cc_activated.append(False)

    return {
        "iteration_times_ML": iteration_times_ML,
        "norm_fval_ML": norm_fval,
        "norm_grad_ML": norm_grad,
        "rel_f_err_ML": rel_f_err,
        "last_iterate_ML": z0.detach().cpu().numpy(),
        "cc_activated": cc_activated
    }