import torch
from torch.linalg import norm
import multilevel.functions as fcts
import numpy as np


def coarse_condition(R, y, grad_y, kappa, eta, y_last = None):
    """
    Euclidean coarse correction condition for multilevel methods.
    kappa checks first order optimality.
    eta checks if the iterate has changed enough.
    Returns (condition_met: bool, y_diff_norm: float).
    """
    with torch.no_grad():
        gcond = (norm(R(grad_y), 'fro') >= kappa * norm(grad_y, 'fro'))
        if gcond:
            if y_last is not None:
                y_diff_norm = norm(y_last - y, 'fro')
                y_norm = norm(y, 'fro')
                return (y_diff_norm >= eta * y_norm), y_diff_norm.item()
            return True, np.nan
        return False, np.nan
        
def coarse_condition_bregman(R, y, grad_y, kappa, eta, y_last = None):
    """
    Bregman coarse correction condition for multilevel methods.
    kappa checks first order optimality.
    eta checks if the iterate has changed enough.
    Returns (condition_met: bool, y_diff_norm: float).
    """
    with torch.no_grad():
        gcond = (norm(R(grad_y), 'fro') >= kappa * norm(grad_y, 'fro'))
        if gcond:
            if y_last is not None:
                y_diff_norm = fcts.kl_distance_no_matrix(y, y_last)
                return (y_diff_norm >= eta), y_diff_norm.item()
            return True, np.nan
        return False, np.nan