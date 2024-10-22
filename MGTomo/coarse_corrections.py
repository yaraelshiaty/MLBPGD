import torch
from torch.linalg import norm, matrix_norm
from MGTomo.gridop import RBox as R


def coarse_condition_v2(y, grad_y, kappa, eta, y_last = None):
    with torch.no_grad():
        gcond = (norm(R(grad_y), 'fro') >= kappa * norm(grad_y, 'fro'))
        if gcond:
            if y_last is not None:
                y_diff_norm = norm(y_last - y, 'fro')
                y_norm = norm(y, 'fro')
                return (y_diff_norm >= eta * y_norm)
            return True
        else:
            return False
        
def coarse_condition_CPU(y, grad_y, kappa, eta, y_last = None):
    with torch.no_grad():
        gcond = (matrix_norm(R(grad_y)) >= kappa * matrix_norm(grad_y))
        if gcond:
            if y_last is not None:
                y_diff_norm = matrix_norm(y_last - y)
                y_norm = matrix_norm(y)
                return (y_diff_norm >= eta * y_norm)
            return True
        else:
            return False