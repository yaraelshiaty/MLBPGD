import torch
import numpy as np

def myexp(x):
    return torch.exp(torch.minimum(torch.maximum(x, torch.tensor(-50.0)),torch.tensor(50.0)))

def mylog(x, delta=torch.tensor(1e-10)):
    return torch.log(torch.maximum(x, delta))

#umweg bc torch.divide doesn't have where keyword argument
def mydiv(x, y):
    mask = (y != 0)
    out = torch.ones_like(x)
    out[mask] = torch.divide(x[mask], y[mask])
    return out

def mysub(x, y):
    return mylog(myexp(x)*myexp(-y))