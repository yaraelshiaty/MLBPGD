import torch
import numpy as np

def myexp(x):
    return torch.exp(torch.minimum(torch.maximum(x, torch.tensor(-80.0)),torch.tensor(80.0)))

def mylog(x, delta=torch.tensor(1e-20)):
    #return torch.log(torch.maximum(x, delta))
    mask = x < delta
    x = torch.where(mask, delta, x)
    return torch.log(x + delta)

#umweg bc torch.divide doesn't have where keyword argument
def mydiv(x, y):
    mask = (y != 0)
    out = torch.ones_like(x)
    #assert x[mask].dtype == y[mask].dtype, x[mask].dtype
    out[mask] = torch.divide(x[mask], y[mask])
    return out

def mysub(x, y):
    return mylog(myexp(x)*myexp(-y))