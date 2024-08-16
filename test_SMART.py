import MGTomo.model as mgmodel
import numpy as np
import MGTomo.tomoprojection as mgproj
from MGTomo.utils import myexp, mylog, mydiv
import MGTomo.functions as fcts
from skimage import data
from skimage.transform import resize
from MGTomo.optimize import armijo_linesearch

from MGTomo.gridop import P,R

import torch
from torch.func import grad

from torch.linalg import matrix_norm

import matplotlib.pyplot as plt 


max_levels = 1
maxIter = [5,5]

N = 511
# load image
x_orig = data.camera()
x_orig = resize(x_orig, (N,N), anti_aliasing = False)

x_torch = torch.tensor(x_orig, requires_grad = True)

model = mgmodel.astra_model(N,{'mode' : 'line', 'num_angles' : 200, 'level_decrease' : 1})
fine_dim = model.dim
A = [mgproj.TomoTorch(model.proj_factory(fine_dim))]
b = [A[0](x_torch)]
level = {int(np.sqrt(A[0].shape[1])): 0}

for i in range(1,max_levels+1):
    coarse_dim = model.reduce_dim(fine_dim)
    A.append(mgproj.TomoTorch(model.proj_factory(coarse_dim)))
    b.append(torch.from_numpy(model.reduce_rhs(b[-1].detach().numpy(), fine_dim, coarse_dim)))
    level.update({int(np.sqrt(A[i].shape[1])): i})
    fine_dim=coarse_dim
    
c0 = 56.0952

fh = lambda x: fcts.kl_distance(x, A[0], b[0])

tau0 = 0.5 * 1/c0

z0 = torch.rand(N, N, requires_grad = True)

z0 = torch.rand(N, N, requires_grad = True)
i = 0
while fh(z0) >= 0.5:
    val = fcts.BSMART(fh, z0, tau0)
    z0 = val.clone().detach().requires_grad_(True)
    
    assert z0.grad is None
    assert z0.max() <= 1
    print(i, ': ', fh(z0))
    i += 1
    
plt.imshow(z0.detach().numpy(), cmap = 'gray')