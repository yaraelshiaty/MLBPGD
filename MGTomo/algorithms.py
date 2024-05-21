import numpy as np
from math import fabs
import MGTomo.grid as grid
from MGTomo.model import astra_model

def adjust_bounds(xh0: np.ndarray, lh: np.ndarray, uh: np.ndarray, model: astra_model):
  lh *= (1./model.g_infnorm)
  uh *= (1./model.g_infnorm)
  lh += xh0
  uh += xh0
  return lh,uh

def dict_update(d: dict,u: dict):
  d.update(u)
  return d

def dict_create(model: astra_model,values: list):
  d = dict()
  for i in range(len(values)):
    d[model.reduce_dim(model.dim,i)] = values[i]
  return d

def multigrid_cycle(h: int,x: np.ndarray, v: np.ndarray = 0., options: dict = None):
  cycles = options['cycles'][0]
  model: astra_model = options['model']

  f = np.inf
  for i in range(cycles):
    x0, f0 = x.copy(), f
    x,f = multigrid_descent(h,x,v,options)
    if np.amax(np.abs(x0-x)) < 1e-5 and fabs(f-f0) < 1e-5:
      return x,f

  options.update({
    'min_dim': model.reduce_dim(model.dim, 0),
    'maxiter': dict_create(model, [options['cycles'][1]])
  })
  x,f = multigrid_descent(h, x, v, options)
  return x,f

def multigrid_descent(h: int,x: np.ndarray, v: np.ndarray = 0., options: dict = None):

  model: astra_model  = options['model']
  maxiter = options['maxiter'][h]
  min_dim, max_dim = options['min_dim'], model.dim
  fobj = options['fobj']
  optimize = options['optimize']
  coarse_correction = options['correction']
  l,u = options['l'], options['u']

  if h > min_dim:

    hh = model.reduce_dim(h)
    vh, xh0 = model.first_order_coherence(lambda z: fobj(h, z, v), lambda z: fobj(hh, z), x)
    lh, uh = grid.get_bounds(x,model.g_stencil,l,u)
    lh, uh = adjust_bounds(xh0, lh, uh,model)

    xh, _ = multigrid_descent(hh, xh0.copy(), vh, dict_update(options.copy(),{'l' : lh,'u' : uh}))
    d = grid.coarse2fine(xh - xh0, model.g_stencil)*model.g_infnorm
    x = coarse_correction(lambda z : fobj(h,z,v),x,d,dict_update(options.copy(),{'h':h,'i':0}))

  x = optimize(lambda z : fobj(h,z,v),x,maxiter,dict_update(options.copy(),{'h':h}))

  f, _ = fobj(h, x, v)
  return x, f

def gradient_descent(h: int, x: np.ndarray, v: np.ndarray, options: dict = None):

  model: astra_model  = options['model']
  maxiter = options['maxiter'][h]
  max_rec = options['max_rec'][h]
  rec_dist = options['rec_dist'][h]

  min_dim, max_dim = options['min_dim'], model.dim
  fobj = options['fobj']
  line_search = options['line_search']
  coarse_correction = options['correction']
  l,u = options['l'], options['u']

  hh = model.reduce_dim(h)
  xr = x.copy()
  rec_i, rec_last = 0, np.inf
  for i in range(maxiter):
    f, g = fobj(h,x,v)
    gh = grid.fine2coarse(g,model.g_stencil)

    if np.linalg.norm(gh,'fro') > options['epsilon'] and \
       np.linalg.norm(gh,'fro') > options['kappa']*np.linalg.norm(g,'fro') and \
       ((np.linalg.norm(xr-x,'fro')**2)/x.size) > options['epsilon'] and \
       h > min_dim and fabs(i - rec_last) > rec_dist and rec_i < max_rec:

      vh, xh0 = model.first_order_coherence(lambda z: fobj(h, z, v), lambda z: fobj(hh, z), x)
      lh, uh = grid.get_bounds(x, model.g_stencil, l, u)
      lh, uh = adjust_bounds(xh0, lh, uh,model)

      xh, _ = gradient_descent(hh, xh0.copy(), vh, dict_update(options.copy(),{'l': lh, 'u': uh}))
      d = grid.coarse2fine(xh - xh0, model.g_stencil) * model.g_sigma
      x = coarse_correction(lambda z: fobj(h, z, v), x, d, dict_update(options.copy(),{'h':h,'i':i}))

      xr = x.copy()
      rec_i += 1
      rec_last = i

    x,a = line_search(lambda z: fobj(h,z,v),x,dict_update(options.copy(),{'h':h,'i':i}))
    if a < 1e-12:
      break

  f, _ = fobj(h, x, v)
  return x, f
