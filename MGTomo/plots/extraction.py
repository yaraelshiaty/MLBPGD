import numpy as np
import MGTomo.plots.database as db

def normalize_time(t : np.ndarray,idx = None):
  t = t.copy()
  if idx is not None:
    t = t[:,idx]
  m = np.amin(t)
  t = t - m
  t /= 1e+9
  return t

def coarse_corrections_gradient(h,x : np.ndarray):
  x = x.copy()
  x = x[x[:, 0] == h, 1:3]
  return x

def coarse_corrections_cycle(h,x : np.ndarray, y : np.ndarray):
  # x -> all iterations
  # y -> coarse correction

  x, y = x.copy(), y.copy()
  x = x[x[:, 0] == h, 1:3]
  y = y[y[:, 0] == h, 2]

  _, idx = np.unique(x[:,0], return_inverse=True)
  z = np.zeros((x.shape[0],3))
  z[:,0] = idx # indices
  z[:,1] = np.arange(x.shape[0]) - 1
  z = z[z[:,0] == 0,:]
  z = z[1:-1,:]
  z[:,2] = y

  return z[:,1:3]

def get_coarse_plot(exp_id: int,config: dict,range=None):
  cfg = db.get_config(config)
  d, f = db.get_db_entry(cfg['collection'], exp_id, cfg['projection'], cfg['path'], cfg['hdfkeys'])
  iters = f['iterations']
  coarse_iters = f['coarse']

  h = np.amax(iters[:,0])
  coarse_iters = coarse_iters[coarse_iters[:, 0] == h, :]
  coarse_iters = coarse_iters[:, [1, 2, 6]]
  idx = coarse_iters[:, 0].astype(np.int)
  iters = iters[iters[:, 0] == h, :]
  iters = iters[:, [2,5,6]]

  return idx+1,iters,coarse_iters[:,[1,2]]
