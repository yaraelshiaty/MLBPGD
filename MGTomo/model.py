from MGTomo.tomoprojection import TomoParallel,TomoParallelSquare
import MGTomo.grid as grid
#import MGTomo.mgtools as mgtools
import MGTomo.Tools.json as mjson

from skimage.util import view_as_windows
import astra
import numpy as np
import scipy.sparse.linalg as sparse_linalg
import math,inspect


def reduce_dim(n: int, k: int = 1):
  if math.frexp(n)[0] == 0.5:
    return int(n / 2 ** k)
  elif math.frexp(n + 1)[0] == 0.5:
    return int(((n + 1) / 2 ** k) - 1)
  else:
    assert 0 > 0, 'Dimension has to be 2^k or 2^k - 1'

def reduce_sinogram(b: np.ndarray, stencil: np.ndarray):
  assert stencil.shape[0] == 1 and stencil.shape[1] > 1

  if stencil.size % 2 == 1 :
    s = int(np.ceil(stencil.size / 2))
  else:
    s = stencil.size

  return np.einsum('ijkl,kl->ij',
                   view_as_windows(b, (1, stencil.size), step=(1,s)),
                   stencil
                   )

def spectral_norm(p: TomoParallelSquare):
  return sparse_linalg.eigsh(p, k=1, return_eigenvectors=False)[0]

class astra_model(object):
  bh = dict()

  def __init__(self,  dim: int, options=None):
    """

    opt = {'mode' : 'cuda',
           'num_angles' : 50,
           'level_decrease' : 1}

    :param x:
    :param options:
    """
    n = dim
    assert 0.5 == math.frexp(n)[0] or 0.5 == math.frexp(n+1)[0]

    if options is None:
      options = dict()

    opt = {'mode' : 'cuda',
           'num_angles' : 50,
           'level_decrease' : 1}
    opt.update(options)

    self.options = opt
    self.dim = dim
    self.mode = opt['mode']
    self.lvl_dec = opt['level_decrease']
    self.angles = np.linspace(0, np.pi, opt['num_angles'], False)

    s = grid.FullWeightStencil(self.dim,reduce_dim(self.dim,self.lvl_dec))
    self.x_stencil = s / np.amax(s)
    self.g_stencil = s / np.sum(s)

    q = np.outer(s,s)
    self.g_infnorm = np.amax(q)/np.sum(q)
    self.g_sigma = 1. # np.sum(q)/np.amax(q)
    self.projs = dict()

    h, i = self.dim, 1
    while h > 4:
      proj_geom = astra.create_proj_geom('parallel', 1., h, self.angles)
      vol_geom = astra.create_vol_geom(h, h)
      proj = TomoParallel(proj_geom, vol_geom, mode=self.mode)
      self.projs[h] = proj
      h = self.reduce_dim(self.dim, i)
      i += 1

  def rhs_dict(self,x):
    proj = self.proj_factory()
    b = proj.matvec(x)
    bh = dict()
    h, i = self.dim, 1
    while h > 4:
      bh[h] = self.reduce_rhs(b, self.dim, h)
      h = self.reduce_dim(self.dim, i)
      i += 1
    return bh

  def proj_factory(self,n = None,dist=1.):
      if n is None:
        n = self.dim
      return  self.projs[n]

  def proj_square_factory(self,n = None,dist=1.):
    if n is None:
      n = self.dim
    if n in self.projs.keys():
      return self.projs[n]
    else:
      proj_geom = astra.create_proj_geom('parallel', dist, n, self.angles)
      vol_geom = astra.create_vol_geom(n, n)
      proj = TomoParallelSquare(proj_geom, vol_geom, mode=self.mode)
      self.projs[n] = proj
      return proj

  def tojson(self):
    d = dict()
    d['name'] = str(self)
    d['source'] = inspect.getsource(astra_model)
    d['sourcefile'] = inspect.getsourcefile(astra_model)
    opt = mjson.make_json_dict(self.options).copy()
    opt.update({ 'dim' : self.dim,
                 'x_stencil' : self.x_stencil.shape ,
                 'g_stencil' : { 'shape' : self.x_stencil.shape,
                                 'g_infnorm' : self.g_infnorm}
                 })
    d['options'] = opt
    return d


  @staticmethod
  def reduce_rhs(b,n,m):
    if m == n:
      return b
    s = grid.FullWeightStencil(n, m)
    s = s.reshape(1, s.size)
    s = s/np.amax(s)
    return reduce_sinogram(b,s)

  def reduce_dim(self,n: int,k: int = None):
    if k is None:
      k = self.lvl_dec
    return reduce_dim(n,k)