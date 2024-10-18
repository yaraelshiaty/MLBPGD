import numpy as np
#import cupy as cp
import math
import scipy.io
from scipy import ndimage
from skimage.util import view_as_windows


def get_bounds(I, stencil: np.ndarray, l, u):
  assert I.shape[0] == I.shape[1], 'dimensions of image has to be equal'
  assert isinstance(stencil,type(I))
  n = I.shape[0]
  if stencil.ndim == 2:
    assert stencil.shape[0] == stencil.shape[1]
    r = stencil.shape[0]
    assert 0.5 == math.frexp(n)[0]
    assert 0.5 == math.frexp(r)[0]
    s = r
  else:
    r = stencil.size
    assert 0.5 == math.frexp(n + 1)[0]
    assert 0.5 == math.frexp(r + 1)[0]
    s = int(np.ceil(r / 2))
    stencil = np.outer(stencil, stencil)

  lx = l - I
  ux = u - I
  lx_patches = view_as_windows(lx, stencil.shape, step=s)
  ux_patches = view_as_windows(ux, stencil.shape, step=s)

  patch_sign = np.sign(stencil)
  assert np.any(patch_sign > 0)

  lh = np.amax(lx_patches[:, :, np.sign(stencil) > 0], axis=2)
  uh = np.amin(ux_patches[:, :, np.sign(stencil) > 0], axis=2)

  if np.any(patch_sign < 0):
    lh_neg = np.amin(ux_patches[:, :, np.sign(stencil) < 0], axis=2)
    uh_neg = np.amax(lx_patches[:, :, np.sign(stencil) < 0], axis=2)

    lh = np.maximum(lh, (-1) * lh_neg)
    uh = np.minimum(uh, (-1) * uh_neg)

  return lh, uh

def GroupingStencil(n, m):
  assert 0.5 == math.frexp(n)[0]
  assert 0.5 == math.frexp(m)[0]
  assert n > m

  k = math.frexp(n)[1] - math.frexp(m)[1]
  return np.ones((2**k,2**k))

def FullWeightStencil(n, m):
  assert 0.5 == math.frexp(n + 1)[0]
  assert 0.5 == math.frexp(m + 1)[0]
  assert n > m

  k = math.frexp(n + 1)[1] - math.frexp(m + 1)[1]
  r = int(2**(k+1) - 1)
  c = int((r+1)/2)

  x = np.arange(1,c)
  x = np.append(x,[c])
  x = np.append(x,np.arange(c-1,0,-1))

  return x

def fine2coarse(I: np.ndarray, stencil: np.ndarray) -> np.ndarray:
  """
  Reducing a 2D array according to the stencil

  :param I:
  :param stencil:
  :return:
  """
  assert I.shape[0] == I.shape[1], 'dimension of image has to be equal'
  n = I.shape[0]

  if stencil.ndim == 2:
    assert stencil.shape[0] == stencil.shape[1]
    r = stencil.shape[0]
    assert 0.5 == math.frexp(n)[0]
    assert 0.5 == math.frexp(r)[0]
    s = r
  else:
    r = stencil.size
    assert 0.5 == math.frexp(n+1)[0]
    assert 0.5 == math.frexp(r+1)[0]
    s = int(np.ceil(r / 2))
    stencil = np.outer(stencil,stencil)

  if not I.flags.c_contiguous:
    I = np.ascontiguousarray(I)
  return np.einsum('ijkl,kl->ij', view_as_windows(I, stencil.shape, step=s), stencil)


def coarse2fine(I: np.ndarray, stencil: np.ndarray) -> np.ndarray:
  assert I.shape[0] == I.shape[1], 'dimension of image has to be equal'
  n = I.shape[0]

  if stencil.ndim == 2:
    assert stencil.shape[0] == stencil.shape[1]
    r = stencil.shape[0]
    assert 0.5 == math.frexp(n)[0]
    assert 0.5 == math.frexp(r)[0]
  else:
    r = stencil.size
    assert 0.5 == math.frexp(n+1)[0]
    assert 0.5 == math.frexp(r+1)[0]

  if r % 2 == 1:
    k = math.frexp(r + 1)[1] - 2
    m = int((2**k) * (n + 1) - 1)
    s = int((r+1)/2)
    b = int((r-1)/2)
    Z = np.zeros((m, m))
    Z[b::s, b::s] = I

    Z = ndimage.convolve1d(Z, stencil, axis=0, mode='constant')
    return ndimage.convolve1d(Z, stencil, axis=1, mode='constant')
  else:
    return np.kron(I, stencil)

if __name__ == '__main__':
  xh = np.abs(np.random.random((3, 3))).astype(np.float32)
  g = np.sign(np.random.random((7, 7)).astype(np.float32)-0.5)
  s = np.array([1, 2, 1])
  s = np.outer(s, s) / 4