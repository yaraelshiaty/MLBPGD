import numpy as np
import scipy.sparse as sci
import timeit
import cupy as cp

import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})
from MGTomo.cython.cython_huber import huber
from MGTomo.tomoprojection import TomoParallel

def image_diff(I: np.ndarray) -> (np.ndarray,np.ndarray):
  """
  Calculates horizontal and vertical differences of a
  quadratic array (from numpy)
  :type I: ndarray
  :returns (horizontal,vertical) differences
  """
  assert I.shape[0] == I.shape[1]
  return np.diff(I,axis=0),np.diff(I,axis=1)

def image_diff_t(p: np.ndarray,q: np.ndarray) -> (np.ndarray,np.ndarray):
  """
  Takes the horizontal and vertical differences of a quadratic array
  to calculate the transposed values
  :param p:
  :param q:
  :return:
  """
  assert p.shape[0] == q.shape[1]
  assert p.shape[1] == q.shape[0]
  n = np.amax(p.shape)
  a = (-1)*np.concatenate((p[0].reshape((1, n)),    np.diff(p, axis=0), (-1) * p[-1].reshape(1, n)),    axis=0)
  b = (-1)*np.concatenate((q[:, 0].reshape((n, 1)), np.diff(q, axis=1), (-1) * q[:, -1].reshape(n, 1)), axis=1)
  return a,b

kernel_kl = cp.ElementwiseKernel(
    'float32 a,float32 b', 'float32 f, float32 x, float32 y',
    ''' 
        if( b > 0 ){
          if( a > 0 ){
            f = a*log(a/b) + b - a;
            x = log(a/b);
          } else {
            f = 0. ;
            x = log(a/b);
          }
        } else {
          f = 0.5*a*a;
          y = a;
        }
    ''', 'kl_distance')

def kl_distance(x: np.ndarray,proj: TomoParallel,b: np.ndarray,cuda=True):
  ax = proj.matvec(x)
  if cuda :
    ax = cp.asarray(ax, dtype=np.float32)
    b = cp.asarray(b,dtype=np.float32)

    f = cp.zeros_like(ax)
    bg = cp.zeros_like(ax)
    xg = cp.zeros_like(ax)

    kernel_kl(ax, b, f, bg, xg)
    f = cp.asnumpy(cp.sum(f))

    g0 = proj.rmatvec(cp.asnumpy(bg))
    g1 = proj.rmatvec(cp.asnumpy(xg))
  else:
    ab = np.divide(ax,b,out=np.ones_like(ax),where= b > 0)
    erg = ax * np.log(ab) + b - ax
    f = np.sum( erg[b > 0.] ) + 0.5*np.sum(ax[b == 0.]**2)
    ax[b > 0.] = 0.

    g0 = proj.rmatvec(np.log(ab))
    g1 = proj.rmatvec(ax)

  return f,g0

def least_square(x: np.ndarray,proj: TomoParallel,b: np.ndarray):
  y = proj.matvec(x) - b

  f = np.linalg.norm(y,'fro')
  f = 0.5*(f**2)
  g = proj.rmatvec(y)
  return f,g

kernel_diff_transpose = cp.ElementwiseKernel(
  'raw float32 x, int32 xdim,int32 zdim', 'raw float32 z',
  ''' 
      int ydim = xdim - 1;
      if( i % zdim == 0 ){
        int idx = ((i - (i % zdim))/zdim)*xdim + (i % zdim);
        z[i] += (-1)*x[idx];
      } else if( i % zdim == zdim - 1 ){
        int idx = ((i - (i % zdim))/zdim)*xdim + xdim - 1;
        z[i] += x[idx];
      } else {
        int idx = ((i - (i % zdim))/zdim)*xdim + (i % zdim) - 1;
        z[i] += x[idx] - x[idx+1];
      }
  ''', 'diff_transpose')

kernel_huber = cp.ElementwiseKernel(
  'raw float32 x, int32 xdim, float32 r', 'raw float32 f, raw float32 g',
  ''' 
      int ydim = xdim - 1;
      int idx = ((i - (i % ydim))/ydim)*xdim + (i % ydim);
      float p = x[idx+1] - x[idx];

      if( fabs(p) <= r ){
          f[i] += (p*p)/(2.*r);
          g[i] = p/r;
      } else {
          f[i] += fabs(p) - 0.5*r;
          g[i] = p/fabs(p);
      }
  ''', 'huberdiff')

def tv_huber(x,rho,cython=False):
  """

  .. math:: \| \\nabla x \|_\\rho

  :param x:   input image
  :param rho: smoothing parameter
  :return:
  """
  assert x.shape[0] == x.shape[1]
  assert x.shape[0] > 1
  assert rho > 0

  if cython :
    p, q = image_diff(x)
    f,gp,gq = huber(p,q,rho)
    a, b = image_diff_t(gp, gq)
    g = a + b
  else:
    x = cp.asarray(x, dtype=np.float32)

    gp = cp.zeros((x.shape[0], x.shape[1] - 1), dtype=np.float32)
    f = cp.zeros_like(gp)
    f, gp = kernel_huber(x, x.shape[1], rho, f, gp, size=gp.size)

    gq = cp.zeros((x.shape[1], x.shape[0] - 1), dtype=np.float32)
    f, gq = kernel_huber(cp.transpose(x), x.shape[0], rho, cp.transpose(f), gq, size=gq.size)
    f = cp.asnumpy(cp.sum(f))

    g = cp.zeros_like(x)
    g = kernel_diff_transpose(gp, gp.shape[1], g.shape[1], g, size=g.size)
    g = kernel_diff_transpose(gq, gq.shape[1], g.shape[1], cp.transpose(g), size=g.size)
    g = cp.asnumpy(cp.transpose(g))

  return f,g

def least_square_smooth_tv(x,proj,b,rho,lbd,v: np.ndarray = 0.):
  f0,g0 = least_square(x,proj,b)
  f1,g1 = tv_huber(x,rho) if lbd > 0 else (0.,0.)

  if np.all(v == 0) :
    return f0 + lbd*f1, g0 + lbd*g1
  else:
    return f0 + lbd*f1 + np.sum(np.multiply(x,v)), g0 + lbd*g1 + v


def kl_distance_smooth_tv(x, proj, b, rho, lbd, v: np.ndarray = 0.):
  f0, g0 = kl_distance(x, proj, b)
  f1, g1 = tv_huber(x,rho) if lbd > 0 else (0.,0.)

  if np.all(v == 0):
    return f0 + lbd * f1, g0 + lbd * g1
  else:
    return f0 + lbd * f1 + np.sum(np.multiply(x, v)), g0 + lbd * g1 + v

if __name__ == '__main__':
  n,rho = 4, 0.01
  a = cp.asarray(np.random.rand(n, 1),dtype=np.float32)
  b = cp.asarray(np.random.rand(n, 1),dtype=np.float32)


  #x = cp.zeros_like(a)
  #y = cp.zeros_like(a)

  f,x,y = kernel_kl(a,b)
  print(f)
  print(x)
  print(y)


