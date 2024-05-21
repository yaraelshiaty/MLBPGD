import numpy as np
import cupy as cp

def myexp(x):
  y = np.exp(np.clip(x,-700.,700.))
  y[x < -700] = 0.
  y[x > 700] = np.inf
  return y

kernel_expmap = cp.ElementwiseKernel(
    'float32 x,float32 v,float32 l,float32 u,float32 m', 'float32 g',
    ''' 
        if( l < u ){
          float e = ((u-l)*(u-l)*v)/((u-x)*(x-l));
          if( e < -50 ){
            g = l + m;
          } else if( e > 50 ){
            g = u - m;
          } else {
            e = exp(e);
            g = ((u-l)*(x-l)*e)/(u-x + (x-l)*e) + l;
            if( g < l + m ) { g = l + m; }
            if( g > u - m){ g = u - m ; }
          }
        } else {
          g = x;
        }        
    ''', 'kl_expmap')

kernel_expmap_grad = cp.ElementwiseKernel(
    'float32 x,float32 v,float32 l,float32 u, float32 m', 'float32 g',
    ''' 
        if( l < u ){
          if( v < -50 ){
            g = l + m;
          } else if( v > 50 ){
            g = u - m;
          } else {
            float e = exp(v);
            g = ((u-l)*(x-l)*e)/(u-x + (x-l)*e) + l;
            if( g < l + m ) { g = l + m; }
            if( g > u - m){ g = u - m; }
          }
        } else {
          g = x;
        }        
    ''', 'kl_expmap_grad')

class manifold(object):
  def __init__(self,vmin=1e-12):
    self.vmin = vmin

class box_manifold(manifold):
  def __init__(self,vmin=1e-6):
    super().__init__(vmin)

  def retraction_cuda(self,v, x, l, u, grad=False):
    x = cp.asarray(x, dtype=np.float32)
    v = cp.asarray(v, dtype=np.float32)
    lc = cp.asarray(l, dtype=np.float32)
    uc = cp.asarray(u, dtype=np.float32)

    if grad:
      r = kernel_expmap_grad(x, v, lc, uc, self.vmin)
    else:
      r = kernel_expmap(x,v,lc,uc,self.vmin)

    return np.clip(cp.asnumpy(r),l+self.vmin,u-self.vmin)

  def retraction(self,v, x, l, u, grad=False):
    ul, xl, ux, idx = u - l, x - l, u - x, l < u
    assert np.amin(ux[idx]) > 0., 'x is to close at u  (%.5e)' % np.amin(ux[idx])
    assert np.abs(np.amax(xl[idx])) > 0., 'x is to close at l  (%.5e)' % np.abs(np.amax(xl[idx]))

    if np.isscalar(l):
      ul_out = ul * np.ones_like(x)
    else:
      ul_out = ul.copy()

    if grad :
      e = np.where(l < u,xl * myexp(v),np.inf)
    else:
      e = xl * myexp(np.divide(np.power(ul, 2) * v, xl * ux,out=np.full_like(x,np.inf),where=idx))

    erg = l + np.divide(ul * e,ux + e,out=ul_out,where=e < np.inf)

    return np.clip(erg, l + self.vmin, u - self.vmin)

  @staticmethod
  def riemannian_gradient(g, x, l, u):
    e = np.divide((x-l) * (u-x), np.power(u - l, 2),out=np.zeros_like(g),where=l < u)
    return e * g


# class pos_manifold(object):
#   def __init__():
#     pass
#
#   def retraction(v, x):
#     return np.clip(np.exp(v / x), eps, 1e+10)
#
#   def retraction_gradient(v, x):
#     return np.clip(np.exp(v), eps, 1e+10)
#
#   def riemannian_gradient(g, x):
#     return x * g

if __name__ == '__main__':
  l = np.zeros((4,4))
  u = np.ones(l.shape)

  x = np.random.random(l.shape)

  s = np.random.randint(0,2,x.shape)
  s[s == 0] = -1
  v = 1e+3*np.random.random(l.shape)*s

  box = box_manifold()

  y = box.retraction(v, x, l, u, grad=True)
  y = box.retraction(v, x, l, u, grad=False)

