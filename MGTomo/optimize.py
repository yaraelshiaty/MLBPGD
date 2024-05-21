import numpy as np

def armijo_linesearch(fct,x: np.ndarray,d: np.ndarray,a=1.,r=0.8,c=1e-4):
  """

  :param fct:  function returns f,g
  :param x:    current point
  :param d:    tangent vector
  :param a:    maximal step
  :param r:    decreasing factor
  :param c:    slope
  :return:
  """

  f,g = fct(x)
  dgk = np.sum(g*d)
  f_new,_ = fct(x + a * d)
  x_new = x.copy()

  assert dgk <= 0, 'd needs to be a descent direction (dkg = %.5e)' % dgk

  if dgk == 0.:
    return x,0.

  while f_new > f + a * c * dgk and a > 1e-7:
    x_new = x + a * d
    f_new, _ = fct(x_new)
    a *= r

  if f_new < f :
    return x_new, np.minimum(a/r,a)
  else:
    return x,0.


def armijo_manifold_linesearch(fct,retr,x: np.ndarray,rg: np.ndarray, d: np.ndarray,a=1.,r=0.8,c=1e-4):
  """

  :param fct:  function returns f,g
  :param retr: retraction
  :param x:    current point
  :param rg:   Riemannian gradient
  :param d:    tangent vector
  :param a:    maximal step
  :param r:    decreasing factor
  :param c:    slope
  :return:
  """
  f, _ = fct(x)
  dgk = np.sum(rg*d)

  x_new = retr(a * d)
  f_new, _ = fct(x_new)

  assert dgk <= 0, 'd needs to be a descent direction (dkg = %.5e)' % dgk

  if dgk == 0.:
    return x,0.

  while f_new > f + a * c * dgk and a > 1e-7:
    x_new = retr(a*d)
    f_new, _ = fct(x_new)
    a *= r

  if f_new < f :
    return x_new, np.minimum(a/r,a)
  else:
    return x,0.

if __name__ == '__main__' :

  A = np.random.rand(3,5)
  At = np.transpose(A)
  x = np.random.rand(5,1)
  b = np.dot(A,x)

  def fobj(x):
    y = np.dot(A,x) - b
    f = 0.5*np.linalg.norm(y,2)**2
    g = np.dot(At,y)
    return  f,g

  x = np.zeros((5,1))
  for i in range(100):
    f,g = fobj(x)
    g = g/np.linalg.norm(g,'fro')
    x,a = armijo_linesearch(fobj,x,np.multiply(g,-1.))
    print('%d -> %.5e  %.5e ' % (i,f,a,))
    if a < 1e-15 :
      break
