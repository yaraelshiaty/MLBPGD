import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport fabs
from cython.parallel import prange

ctypedef fused REAL:
  float
  double


cdef inline REAL sign(REAL x) nogil:
  if x > 0.:
    return 1.
  elif x < 0.:
    return -1.
  else:
    return 0.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def huber(cnp.ndarray[REAL, ndim=2] p, cnp.ndarray[REAL, ndim=2] q, REAL rho):
  """
  Calculates value for smoothed l1-norm
  """
  cdef int m = p.shape[0], n = p.shape[1]
  cdef int i, j
  cdef REAL f = 0.
  cdef cnp.ndarray[REAL, ndim=2] gp = np.empty((m,n))
  cdef cnp.ndarray[REAL, ndim=2] gq = np.empty((n,m))

  for i in prange(m, nogil=True, schedule='static'):
    for j in range(n):
      if fabs(p[i,j]) <= rho:
        f += p[i,j] * p[i,j] / (2. * rho)
        gp[i,j] = p[i,j] / rho
      else:
        f += fabs(p[i,j]) - 0.5 * rho
        gp[i,j] = sign(p[i,j])
        
  for j in prange(n, nogil=True, schedule='static'):
    for i in range(m):
      if fabs(q[j,i]) <= rho:
        f += q[j,i] * q[j,i] / (2. * rho)
        gq[j,i] = q[j,i] / rho
      else:
        f += fabs(q[j,i]) - 0.5 * rho
        gq[j,i] = sign(q[j,i])
          
  return f,gp,gq
