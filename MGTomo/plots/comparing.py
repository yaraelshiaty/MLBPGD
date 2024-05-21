import numpy as np

def compare_results(x,y,l = None,u = None, e = 1e-6):
  if l is not None:
    y = np.where(np.abs(y - l) < e ,l ,y )
  if u is not None:
    y = np.where(np.abs(u - y) < e ,u ,y )
  d = np.linalg.norm(x-y,'fro')
  return d ,(d**2)/x.size

def map2bounds(x,l = None,u = None, e = 1e-6):
  if l is not None:
    x = np.where(np.abs(x - l) < e ,l ,x )
  if u is not None:
    x = np.where(np.abs(u - x) < e ,u ,x )
  return x