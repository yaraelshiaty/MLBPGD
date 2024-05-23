import astra
import numpy as np

import torch 
import torch.nn as nn

class AstraLinearFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, tomo_matrix, input_vector):
        # Perform the sparse matrix-vector multiplication
        output = tomo_matrix.matvec(input_vector.cpu().numpy())
        # Save the custom sparse matrix and input vector for backward pass
        ctx.tomo_matrix = tomo_matrix
        return torch.from_numpy(output)
    
    @staticmethod
    def backward(ctx, grad_output):
        tomo_matrix = ctx.tomo_matrix
        # Compute the gradient with respect to the input vector
        grad_input_vector = tomo_matrix.rmatvec(grad_output.cpu().numpy())
        # Return gradients for each input
        # Note: Since custom_sparse_matrix is not a tensor that requires gradients,
        # we return None for its gradient
        return None, torch.from_numpy(grad_input_vector)


class TomoTorch(nn.Module):
    def __init__(self, tomop):
        super(TomoTorch, self).__init__()
        self.tomop = tomop
        self.shape = self.tomop.shape
        self.dtype = self.tomop.dtype
        
    def forward(self, v):
        return AstraLinearFunction.apply(self.tomop, v)
    

class TomoParallel(object):
  def __init__(self, proj_geom, vol_geom, mode='line'):
    self.proj_id = astra.create_projector(mode, proj_geom, vol_geom)
    self.shape = (proj_geom['DetectorCount'] * len(proj_geom['ProjectionAngles']),
                  vol_geom['GridColCount'] * vol_geom['GridRowCount'])
    self.dtype = float

  def matvec(self, v):
    sid, s = astra.create_sino(v, self.proj_id) # np.reshape(v, (vol_geom['GridRowCount'], vol_geom['GridColCount']))
    astra.data2d.delete(sid)
    return s

  def rmatvec(self, v):
    bid, b = astra.create_backprojection(v,self.proj_id)
      # np.reshape(v, (len(proj_geom['ProjectionAngles']), proj_geom['DetectorCount'],)), self.proj_id)
    astra.data2d.delete(bid)
    return b

  def __del__(self):
    astra.projector.delete(self.proj_id)

class TomoParallelSquare(object):
  def __init__(self, proj_geom, vol_geom, mode='line'):
    self.proj_id = astra.create_projector(mode, proj_geom, vol_geom)
    self.shape = (vol_geom['GridColCount'] * vol_geom['GridRowCount'],
                  vol_geom['GridColCount'] * vol_geom['GridRowCount'])
    self.dtype = np.dtype('float')
    self.vol_geom = vol_geom

  def mp(self,v):
    sid = astra.create_sino(np.reshape(v, (self.vol_geom['GridRowCount'], self.vol_geom['GridColCount'])),
                          self.proj_id,returnData=False)
    bid, b = astra.create_backprojection(sid, self.proj_id)
    astra.data2d.delete(bid)
    astra.data2d.delete(sid)

    return b.ravel()

  def matvec(self, v):
    return  self.mp(v).ravel()

  def rmatvec(self, v):
    return  self.mp(v).ravel()

  def __del__(self):
    astra.projector.delete(self.proj_id)

if __name__ == '__main__':
  vol_geom = astra.create_vol_geom(256, 256)
  proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0, np.pi, 180, False))

  # Create a 256x256 phantom image
  import scipy.io

  P = scipy.io.loadmat('phantom.mat')['phantom256']

  # Create a sinogram using the GPU.
  proj_id = astra.create_projector('line', proj_geom, vol_geom)
  sinogram_id, sinogram = astra.create_sino(P, proj_id)

  # Reshape the sinogram into a vector
  b = sinogram.ravel()

  wrapper = TomoParallelSquare(proj_geom, vol_geom)
  import scipy.sparse.linalg as linalg
  print(linalg.eigs(wrapper, k=1,return_eigenvectors=False))

  astra.data2d.delete(sinogram_id)
  astra.projector.delete(proj_id)
  astra.projector.delete(wrapper.proj_id)
