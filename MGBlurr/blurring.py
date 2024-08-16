import torch
import torchvision.transforms as T
import torch.nn.functional as F

import math

class GaussianBlurOperator:
    def __init__(self, image_size, kernel_size, sigma):
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, x):
        # Ensure x is a torch tensor and has the correct shape
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        if x.ndim == 2:  # If x is a 2D image, add batch and channel dimensions
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:  # If x is a 3D image, add only the channel dimension
            x = x.unsqueeze(0)

        # Apply Gaussian blur
        blurred_x = self.blur(x)

        if x.requires_grad:
           blurred_x.requires_grad_(True)

        # Remove the batch and channel dimensions if necessary
        if blurred_x.shape[0] == 1 and blurred_x.shape[1] == 1:
            return blurred_x.squeeze(0).squeeze(0)
        elif blurred_x.shape[0] == 1:
            return blurred_x.squeeze(0)
        else:
            return blurred_x
    
    def row_sum(self):
        # Initialize a tensor for the Gaussian kernel
        kernel_size = self.kernel_size
        sigma = self.sigma

        # Generate a Gaussian kernel
        kernel = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum()

        # Compute the row sum norm as the sum of the kernel's absolute values
        row_sum_norm = kernel.sum().item()

        return row_sum_norm
    

def reduce_dim(n: int, k: int = 1):
  if math.frexp(n)[0] == 0.5:
    return int(n / 2 ** k)
  elif math.frexp(n + 1)[0] == 0.5:
    return int(((n + 1) / 2 ** k) - 1)
  else:
    assert 0 > 0, 'Dimension has to be 2^k or 2^k - 1'

if __name__ == "__main__":
   # Example usage:
    image_size = 1024  # Size of the image (e.g., 32x32)
    kernel_size = 33  # Size of the Gaussian kernel
    sigma = 5.0  # Standard deviation of the Gaussian kernel

    # Create the GaussianBlurOperator
    A = GaussianBlurOperator(image_size, kernel_size, sigma)

    # Create a sample input image (e.g., a 32x32 random image)
    x = torch.rand(image_size, image_size)

    # Apply the blur operator
    blurred_x = A(x)

    print(blurred_x.shape)  # Should be (32, 32), same as the input image size

    # Compute the row sum norm of the Gaussian blur operator
    row_sum_norm = A.row_sum()
    print("Row sum norm of the Gaussian blur operator:", row_sum_norm)