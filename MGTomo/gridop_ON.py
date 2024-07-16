import torch
import torch.nn.functional as F
from torch.nn.functional import unfold, fold

def orthonormal_kernel(kernel_size=3):
    # Define the bilinear interpolation kernel
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32)
    
    # Normalize the kernel to have unit norm
    kernel = kernel / torch.norm(kernel)

    # Perform SVD on the kernel
    U, S, Vt = torch.svd(kernel)
    
    # Construct an orthonormal kernel
    orthonormal_kernel = U @ torch.diag(S) @ Vt.T

    return orthonormal_kernel

def orthonormal_downsampling(input_image, kernel_size=3):
    # Get the orthonormal kernel and convert to a 4D tensor
    kernel = orthonormal_kernel(kernel_size).unsqueeze(0).unsqueeze(0)

    # Downsample the input image using the orthonormal kernel
    output_image = F.conv2d(input_image.unsqueeze(0).unsqueeze(0), kernel, stride=2)

    return output_image.squeeze(0).squeeze(0)


def upsampling_transposed(input_image, kernel_size=3):
    # Get the orthonormal kernel
    kernel = orthonormal_kernel(kernel_size)
    
    # Transpose the kernel for upsampling
    #transposed_kernel = kernel.T
    
    # Convert the transposed kernel to a 4D tensor
    #transposed_kernel = transposed_kernel.unsqueeze(0).unsqueeze(0)
    
    # Apply transposed convolution for upsampling
    output_image = F.conv_transpose2d(input_image.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), stride=2)

    return output_image.squeeze(0).squeeze(0)

