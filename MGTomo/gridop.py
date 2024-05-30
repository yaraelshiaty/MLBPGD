import torch
import torch.nn.functional as F

def P(input_image):
    # Define the bilinear interpolation kernel
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32) / 4
    
    # Convert the kernel to 4D tensor
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Apply convolution with bilinear interpolation kernel
    output_image = F.conv_transpose2d(input_image.unsqueeze(0).unsqueeze(0), kernel, stride=2)

    return output_image.squeeze(0).squeeze(0)

def R(input_image):
    # Define the bilinear interpolation kernel for transposed convolution
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32) / 4
    
    # Convert the kernel to 4D tensor
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Apply transposed convolution with bilinear interpolation kernel
    output_image = F.conv2d(input_image.unsqueeze(0).unsqueeze(0), kernel, stride=2)

    return output_image.squeeze(0).squeeze(0)

def R_v2(input_image):
    # Perform downsampling using nearest neighbor interpolation
    output_image = F.interpolate(input_image.unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest')

    return output_image.squeeze(0).squeeze(0)