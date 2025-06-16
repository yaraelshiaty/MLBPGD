import torch
import torch.nn.functional as F

def P(input_image):
    # Define the bilinear interpolation kernel
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32)
    
    # Convert the kernel to 4D tensor
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Apply convolution with bilinear interpolation kernel
    output_image = F.conv_transpose2d(input_image.unsqueeze(0).unsqueeze(0), kernel, stride=2)

    return output_image.squeeze(0).squeeze(0)

def R(input_image):
    # Define the bilinear interpolation kernel for transposed convolution
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32)
    
    # Convert the kernel to 4D tensor
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Apply transposed convolution with bilinear interpolation kernel
    output_image = F.conv2d(input_image.unsqueeze(0).unsqueeze(0), kernel, stride=2)

    return output_image.squeeze(0).squeeze(0)


def compute_nonzero_elements_of_P(input_size):
    # Define the bilinear interpolation kernel (normalized)
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32) / 16.0
    
    # Get dimensions
    n = input_size
    out_n = 2 * n + 1
    
    # Dictionary to store nonzero elements indexed by the columns of P
    P_dict = {}
    
    # Iterate over the input image
    for i in range(n):
        for j in range(n):
            # Calculate the top-left corner of the region in the output affected by (i, j)
            out_i = 2 * i
            out_j = 2 * j
            
            # Get the 2D coordinates of the column index in the input image
            input_coords = (i, j)
            
            # Initialize the list for the current column if not already initialized
            if input_coords not in P_dict:
                P_dict[input_coords] = []
            
            # Iterate over the kernel
            for ki in range(3):
                for kj in range(3):
                    # Compute the position in the output image
                    out_pos_i = out_i + ki
                    out_pos_j = out_j + kj
                    
                    # Check if the position is within bounds
                    if 0 <= out_pos_i < out_n and 0 <= out_pos_j < out_n:
                        # Get the 2D coordinates of the row index in the output image
                        output_coords = (out_pos_i, out_pos_j)
                        
                        # Append the (row coordinates, weight) pair to the column's list
                        P_dict[input_coords].append(output_coords)
    
    return P_dict

def R_v2(input_image):
    # Perform downsampling using nearest neighbor interpolation
    output_image = F.interpolate(input_image.unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest')

    return output_image.squeeze(0).squeeze(0)

def R_v3(y):
    
    x = y[1:-1:2, 1:-1:2]
    
    return x

def RBox(input_image):
    # Define the bilinear interpolation kernel for transposed convolution
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32) / 4
    
    # Normalize the kernel so that the sum of its weights equals 1
    kernel = kernel / kernel.sum()
    
    # Convert the kernel to 4D tensor
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Apply transposed convolution with bilinear interpolation kernel
    output_image = F.conv2d(input_image.unsqueeze(0).unsqueeze(0), kernel, stride=2)

    return output_image.squeeze(0).squeeze(0)

def PBox(input_image):
    # Define the bilinear interpolation kernel
    kernel = torch.tensor([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=torch.float32) / 4
    
    # Normalize the kernel so that the sum of its weights equals 1
    kernel = kernel / kernel.sum()
    
    # Convert the kernel to 4D tensor
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Apply convolution with bilinear interpolation kernel
    output_image = F.conv_transpose2d(input_image.unsqueeze(0).unsqueeze(0), kernel, stride=2)

    return output_image.squeeze(0).squeeze(0)
