import torch
import torch.nn.functional as F

class MultigridOperator2D:
    """
    Class for constructing 2D prolongation (P) and restriction (R) operators
    from a given kernel, and for computing nonzero structure and exact norm.
    """
    def __init__(self, kernel_2d: torch.Tensor):
        """
        Args:
            kernel_2d (torch.Tensor): 2D tensor (e.g., shape (3,3)) for interpolation.
        """
        self.kernel = kernel_2d.float()
        self.kernel_sum = self.kernel.sum()
        if self.kernel_sum != 0:
            self.kernel = self.kernel / self.kernel_sum  # Normalize

    def P(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Prolongation operator (upsampling) using transposed convolution.
        """
        k = self.kernel.unsqueeze(0).unsqueeze(0)
        return F.conv_transpose2d(input_image.unsqueeze(0).unsqueeze(0), k, stride=2).squeeze(0).squeeze(0)

    def R(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Restriction operator (downsampling) using convolution.
        """
        k = self.kernel.unsqueeze(0).unsqueeze(0)
        return F.conv2d(input_image.unsqueeze(0).unsqueeze(0), k, stride=2).squeeze(0).squeeze(0)

    def compute_nonzero_elements_of_P(self, input_size: int):
        """
        Returns a dictionary mapping each input pixel (i, j) to a list of output pixels (affected by the kernel).
        """
        kernel_size = self.kernel.shape[0]
        n = input_size
        out_n = 2 * n + 1
        P_dict = {}
        for i in range(n):
            for j in range(n):
                out_i = 2 * i
                out_j = 2 * j
                input_coords = (i, j)
                if input_coords not in P_dict:
                    P_dict[input_coords] = []
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        out_pos_i = out_i + ki
                        out_pos_j = out_j + kj
                        if 0 <= out_pos_i < out_n and 0 <= out_pos_j < out_n:
                            output_coords = (out_pos_i, out_pos_j)
                            P_dict[input_coords].append(output_coords)
        return P_dict

    def norm_infty_P(self):
        # kernel: 2D torch tensor
        abs_kernel = torch.abs(self.kernel)
        col_sums = abs_kernel.sum(dim=1)
        return col_sums.max().item()
    

# def P_rows_only_flat(input_flat, N):
#     """
#     1D prolongation (upsampling) using transposed convolution on a flattened vector.
#     input_flat: shape (N,)
#     Returns: shape (2*N + 1,)
#     """
#     input_1d = input_flat.view(1, 1, N)  # (batch, channel, width)
#     kernel_1d = torch.tensor([1, 2, 1], dtype=torch.float32)
#     kernel_1d = kernel_1d / kernel_1d.sum()
#     kernel_1d = kernel_1d.view(1, 1, 3)
#     output = F.conv_transpose1d(input_1d, kernel_1d, stride=2)
#     return output.view(-1)

# def R_rows_only_flat(input_flat, N):
#     """
#     1D restriction (downsampling) using convolution on a flattened vector.
#     input_flat: shape (N,)
#     Returns: shape (floor((N-3)/2)+1,)
#     """
#     input_1d = input_flat.view(1, 1, N)  # (batch, channel, width)
#     kernel_1d = torch.tensor([1, 2, 1], dtype=torch.float32)
#     kernel_1d = kernel_1d / kernel_1d.sum()
#     kernel_1d = kernel_1d.view(1, 1, 3)
#     output = F.conv1d(input_1d, kernel_1d, stride=2)
#     return output.view(-1)

class MultigridOperator1D:
    """
    1D multigrid operator using the fixed [1, 2, 1] interpolation kernel.
    Provides prolongation (P), restriction (R), and nonzero structure.
    """
    def __init__(self):
        # Fixed 3-width kernel [1, 2, 1]/4
        self.kernel = torch.tensor([1, 2, 1], dtype=torch.float32) / 4.0

    def P(self, input_flat: torch.Tensor) -> torch.Tensor:
        """
        1D prolongation (upsampling) using transposed convolution on a flattened vector.
        input_flat: shape (N,)
        Returns: shape (2*N + 1,)
        """
        N = input_flat.numel()
        input_1d = input_flat.view(1, 1, N)  # (batch, channel, width)
        kernel_1d = self.kernel.view(1, 1, 3)
        output = F.conv_transpose1d(input_1d, kernel_1d, stride=2)
        return output.view(-1)

    def R(self, input_flat: torch.Tensor) -> torch.Tensor:
        """
        1D restriction (downsampling) using convolution on a flattened vector.
        input_flat: shape (N,)
        Returns: shape (floor((N-3)/2)+1,)
        """
        N = input_flat.numel()
        input_1d = input_flat.view(1, 1, N)  # (batch, channel, width)
        kernel_1d = self.kernel.view(1, 1, 3)
        output = F.conv1d(input_1d, kernel_1d, stride=2)
        return output.view(-1)

    def norm_infty_P(self):
        return torch.abs(self.kernel).sum().item()

    def compute_nonzero_elements_of_P_rows_only_flat(self, input_size):
        """
        Returns a dictionary mapping input flat indices to lists of output flat indices
        for 1D row interpolation, for flattened arrays.
        """
        n = input_size  # input vector length
        out_n = 2 * n + 1

        P_dict = {}

        for i in range(n):
            input_flat_idx = i
            P_dict[input_flat_idx] = []
            for ki in range(3):  # 3-tap kernel
                out_i = 2 * i + ki
                if 0 <= out_i < out_n:
                    output_flat_idx = out_i
                    P_dict[input_flat_idx].append(output_flat_idx)

        return P_dict

def R_rows_only_flat_2d(input_flat, H, W):
    """
    Restrict a flattened (H, W) image along rows only, output shape (H_new, W).
    """
    input_image = input_flat.view(H, W)
    kernel_1d = torch.tensor([1, 2, 1], dtype=torch.float32)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None]
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)
    # Restrict along rows (height)
    output_image = F.conv2d(input_image.unsqueeze(0).unsqueeze(0), kernel_4d, stride=(2, 1))
    return output_image.squeeze(0).squeeze(0).flatten()

def P_rows_only_flat_2d(input_flat, H, W):
    """
    Prolongate a flattened (H, W) image along rows only, output shape (H_new, W).
    """
    input_image = input_flat.view(H, W)
    kernel_1d = torch.tensor([1, 2, 1], dtype=torch.float32)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None]
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)
    # Prolongate along rows (height)
    output_image = F.conv_transpose2d(input_image.unsqueeze(0).unsqueeze(0), kernel_4d, stride=(2, 1))
    return output_image.squeeze(0).squeeze(0).flatten()

class MultigridOperatorRowsOnly2D:
    """
    Multigrid operator that restricts/prolongs only along rows for 2D flattened input,
    with a fixed number of columns W for all levels.
    """
    def __init__(self, width):
        self.W = width  # Number of columns (detectors/projections)

    def R(self, x):
        H = x.numel() // self.W
        input_image = x.view(H, self.W)
        kernel_1d = torch.tensor([1, 2, 1], dtype=torch.float32) / 4.0
        kernel_2d = kernel_1d[:, None]
        kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)
        output_image = F.conv2d(input_image.unsqueeze(0).unsqueeze(0), kernel_4d, stride=(2, 1))
        return output_image.squeeze(0).squeeze(0).flatten()

    def P(self, x):
        H = x.numel() // self.W
        input_image = x.view(H, self.W)
        kernel_1d = torch.tensor([1, 2, 1], dtype=torch.float32) / 4.0
        kernel_2d = kernel_1d[:, None]
        kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)
        output_image = F.conv_transpose2d(input_image.unsqueeze(0).unsqueeze(0), kernel_4d, stride=(2, 1))
        return output_image.squeeze(0).squeeze(0).flatten()

    def norm_infty_P(self):
        return 1.0

    def compute_nonzero_elements_of_P_rows_only_flat(self, input_size):
        H = input_size // self.W
        out_H = 2 * H + 1
        out_W = self.W
        P_dict = {}
        for i in range(H):
            for j in range(out_W):
                input_flat_idx = i * out_W + j
                P_dict[input_flat_idx] = []
                for ki in range(3):  # 3-tap kernel
                    out_i = 2 * i + ki
                    if 0 <= out_i < out_H:
                        output_flat_idx = out_i * out_W + j
                        P_dict[input_flat_idx].append(output_flat_idx)
        return P_dict