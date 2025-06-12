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