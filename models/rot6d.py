import torch
import torch.nn.functional as F


def rot6d_to_matrix(rot6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Handles both 2D tensors (batch, 6) and arbitrary batch shapes (..., 6).

    Args:
        rot6d: (batch, 6) or (..., 6) tensor where first 3 are x-axis, next 3 are y-axis

    Returns:
        R: (batch, 3, 3) or (..., 3, 3) rotation matrix
    """
    # Handle both 2D and higher-dimensional inputs
    original_shape = rot6d.shape

    # If input is not 2D, reshape to 2D for processing
    if len(original_shape) > 2:
        # Reshape to (total_batch_size, 6)
        rot6d = rot6d.reshape(-1, 6)

    # Extract x and y axes
    a1 = rot6d[:, 0:3]  # (batch, 3)
    a2 = rot6d[:, 3:6]  # (batch, 3)

    # Gram-Schmidt orthogonalization
    b1 = F.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)

    # Stack into (batch, 3, 3) matrix
    R = torch.stack([b1, b2, b3], dim=2)  # (batch, 3, 3)

    # If input was higher-dimensional, reshape back
    if len(original_shape) > 2:
        output_shape = list(original_shape[:-1]) + [3, 3]
        R = R.reshape(output_shape)

    return R
