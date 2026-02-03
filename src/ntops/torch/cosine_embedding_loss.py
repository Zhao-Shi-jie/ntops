import torch
import ntops
from ntops.torch.utils import _cached_make, _get_matmul_input_precision


def cosine_embedding_loss(x1, x2, y, margin=0.0, reduction='mean'):
    embedding_dim = x1.shape[-1]
    dims = len(x1.shape)
    output = torch.empty(x1.shape[:-1], dtype=x1.dtype, device=x1.device)
    kernel_loss = _cached_make(
        ntops.kernels.cosine_embedding_loss.cosine_embedding_loss_premake,
        dims = dims,
        embedding_dim = embedding_dim,
        block_size = 16,
    )
    kernel_loss(x1, x2, y, margin, output, _get_matmul_input_precision())
    
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:
        return output
