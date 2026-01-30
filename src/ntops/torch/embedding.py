import torch
import ntops
from ntops.torch.utils import _cached_make

def embedding(input, weight, out=None, max_norm=None, norm_type=2.0):
    if out is None:
        out = torch.empty(*input.shape, weight.shape[1], dtype=weight.dtype, device=input.device)
    
    
    kernel = _cached_make(ntops.kernels.embedding.premake, input.dim(), weight.shape[0], weight.shape[1], block_size_m=4, block_size_n=4)
    kernel(input, weight, out, max_norm, norm_type)
    return out