import torch
import ntops
from ntops.torch.utils import _cached_make

def embedding(input, weight, out=None):
    if out is None:
        out = torch.empty(*input.shape, weight.shape[1], dtype=weight.dtype, device=input.device)
    
    
    kernel = _cached_make(ntops.kernels.embedding.premake, input.dim(), weight.shape[0], weight.shape[1], block_size_m=4, block_size_n=4)
    kernel(input, weight, out, padding_idx)
    return out