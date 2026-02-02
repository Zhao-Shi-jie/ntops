import torch
import ntops
from ntops.torch.utils import _cached_make

def embedding(input, weight, out=None, max_norm=None, norm_type=2.0):
    if out is None:
        out = torch.empty(*input.shape, weight.shape[1], dtype=weight.dtype, device=input.device)
    
    
    # kernel = _cached_make(ntops.kernels.embedding.premake, input.dim(), weight.shape[0], weight.shape[1], block_size_m=4, block_size_n=4)
    # kernel(input, weight, out, max_norm, norm_type)
    
    # Find unique indices to reduce redundant computations, then map back
    # to original output positions. This is especially beneficial when input
    # contains many repeated indices. Otherwise, data races in parallelism will cause errors.
    unique_indices, inverse_indices = torch.unique(input.flatten(), return_inverse=True)
    temp_out = torch.empty(unique_indices.shape[0], weight.shape[1], 
                           dtype=weight.dtype, device=input.device)
    if max_norm is None:
        kernel = _cached_make(
            ntops.kernels.embedding.premake_without_norm, 
            unique_indices.dim(),
            dtype=weight.dtype,
            block_size_m=4, 
            block_size_n=4
        )
        kernel(unique_indices, weight, temp_out)
    else:
        kernel = _cached_make(
            ntops.kernels.embedding.premake, 
            unique_indices.dim(),
            embedding_dim=weight.shape[1],
            dtype=weight.dtype,
            block_size_m=4, 
            block_size_n=4
        )
        kernel(unique_indices, weight, temp_out, max_norm, norm_type)
        
    out_flat = out.view(-1, weight.shape[1])
    out_flat[:] = temp_out[inverse_indices]

    return out
