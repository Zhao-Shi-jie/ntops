import torch
import ntops
from ntops.torch.utils import _cached_make


def argmin(input, axis=None, keepdim=False):
    if axis is None:
        axis = tuple(range(input.dim()))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)

    output_shape = list(input.shape)
    for ax in axis:
        output_shape[ax] = 1 if keepdim else 0
    output_shape = tuple([s for s in output_shape if s > 0])

    output = torch.empty(output_shape, dtype=torch.int64, device=input.device)

    if keepdim:
        kernel = _cached_make(
            ntops.kernels.argmin.premake,
            dtype=input.dtype,
            dims=input.dim(),
        )

        kernel(
            input,
            output,
            None,
        )

    return output
