import torch
import ntops
from ntops.torch.utils import _cached_make


def argmin(input, axis=None, keepdim=False):
    if axis is None:
        target_dim = 0
        axis_is_none = True
    else:
        target_dim = axis
        axis_is_none = False

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

    print(f"input shape: {input.shape}, output shape: {output.shape}, axis: {axis}, keepdim: {keepdim}")
    print(f"input dtype: {input.dtype}, output dtype: {output.dtype}, output dim: {output.dim()}")
    print(f"target_dim: {target_dim}, axis_is_none: {axis_is_none}")

    kernel = _cached_make(
        ntops.kernels.argmin.premake,
        input.shape[target_dim] if not axis_is_none else 1,
        dtype=input.dtype,
        in_dims=input.dim(),
        out_dims=output.dim(),
        axis=target_dim,
        axis_is_none=axis_is_none,
        keep_dims=keepdim,
    )

    kernel(
        input,
        output,
        input.shape[target_dim] if not axis_is_none else 1,
    )

    return output
