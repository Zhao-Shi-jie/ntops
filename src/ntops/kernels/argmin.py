import enum
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE = ninetoothed.block_size()

def arrangement(input, output, axis):
    input = input.flatten()
    output = output.flatten()
    input_arranged = input.tile((512 * 4,))
    output_arranged = output.tile((512 * 4,))
    axis_arranged = axis

    return input_arranged, output_arranged, axis_arranged


def application(input, output, axis):
    output = ntl.argmin(input, axis=0, keep_dims=True)


def premake(dtype=None, dims=None):
    arrangement_ = functools.partial(
        arrangement,
    )
    
    tensors = (
        Tensor(dims, dtype=dtype),
        Tensor(dims, dtype=ninetoothed.int64),
        Tensor(0, dtype=ninetoothed.int32),
    )
    
    return arrangement_, application, tensors
