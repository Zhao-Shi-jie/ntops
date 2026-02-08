import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


def generate_arguments():
    return 'shape,dtype,device,rtol,atol', [
        ((4, 512,), torch.float16, 'cuda', 1e-2, 1e-2),]
@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_argmin(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    axis = None
    keepdim = True

    output = ntops.torch.argmin(input, axis=axis, keepdim=keepdim)

    reference = torch.argmin(input, dim=axis, keepdim=keepdim)

    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
