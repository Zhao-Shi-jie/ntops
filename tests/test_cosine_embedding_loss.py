import pytest
import torch
import torch.nn.functional as F

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cosine_embedding_loss(shape, dtype, device, rtol, atol):
    if len(shape) == 1:
        shape = (1, shape[0])
    batch_size = shape[0]
    dim = shape[1]
    
    x1 = torch.randn(batch_size, dim, dtype=dtype, device=device)
    x2 = torch.randn(batch_size, dim, dtype=dtype, device=device)
    y = torch.randint(-1, 2, (batch_size,), device=device).float()
    y[y == 0] = 1  # 确保只有 +1 或 -1
    margin = torch.tensor(0.278, dtype=dtype, device=device)
    
    ninetoothed_output = ntops.torch.cosine_embedding_loss(x1, x2, y, margin=margin)
    reference_output = F.cosine_embedding_loss(x1, x2, y, margin=margin)
    
    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)