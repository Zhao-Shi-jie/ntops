import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


def generate_arguments():
    return "shape, dtype, device, rtol, atol", [
        ([9], torch.float32, 'cuda', 0.001, 0.001),
    ]

@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_embedding(shape, dtype, device, rtol, atol):
    vocab_size = 100
    embedding_dim = 33
    
    input = torch.randint(0, vocab_size, shape, device=device)
    weight = torch.randn(vocab_size, embedding_dim, dtype=dtype, device=device)

    ninetoothed_output = ntops.torch.embedding(input, weight, max_norm=10, norm_type=1.0)
    reference_output = torch.nn.functional.embedding(input, weight, padding_idx=10, max_norm=10,
                                                    norm_type=1.0, scale_grad_by_freq=False, sparse=False)
    # 逐行打印输出以进行调试
    print("Ninetoothed output:", ninetoothed_output)
    print("Reference output:", reference_output)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
