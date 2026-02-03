import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments

def manual_cosine_embedding_loss(x1, x2, y, margin=0.0, reduction='mean'):
    """
    手写的 cosine embedding loss 参考实现
    
    Args:
        x1: shape (N, D) or (D,)
        x2: shape (N, D) or (D,)
        y: shape (N,) or (), values in {-1, 1}
        margin: float
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss tensor
    """
    # 确保输入是 float 类型
    x1 = x1.float()
    x2 = x2.float()
    
    # 处理 1D 输入
    if x1.dim() == 1:
        x1 = x1.unsqueeze(0)  # (D,) -> (1, D)
        x2 = x2.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)  # () -> (1,)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 计算余弦相似度
    # cosine = (x1 · x2) / (||x1|| * ||x2||)
    dot_product = (x1 * x2).sum(dim=-1)  # shape: (N,)
    norm_x1 = x1.norm(p=2, dim=-1)       # shape: (N,)
    norm_x2 = x2.norm(p=2, dim=-1)       # shape: (N,)
    
    # 避免除零
    cosine = dot_product / (norm_x1 * norm_x2 + 1e-8)
    
    # 计算损失
    # 如果 y = 1 (相似)：loss = 1 - cosine
    # 如果 y = -1 (不相似)：loss = max(0, cosine - margin)
    loss = torch.where(
        y == 1,
        1.0 - cosine,
        torch.clamp(cosine - margin, min=0.0)
    )
    
    # 恢复原始形状
    if squeeze_output:
        loss = loss.squeeze(0)  # (1,) -> ()
    
    # 应用 reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def generate_arguments():
    return 'shape,dtype,device,rtol,atol', [
        ((334,), torch.float16, 'cuda', 1e-2, 1e-2),]
@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_cosine_embedding_loss(shape, dtype, device, rtol, atol):
    if len(shape) > 2:
        pytest.skip("Skipping test for tensors with more than 2 dimensions.")
    else:
        x1 = torch.randn(shape, dtype=dtype, device=device)
        x2 = torch.randn(shape, dtype=dtype, device=device)
        if len(shape) == 1:
            y = torch.randint(-1, 2, (1,), device=device).float()
            y[y == 0] = 1
            y = y.squeeze()
        else:
            y = torch.randint(-1, 2, shape[:-1], device=device).float()
            y[y == 0] = 1
        margin = 0.5
        
        manual_output = manual_cosine_embedding_loss(
        x1.clone(), x2.clone(), y.clone(), 
        margin=margin, 
        reduction='none')
        ninetoothed_output = ntops.torch.cosine_embedding_loss(x1.clone(), x2.clone(), y.clone(), margin=margin, reduction='none')
        reference_output = torch.nn.functional.cosine_embedding_loss(x1, x2, y, margin=margin, reduction='none')
        print("Input x1:", x1)
        print("Ninetoothed output:", ninetoothed_output)
        print("Reference output:", reference_output)
        print("Manual output:", manual_output)

        
        assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
