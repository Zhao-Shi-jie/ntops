import torch
import ntops
from ntops.torch.utils import _cached_make, _get_matmul_input_precision


def cosine_embedding_loss(x1, x2, y, margin=0.0, reduction='mean'):
    """
    计算 Cosine Embedding Loss
    
    Args:
        x1: 第一个嵌入向量 [batch, dim]
        x2: 第二个嵌入向量 [batch, dim]
        y: 标签，+1表示相似，-1表示不相似 [batch]
        margin: 边界阈值
        reduction: 'mean', 'sum', 或 'none'
    """
    batch_size, dim = x1.shape
    margin_tensor = margin.unsqueeze(0)
    output = torch.empty(batch_size, dtype=x1.dtype, device=x1.device)
    kernel_loss = _cached_make(
        ntops.kernels.cosine_embedding_loss.cosine_embedding_loss_premake,
        batch_size = batch_size,
        dim = dim,
        block_size = 16,
    )
    kernel_loss(x1, x2, y, margin_tensor, output, _get_matmul_input_precision())
    
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:
        return output
