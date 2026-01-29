import enum
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE = ninetoothed.block_size()

class InputPrecisionVariant(enum.IntEnum):
    TF32 = enum.auto()

    IEEE = enum.auto()

def cosine_embedding_loss_arrangement(*tensors, block_size=None, batch_size=None, dim=None):
    """用于计算余弦相似度的arrangement"""
    if block_size is None:
        block_size = BLOCK_SIZE
    if batch_size is None:
        batch_size = 32
    if dim is None:
        dim = 128
    
    x1, x2, y, margin, output, input_precision = tensors
    
    # x1, x2: [batch, dim] -> 按batch维度tile
    # x1_arranged = x1.tile((1, dim))
    # x1_arranged = x1_arranged.squeeze(1)
    # x1_arranged = x1_arranged.tile((block_size,))
    # x2_arranged = x2.tile((1, dim))
    # x2_arranged = x2_arranged.squeeze(1)
    # x2_arranged = x2_arranged.tile((block_size,))
    x_1_arranged = x1.tile((1, block_size))
    x1_arranged = x_1_arranged.tile((1, -1))
    # x1_arranged = x1_arranged.tile((block_size,))

    x_2_arranged = x2.tile((1, block_size))
    x2_arranged = x_2_arranged.tile((1, -1))
    # x2_arranged = x2_arranged.tile((block_size,))

    # y, margin: [batch] -> 按batch维度tile
    y_arranged = y.tile((1,))

    margin_arranged = margin.expand((batch_size,))
    margin_arranged = margin_arranged.tile((1,))

    
    # output: [batch] -> 按batch维度tile
    output_arranged = output.tile((1,))
    
    return x1_arranged, x2_arranged, y_arranged, margin_arranged, output_arranged, input_precision


def cosine_embedding_loss_application(x1, x2, y, margin, output, input_precision):
    """计算余弦相似度"""
    # 计算点积
    # accumulator_dot_product = ntl.zeros([], dtype=ntl.float32)
    # accumulator_norm1_sq = ntl.zeros([], dtype=ntl.float32)
    # accumulator_norm2_sq = ntl.zeros([], dtype=ntl.float32)

    if input_precision == 2:  # InputPrecisionVariant.IEEE:
        input_precision_: ntl.constexpr = "ieee"
    else:
        input_precision_: ntl.constexpr = "tf32"

    margin_value = margin[0]
    y_value = y[0]

    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0 
    for i in range(x1.shape[0]):
        dot_product += ntl.sum(x1[i] * x2[i])
        norm1_sq += ntl.sum(x1[i] * x1[i])
        norm2_sq += ntl.sum(x2[i] * x2[i])
    norm1 = ntl.sqrt(norm1_sq)
    norm2 = ntl.sqrt(norm2_sq)
    
    cosine = dot_product / (norm1 * norm2)  

    if y_value > 0:
        output[0] = 1.0 - cosine
    else:
        diff = cosine - margin_value
        output[0] = ntl.maximum(0.0, diff)
    # output[0] = 1.0 - cosine
    # diff = cosine - margin_value
    # output[0] = ntl.maximum(0.0, diff)

    # positive_loss = 1.0 - cosine
    # negative_loss = ntl.maximum(0.0, cosine - margin_value)
    # result = ntl.where(y_value > 0, positive_loss, negative_loss)
    # output[0] = result


def cosine_embedding_loss_premake(dtype=None, 
                                  block_size=None, 
                                  batch_size=None, 
                                  dim=None,
                                  input_precision=None,):
    arrangement_ = functools.partial(
        cosine_embedding_loss_arrangement,
        block_size=block_size,
        batch_size=batch_size,
        dim=dim,
    )
    
    tensors = (
        Tensor(2, dtype=dtype),  # x1: [batch, dim]
        Tensor(2, dtype=dtype),  # x2: [batch, dim]
        Tensor(1, dtype=ninetoothed.int32),  # y: [batch] 标签
        Tensor(1, dtype=dtype),  # margin: 常量
        Tensor(1, dtype=dtype),  # output: [batch]
        Tensor(0, constexpr=True, value=input_precision),
    )
    
    return arrangement_, cosine_embedding_loss_application, tensors
