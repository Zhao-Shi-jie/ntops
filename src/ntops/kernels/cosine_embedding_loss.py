import enum
import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE = ninetoothed.block_size()

class InputPrecisionVariant(enum.IntEnum):
    TF32 = enum.auto()

    IEEE = enum.auto()

def cosine_embedding_loss_arrangement(
        x1, 
        x2,
        y, 
        margin, 
        output, 
        input_precision, 
        block_size=None,
        dims=None,
        embedding_dim=None):
    """用于计算余弦相似度的arrangement"""
    if block_size is None:
        block_size = BLOCK_SIZE
    
    if dims == 1:
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        y = y.unsqueeze(0)
        output = output.unsqueeze(0)

    x1_arranged = x1.flatten(start_dim=0, end_dim=-1)
    x1_arranged = x1_arranged.tile((1, embedding_dim))
    x1_arranged = x1_arranged.squeeze(1)
    x1_arranged.dtype = x1_arranged.dtype.squeeze(0)
    x1_arranged = x1_arranged.tile((1,))


    x2_arranged = x2.flatten(start_dim=0, end_dim=-1)
    x2_arranged = x2_arranged.tile((1, embedding_dim))
    x2_arranged = x2_arranged.squeeze(1)
    x2_arranged.dtype = x2_arranged.dtype.squeeze(0)
    x2_arranged = x2_arranged.tile((1,))

    y_arranged = y.flatten()
    y_arranged = y_arranged.tile((1,))

    # margin_arranged = margin

    output_arranged = output.flatten()
    output_arranged = output_arranged.tile((1,))
    
    return x1_arranged, x2_arranged, y_arranged, margin, output_arranged, input_precision


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

    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    dot_product += ntl.sum(x1[0] * x2[0])
    norm1_sq += ntl.sum(x1[0] * x1[0])
    norm2_sq += ntl.sum(x2[0] * x2[0])
    norm1 = ntl.sqrt(norm1_sq)
    norm2 = ntl.sqrt(norm2_sq)
    
    cosine = dot_product / (norm1 * norm2)  

    if y[0] > 0:
        output[0] = 1.0 - cosine
    else:
        diff = cosine - margin
        output[0] = ntl.maximum(0.0, diff)


def cosine_embedding_loss_premake(dtype=None, 
                                  block_size=None, 
                                  dims=None, 
                                  embedding_dim=None,
                                  input_precision=None,):
    import math
    embedding_dim_power_of_2 = 2 ** math.ceil(math.log2(embedding_dim)) if embedding_dim > 0 else 1
    arrangement_ = functools.partial(
        cosine_embedding_loss_arrangement,
        block_size=block_size,
        dims=dims,
        embedding_dim=embedding_dim_power_of_2,
    )
    
    tensors = (
        Tensor(dims, dtype=dtype),
        Tensor(dims, dtype=dtype),
        Tensor(dims-1, dtype=ninetoothed.int32),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(dims-1, dtype=dtype),
        Tensor(0, constexpr=True, value=input_precision),
    )
    
    return arrangement_, cosine_embedding_loss_application, tensors
