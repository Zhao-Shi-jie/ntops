import functools
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE = ninetoothed.block_size()

def arrangement(
    input,
    weight,
    output,
    block_size_m=None,
    block_size_n=None,
):
    if block_size_m is None: 
        block_size_m = BLOCK_SIZE
    if block_size_n is None: 
        block_size_n = BLOCK_SIZE

    
    output = output.flatten(start_dim=0, end_dim=-1)
    input = input.flatten()
    # output:  [N, embedding_dim] -> tile to [block_size_m, block_size_n]
    output_arranged = output.tile((1, block_size_n))
    output_arranged = output_arranged.tile((block_size_m, 1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)
    
    # input: [N] -> tile to [block_size_m], 这是索引
    input = input.unsqueeze(1) # 增加一个维度以便进行expand
    input_arranged = input.tile((block_size_m, 1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(1)
    
    # weight: [vocab_size, embedding_dim] -> tile embedding_dim dimension
    # 然后 expand 到匹配 output 的 batch 维度
    weight_arranged = weight.tile((1, block_size_n))
    weight_arranged = weight_arranged.tile((-1, 1))  # 保持所有 vocab 行
    weight_arranged = weight_arranged.expand((output_arranged.shape[0], -1))
    weight_arranged.dtype = weight_arranged.dtype.squeeze(1)


    return input_arranged, weight_arranged, output_arranged


def application(input, weight, output):
    # input 是索引，weight 是嵌入矩阵
    for i in range(output.shape[0]):
        idx = input[i]
        output[i] = weight[idx]


def premake(ndim, vocab_size, embedding_dim, dtype=None, block_size_m=None, block_size_n=None):
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )
    
    tensors = (
        Tensor(ndim, dtype=ninetoothed.int64),                                      # input:  [N]
        # Tensor(2, dtype=dtype, shape_options=shape_options_emb),                # weight: [vocab_size, embedding_dim]
        # Tensor(2, dtype=dtype, shape_options=shape_options_emb),                # output: [N, embedding_dim]
        Tensor(2, dtype=dtype),                                                 # weight: [vocab_size, embedding_dim]
        Tensor(ndim + 1, dtype=dtype),                                                 # output: [N, embedding_dim]
    )
    
    return arrangement_, application, tensors