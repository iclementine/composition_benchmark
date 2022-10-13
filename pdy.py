import paddle


def add(x, y):
    
    out = paddle.add(x, y)
    return out


def sub(x, y):
    
    out = paddle.subtract(x, y)
    return out


def mul(x, y):
    
    out = paddle.multiply(x, y)
    return out


def div(x, y):
    
    out = paddle.divide(x, y)
    return out


def eq(x, y):
    
    return paddle.equal(x, y)


def ne(x, y):
    return paddle.logical_not(eq(x, y))


def logical_and(x, y):
    
    return paddle.logical_and(x, y)


def logical_or(x, y):
    
    return paddle.logical_or(x, y)


def maximum(x, y):
    
    out = paddle.maximum(x, y)
    return out


def minimum(x, y):
    
    out = paddle.minimum(x, y)
    return out


def power(x, y):
    
    out = paddle.pow(x, y)
    return out


def std(x, axis, keepdim=False):
    mu = paddle.mean(x, axis, keepdim=True)
    two = paddle.to_tensor([2.0], dtype=x.dtype)
    half = paddle.to_tensor([0.5], dtype=x.dtype)
    temp1 = sub(x, mu)
    temp2 = power(temp1, two)
    var = paddle.mean(temp2, axis, keepdim=keepdim)
    out = power(var, half)
    return out

def layer_norm(x, w, b, epsilon):
    mu = paddle.sum(x, -1, keepdim=True)
    sigma = std(x, -1, keepdim=True)
    nominator = sub(x, mu)
    denominator = add(sigma, epsilon)
    normalized = nominator / denominator
    out = add(w * normalized, b)
    return out


def transpose_last2_axes(x):
    rank_x = len(x.shape)
    perm = list(range(rank_x))
    perm[-1] = rank_x - 2
    perm[-2] = rank_x - 1
    return paddle.transpose(x, perm)


def matmul(x, y, transpose_x=False, transpose_y=False):
    if transpose_x:
        x = transpose_last2_axes(x)
    if transpose_y:
        y = transpose_last2_axes(y)
    out = paddle.matmul(x, y)
    return out


def linear(x, w, b):
    out = add(matmul(x, w), b)
    return out


def relu(x):
    zero = paddle.to_tensor(0, dtype=x.dtype)
    return maximum(x, zero)


def lookup(w, ids):
    return paddle.gather(w, ids)




# def layer_norm(x, w, b, epsilon):
#     mu = paddle.sum(x, -1, keepdim=True)
#     std = paddle.std(x, -1, keepdim=True)
#     out = w * (x - mu) / (std + epsilon) + b
#     return out


def lt(x, y):
    
    out = paddle.less_than(x, y)
    return out


def le(x, y):
    
    out = paddle.logical_or(paddle.less_than(x, y), paddle.equal(x, y))
    return out


def gt(x, y):
    
    out = paddle.greater_than(x, y)
    return out


def ge(x, y):
    
    out = paddle.logical_or(paddle.greater_than(x, y), paddle.equal(x, y))
    return out


def dropout(x, p, training=True):
    if not training:
        return x
    shape = paddle.shape(x)
    p_keep = 1.0 - p
    mask = lt(paddle.rand(shape, x.dtype), paddle.to_tensor(p_keep))
    mask = paddle.cast(mask, x.dtype)
    out = mul(mul(x, mask), paddle.to_tensor(1.0 / p_keep))
    return out


def softmax(x, axis, keepdim=False):
    max = paddle.max(x, axis, keepdim=True)
    shifted = sub(x, max)
    exp = paddle.exp(shifted)
    sum = paddle.sum(exp, axis, keepdim=True)
    out = div(exp, sum)
    return out


def log_softmax(x, axis, keepdim=False):
    out = log(softmax(x, axis, keepdim))
    return out


def id_mask(input, padding_index=0, dtype="bool"):
    """Generate mask with input ids. 
    Those positions where the value equals ``padding_index`` correspond to 0 or
    ``False``, otherwise, 1 or ``True``.
    Parameters
    ----------
    input : Tensor [dtype: int]
        The input tensor. It represents the ids.
    padding_index : int, optional
        The id which represents padding, by default 0.
    dtype : str, optional
        Data type of the returned mask, by default "bool".
    Returns
    -------
    Tensor
        The generate mask. It has the same shape as ``input`` does.
    """
    return paddle.cast(eq(input, paddle.to_tensor(padding_index)), dtype)


def feature_mask(input, axis, dtype="bool"):
    """Compute mask from input features.
    For a input features, represented as batched feature vectors, those vectors
    which all zeros are considerd padding vectors.
    Parameters
    ----------
    input : Tensor [dtype: float]
        The input tensor which represents featues.
    axis : int
        The index of the feature dimension in ``input``. Other dimensions are
        considered ``spatial`` dimensions.
    dtype : str, optional
        Data type of the generated mask, by default "bool"
    Returns
    -------
    Tensor
        The geenrated mask with ``spatial`` shape as mentioned above.
        It has one less dimension than ``input`` does.
    """
    feature_sum = paddle.sum(paddle.abs(input), axis)
    return paddle.cast(
        ne(feature_sum, paddle.to_tensor(0, dtype=feature_sum.dtype)), dtype)


def combine_mask(mask1, mask2):
    """Combine two mask with multiplication or logical and.
    Parameters
    -----------
    mask1 : Tensor
        The first mask.
    mask2 : Tensor
        The second mask with broadcastable shape with ``mask1``.
    Returns
    --------
    Tensor
        Combined mask.
    Notes
    ------
    It is mainly used to combine the padding mask and no future mask for
    transformer decoder. 
    Padding mask is used to mask padding positions of the decoder inputs and
    no future mask is used to prevent the decoder to see future information.
    """
    if mask1.dtype == paddle.fluid.core.VarDesc.VarType.BOOL:
        return logical_and(mask1, mask2)
    else:
        return mul(mask1, mask2)


def future_mask(time_steps, dtype="bool"):
    """Generate lower triangular mask.
    It is used at transformer decoder to prevent the decoder to see future
    information.
    Parameters
    ----------
    time_steps : int
        Decoder time steps.
    dtype : str, optional
        The data type of the generate mask, by default "bool".
    Returns
    -------
    Tensor
        The generated mask.
    """
    ids = paddle.arange(time_steps)
    mask = le(ids, ids.unsqueeze(-1))
    return paddle.cast(mask, dtype)

def sinusoid_position_encoding(num_positions: int,
                               feature_size: int,
                               omega: float=1.0,
                               start_pos: int=0,
                               dtype=None) -> paddle.Tensor:
    # return tensor shape (num_positions, feature_size)
    # NOTE: to be compatible with paddle's to_static, we cannnot raise 
    # an exception here, take care of it by yourself
    # if (feature_size % 2 != 0):
    #     raise ValueError("size should be divisible by 2")
    dtype = dtype or paddle.get_default_dtype()

    channel = paddle.arange(0, feature_size, 2, dtype=dtype)
    index = paddle.arange(start_pos, start_pos + num_positions, 1, dtype=dtype)
    nominator = mul(paddle.unsqueeze(index, -1), paddle.to_tensor(omega))
    denominator = power(paddle.to_tensor(10000.0), div(channel, paddle.to_tensor(float(feature_size))))
    p = div(nominator, denominator)
    encodings = paddle.zeros([num_positions, feature_size], dtype=dtype)
    encodings[:, 0::2] = paddle.sin(p)
    encodings[:, 1::2] = paddle.cos(p)
    return encodings