import math
import paddle
import pdy as fx
from paddle import nn
from paddle.nn import initializer as I


class Linear(nn.Layer):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.w = self.create_parameter((num_inputs, num_outputs))
        self.b = self.create_parameter((num_outputs, ), is_bias=True)

    def forward(self, x):
        return fx.linear(x, self.w, self.b)


class Dropout(nn.Layer):
    """
    Dropout Layer.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return fx.dropout(x, self.p, training=self.training)


class LayerNorm(nn.Layer):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.w = self.create_parameter(
            (features, ), default_initializer=nn.initializer.Constant(1.0))
        self.b = self.create_parameter(
            (features, ), default_initializer=nn.initializer.Constant(0.0))
        self.epsilon = epsilon

    def forward(self, x):
        return fx.layer_norm(x, self.w, self.b, paddle.to_tensor(self.epsilon))


class SublayerConnection(nn.Layer):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return fx.add(x, self.dropout(sublayer(self.norm(x))))


def scaled_dot_product_attention(q,
                                 k,
                                 v,
                                 mask=None,
                                 dropout=0.0,
                                 training=True):
    r"""Scaled dot product attention with masking. 
    
    Assume that q, k, v all have the same leading dimensions (denoted as * in 
    descriptions below). Dropout is applied to attention weights before 
    weighted sum of values.
    Parameters
    -----------
    q : Tensor [shape=(\*, T_q, d)]
        the query tensor.
    k : Tensor [shape=(\*, T_k, d)]
        the key tensor.
    v : Tensor [shape=(\*, T_k, d_v)]
        the value tensor.
    mask : Tensor, [shape=(\*, T_q, T_k) or broadcastable shape], optional
        the mask tensor, zeros correspond to paddings. Defaults to None.
    Returns
    ----------
    out : Tensor [shape=(\*, T_q, d_v)]
        the context vector.
    attn_weights : Tensor [shape=(\*, T_q, T_k)]
        the attention weights.
    """
    d = q.shape[-1]  # we only support imperative execution
    qk = fx.matmul(q, k, transpose_y=True)
    scaled_logit = fx.mul(qk, paddle.to_tensor(1.0 / math.sqrt(d)))

    if mask is not None:
        scaled_logit += fx.mul(fx.sub(paddle.to_tensor(1.0), mask),
                               paddle.to_tensor(-1e9))  # hard coded here

    attn_weights = fx.softmax(scaled_logit, axis=-1)
    attn_weights = fx.dropout(attn_weights, dropout, training=training)
    out = fx.matmul(attn_weights, v)
    return out, attn_weights


def _split_heads(x, num_heads):
    batch_size, time_steps, _ = x.shape
    x = paddle.reshape(x, [batch_size, time_steps, num_heads, -1])
    x = paddle.transpose(x, [0, 2, 1, 3])
    return x


def _concat_heads(x):
    batch_size, _, time_steps, _ = x.shape
    x = paddle.transpose(x, [0, 2, 1, 3])
    x = paddle.reshape(x, [batch_size, time_steps, -1])
    return x


class MultiheadAttention(nn.Layer):
    """Multihead Attention module.
    Parameters
    -----------
    model_dim: int
        The feature size of query.
    num_heads : int
        The number of attention heads.
    dropout : float, optional
        Dropout probability of scaled dot product attention and final context
        vector. Defaults to 0.0.
    k_dim : int, optional
        Feature size of the key of each scaled dot product attention. If not
        provided, it is set to ``model_dim / num_heads``. Defaults to None.
    v_dim : int, optional
        Feature size of the key of each scaled dot product attention. If not
        provided, it is set to ``model_dim / num_heads``. Defaults to None.
    Raises
    ---------
    ValueError
        If ``model_dim`` is not divisible by ``num_heads``.
    """

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 k_dim: int = None,
                 v_dim: int = None,
                 k_input_dim=None,
                 v_input_dim=None):
        super(MultiheadAttention, self).__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        depth = model_dim // num_heads
        k_dim = k_dim or depth
        v_dim = v_dim or depth
        k_input_dim = k_input_dim or model_dim
        v_input_dim = v_input_dim or model_dim
        self.affine_q = Linear(model_dim, num_heads * k_dim)
        self.affine_k = Linear(k_input_dim, num_heads * k_dim)
        self.affine_v = Linear(v_input_dim, num_heads * v_dim)
        self.affine_o = Linear(num_heads * v_dim, model_dim)

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout = dropout

    def forward(self, q, k, v, mask):
        """Compute context vector and attention weights.
        
        Parameters
        -----------
        q : Tensor [shape=(batch_size, time_steps_q, model_dim)]
            The queries.
        k : Tensor [shape=(batch_size, time_steps_k, model_dim)]
            The keys.
        v : Tensor [shape=(batch_size, time_steps_k, model_dim)]
            The values.
        mask : Tensor [shape=(batch_size, times_steps_q, time_steps_k] or broadcastable shape
            The mask.
        Returns
        ----------
        out : Tensor [shape=(batch_size, time_steps_q, model_dim)]
            The context vector.
        attention_weights : Tensor [shape=(batch_size, times_steps_q, time_steps_k)]
            The attention weights.
        """
        q = _split_heads(self.affine_q(q), self.num_heads)  # (B, h, T, C)
        k = _split_heads(self.affine_k(k), self.num_heads)
        v = _split_heads(self.affine_v(v), self.num_heads)
        mask = paddle.unsqueeze(mask, 1)  # unsqueeze for the h dim

        context_vectors, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, self.dropout, self.training)
        # NOTE: there is more sophisticated implementation: Scheduled DropHead
        context_vectors = _concat_heads(context_vectors)  # (B, T, h*C)
        out = self.affine_o(context_vectors)
        return out, attention_weights


class PositionwiseFFN(nn.Layer):
    """A faithful implementation of Position-wise Feed-Forward Network 
    in `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_.
    It is basically a 2-layer MLP, with relu actication and dropout in between.
    Parameters
    ----------
    input_size: int
        The feature size of the intput. It is also the feature size of the
        output.
    hidden_size: int
        The hidden size.
    dropout: float
        The probability of the Dropout applied to the output of the first
        layer, by default 0.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout=0.0):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = Linear(input_size, hidden_size)
        self.linear2 = Linear(hidden_size, input_size)
        self.dropout = Dropout(dropout)

        self.input_size = input_size
        self.hidden_szie = hidden_size

    def forward(self, x):
        r"""Forward pass of positionwise feed forward network.
        Parameters
        ----------
        x : Tensor [shape=(\*, input_size)]
            The input tensor, where ``\*`` means arbitary shape.
        Returns
        -------
        Tensor [shape=(\*, input_size)]
            The output tensor.
        """
        l1 = self.dropout(fx.relu(self.linear1(x)))
        l2 = self.linear2(l1)
        return l2


class TransformerEncoderLayer(nn.Layer):
    """
    Transformer encoder layer.
    """

    def __init__(self, d_model, n_heads, d_ffn, dropout=0.):
        """
        Args:
            d_model (int): the feature size of the input, and the output.
            n_heads (int): the number of heads in the internal MultiHeadAttention layer.
            d_ffn (int): the hidden size of the internal PositionwiseFFN.
            dropout (float, optional): the probability of the dropout in 
                MultiHeadAttention and PositionwiseFFN. Defaults to 0.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_mha = MultiheadAttention(d_model, n_heads)
        self.layer_norm1 = LayerNorm(d_model, epsilon=1e-6)

        self.ffn = PositionwiseFFN(d_model, d_ffn, dropout)
        self.layer_norm2 = LayerNorm(d_model, epsilon=1e-6)

        self.dropout = dropout

    def _forward_mha(self, x, mask):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        x_in = x
        x = self.layer_norm1(x)
        context_vector, attn_weights = self.self_mha(x, x, x, mask)
        context_vector = fx.add(
            x_in,
            fx.dropout(context_vector, self.dropout, training=self.training))
        return context_vector, attn_weights

    def _forward_ffn(self, x):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        x_in = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        out = fx.add(x_in, fx.dropout(x, self.dropout, training=self.training))
        return out

    def forward(self, x, mask):
        """
        Args:
            x (Tensor): shape(batch_size, time_steps, d_model), the decoder input.
            mask (Tensor): shape(batch_size, 1, time_steps), the padding mask.
        
        Returns:
            x (Tensor): shape(batch_size, time_steps, d_model), the decoded.
            attn_weights (Tensor), shape(batch_size, n_heads, time_steps, time_steps), self attention.
        """
        x, attn_weights = self._forward_mha(x, mask)
        x = self._forward_ffn(x)
        return x, attn_weights


class TransformerDecoderLayer(nn.Layer):
    """
    Transformer decoder layer.
    """

    def __init__(self, d_model, n_heads, d_ffn, dropout=0., d_encoder=None):
        """
        Args:
            d_model (int): the feature size of the input, and the output.
            n_heads (int): the number of heads in the internal MultiHeadAttention layer.
            d_ffn (int): the hidden size of the internal PositionwiseFFN.
            dropout (float, optional): the probability of the dropout in 
                MultiHeadAttention and PositionwiseFFN. Defaults to 0.
        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_mha = MultiheadAttention(d_model, n_heads, dropout)
        self.layer_norm1 = LayerNorm(d_model, epsilon=1e-6)

        self.cross_mha = MultiheadAttention(d_model,
                                            n_heads,
                                            dropout,
                                            k_input_dim=d_encoder,
                                            v_input_dim=d_encoder)
        self.layer_norm2 = LayerNorm(d_model, epsilon=1e-6)

        self.ffn = PositionwiseFFN(d_model, d_ffn, dropout)
        self.layer_norm3 = LayerNorm(d_model, epsilon=1e-6)

        self.dropout = dropout

    def _forward_self_mha(self, x, mask):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        x_in = x
        x = self.layer_norm1(x)
        context_vector, attn_weights = self.self_mha(x, x, x, mask)
        context_vector = fx.add(
            x_in,
            fx.dropout(context_vector, self.dropout, training=self.training))
        return context_vector, attn_weights

    def _forward_cross_mha(self, q, k, v, mask):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        q_in = q
        q = self.layer_norm2(q)
        context_vector, attn_weights = self.cross_mha(q, k, v, mask)
        context_vector = fx.add(
            q_in,
            fx.dropout(context_vector, self.dropout, training=self.training))
        return context_vector, attn_weights

    def _forward_ffn(self, x):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        x_in = x
        x = self.layer_norm3(x)
        x = self.ffn(x)
        out = fx.add(x_in, fx.dropout(x, self.dropout, training=self.training))
        return out

    def forward(self, q, k, v, encoder_mask, decoder_mask):
        """
        Args:
            q (Tensor): shape(batch_size, time_steps_q, d_model), the decoder input.
            k (Tensor): shape(batch_size, time_steps_k, d_model), keys.
            v (Tensor): shape(batch_size, time_steps_k, d_model), values
            encoder_mask (Tensor): shape(batch_size, 1, time_steps_k) encoder padding mask.
            decoder_mask (Tensor): shape(batch_size, time_steps_q, time_steps_q) or broadcastable shape, decoder padding mask.
        
        Returns:
            q (Tensor): shape(batch_size, time_steps_q, d_model), the decoded.
            self_attn_weights (Tensor), shape(batch_size, n_heads, time_steps_q, time_steps_q), decoder self attention.
            cross_attn_weights (Tensor), shape(batch_size, n_heads, time_steps_q, time_steps_k), decoder-encoder cross attention.
        """
        q, self_attn_weights = self._forward_self_mha(q, decoder_mask)
        q, cross_attn_weights = self._forward_cross_mha(q, k, v, encoder_mask)
        q = self._forward_ffn(q)
        return q, self_attn_weights, cross_attn_weights


class MLPPreNet(nn.Layer):
    """Decoder's prenet."""

    def __init__(self, d_input, d_hidden, d_output, dropout=0.0):
        # (lin + relu + dropout) * n + last projection
        super(MLPPreNet, self).__init__()
        self.lin1 = Linear(d_input, d_hidden)
        self.lin2 = Linear(d_hidden, d_hidden)
        self.lin3 = Linear(d_hidden, d_output)
        self.dropout = dropout

    def forward(self, x, dropout):
        l1 = fx.dropout(fx.relu(self.lin1(x)), self.dropout, training=True)
        l2 = fx.dropout(fx.relu(self.lin2(l1)), self.dropout, training=True)
        l3 = self.lin3(l2)
        return l3


class TransformerEncoder(nn.LayerList):

    def __init__(self, d_model, n_heads, d_ffn, n_layers, dropout=0.):
        super(TransformerEncoder, self).__init__()
        for _ in range(n_layers):
            self.append(
                TransformerEncoderLayer(d_model, n_heads, d_ffn, dropout))

    def forward(self, x, mask):
        """
        Args:
            x (Tensor): shape(batch_size, time_steps, feature_size), the input tensor.
            mask (Tensor): shape(batch_size, 1, time_steps), the mask.
            drop_n_heads (int, optional): how many heads to drop. Defaults to 0.
        Returns:
            x (Tensor): shape(batch_size, time_steps, feature_size), the context vector.
            attention_weights(list[Tensor]), each of shape
                (batch_size, n_heads, time_steps, time_steps), the attention weights.
        """
        attention_weights = []
        for layer in self:
            x, attention_weights_i = layer(x, mask)
            attention_weights.append(attention_weights_i)
        return x, attention_weights


class TransformerDecoder(nn.LayerList):

    def __init__(self,
                 d_model,
                 n_heads,
                 d_ffn,
                 n_layers,
                 dropout=0.,
                 d_encoder=None):
        super(TransformerDecoder, self).__init__()
        for _ in range(n_layers):
            self.append(
                TransformerDecoderLayer(d_model,
                                        n_heads,
                                        d_ffn,
                                        dropout,
                                        d_encoder=d_encoder))

    def forward(self, q, k, v, encoder_mask, decoder_mask):
        """
        Args:
            q (Tensor): shape(batch_size, time_steps_q, d_model)
            k (Tensor): shape(batch_size, time_steps_k, d_encoder)
            v (Tensor): shape(batch_size, time_steps_k, k_encoder)
            encoder_mask (Tensor): shape(batch_size, 1, time_steps_k)
            decoder_mask (Tensor): shape(batch_size, time_steps_q, time_steps_q)
        Returns:
            q (Tensor): shape(batch_size, time_steps_q, d_model), the output.
            self_attention_weights (List[Tensor]): shape (batch_size, num_heads, encoder_steps, encoder_steps)
            cross_attention_weights (List[Tensor]): shape (batch_size, num_heads, decoder_steps, encoder_steps)
        """
        self_attention_weights = []
        cross_attention_weights = []
        for layer in self:
            q, self_attention_weights_i, cross_attention_weights_i = layer(
                q, k, v, encoder_mask, decoder_mask)
            self_attention_weights.append(self_attention_weights_i)
            cross_attention_weights.append(cross_attention_weights_i)
        return q, self_attention_weights, cross_attention_weights


class Transformer(nn.Layer):

    def __init__(
        self,
        n_vocab: int,
        d_encoder: int,
        d_decoder: int,
        d_mel: int,
        n_heads: int,
        d_ffn: int,
        encoder_layers: int,
        decoder_layers: int,
        d_prenet: int,
        d_postnet: int,
        max_reduction_factor: int,
        decoder_prenet_dropout: float,
        dropout: float,
    ):
        super(Transformer, self).__init__()

        # encoder
        self.encoder_prenet = nn.Embedding(n_vocab,
                                           d_encoder,
                                           padding_idx=0,
                                           weight_attr=I.Uniform(-0.05, 0.05))
        # position encoding matrix may be extended later
        self.encoder_pe = fx.sinusoid_position_encoding(1000, d_encoder)

        self.encoder_pe_scalar = self.create_parameter([1],
                                                       attr=I.Constant(1.))
        self.encoder = TransformerEncoder(d_encoder, n_heads, d_ffn,
                                          encoder_layers, dropout)

        # decoder
        self.decoder_prenet = MLPPreNet(d_mel, d_prenet, d_decoder, dropout)
        self.decoder_pe = fx.sinusoid_position_encoding(1000, d_decoder)
        self.decoder_pe_scalar = self.create_parameter([1],
                                                       attr=I.Constant(1.))
        self.decoder = TransformerDecoder(d_decoder,
                                          n_heads,
                                          d_ffn,
                                          decoder_layers,
                                          dropout,
                                          d_encoder=d_encoder)
        self.final_proj = Linear(d_decoder, max_reduction_factor * d_mel)
        self.decoder_postnet = MLPPreNet(d_mel, d_postnet, d_mel)
        self.stop_conditioner = Linear(d_mel, 3)

        # specs
        self.padding_idx = 0
        self.d_encoder = d_encoder
        self.d_decoder = d_decoder
        self.d_mel = d_mel
        self.max_r = max_reduction_factor
        self.dropout = dropout
        self.decoder_prenet_dropout = decoder_prenet_dropout

        # start and end: though it is only used in predict
        # it can also be used in training
        dtype = paddle.get_default_dtype()
        self.start_vec = paddle.full([1, d_mel], 0.5, dtype=dtype)
        self.end_vec = paddle.full([1, d_mel], -0.5, dtype=dtype)
        self.stop_prob_index = 2

        # mutables
        self.r = max_reduction_factor  # set it every call
        self.drop_n_heads = 0

    def forward(self, text, mel):
        encoded, encoder_attention_weights, encoder_mask = self.encode(text)
        mel_output, mel_intermediate, cross_attention_weights, stop_logits = self.decode(
            encoded, mel, encoder_mask)
        outputs = {
            "mel_output": mel_output,
            "mel_intermediate": mel_intermediate,
            "encoder_attention_weights": encoder_attention_weights,
            "cross_attention_weights": cross_attention_weights,
            "stop_logits": stop_logits,
        }
        return outputs

    def encode(self, text):
        T_enc = text.shape[-1]
        embed = self.encoder_prenet(text)
        if embed.shape[1] > self.encoder_pe.shape[0]:
            new_T = max(embed.shape[1], self.encoder_pe.shape[0] * 2)
            self.encoder_pe = fx.sinusoid_position_encoding(
                new_T, self.d_encoder)
        pos_enc = self.encoder_pe[:T_enc, :]  # (T, C)
        x = fx.add(fx.mul(embed, paddle.to_tensor(math.sqrt(self.d_encoder))),
                   fx.mul(pos_enc, self.encoder_pe_scalar))
        x = fx.dropout(x, self.dropout, training=self.training)

        # TODO(chenfeiyu): unsqueeze a decoder_time_steps=1 for the mask
        encoder_padding_mask = paddle.unsqueeze(
            fx.id_mask(text, self.padding_idx, dtype=x.dtype), 1)
        x, attention_weights = self.encoder(x, encoder_padding_mask)
        return x, attention_weights, encoder_padding_mask

    def decode(self, encoder_output, input, encoder_padding_mask):
        batch_size, T_dec, mel_dim = input.shape

        x = self.decoder_prenet(input, self.decoder_prenet_dropout)
        # twice its length if needed
        if x.shape[1] * self.r > self.decoder_pe.shape[0]:
            new_T = max(x.shape[1] * self.r, self.decoder_pe.shape[0] * 2)
            self.decoder_pe = fx.sinusoid_position_encoding(
                new_T, self.d_decoder)
        pos_enc = self.decoder_pe[:T_dec * self.r:self.r, :]
        x = fx.add(fx.mul(x, paddle.to_tensor(math.sqrt(self.d_decoder))),
                   fx.mul(pos_enc, self.decoder_pe_scalar))
        x = fx.dropout(x, self.dropout, training=self.training)

        no_future_mask = fx.future_mask(T_dec, dtype=input.dtype)
        decoder_padding_mask = fx.feature_mask(input,
                                               axis=-1,
                                               dtype=input.dtype)
        decoder_mask = fx.combine_mask(decoder_padding_mask.unsqueeze(1),
                                       no_future_mask)
        decoder_output, _, cross_attention_weights = self.decoder(
            x, encoder_output, encoder_output, encoder_padding_mask,
            decoder_mask)

        # use only parts of it
        output_proj = self.final_proj(decoder_output)[:, :, :self.r * mel_dim]
        mel_intermediate = paddle.reshape(output_proj,
                                          [batch_size, -1, mel_dim])
        stop_logits = self.stop_conditioner(mel_intermediate)

        # cnn postnet
        mel_output = self.decoder_postnet(mel_intermediate, self.dropout)

        return mel_output, mel_intermediate, cross_attention_weights, stop_logits