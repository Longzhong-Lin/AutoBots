import copy
import math
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden=128, bias=True, activation="relu", norm='layer'):
        super(MLP, self).__init__()

        # define the activation function
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "relu6":
            act_layer = nn.ReLU6
        elif activation == "leaky":
            act_layer = nn.LeakyReLU
        elif activation == "prelu":
            act_layer = nn.PReLU
        else:
            raise NotImplementedError

        # define the normalization function
        if norm == "layer":
            norm_layer = nn.LayerNorm
        elif norm == "batch":
            norm_layer = nn.BatchNorm1d
        else:
            raise NotImplementedError

        # insert the layers
        self.linear1 = nn.Linear(in_channel, hidden, bias=bias)
        self.linear1.apply(self._init_weights)
        self.linear2 = nn.Linear(hidden, out_channel, bias=bias)
        self.linear2.apply(self._init_weights)

        self.norm1 = norm_layer(hidden)
        self.norm2 = norm_layer(out_channel)

        self.act1 = act_layer(inplace=True)
        self.act2 = act_layer(inplace=True)

        self.shortcut = None
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=bias),
                norm_layer(out_channel)
            )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if not m.bias is None:
                m.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
        return self.act2(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        '''
        :param x: (T, B, D)
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SocialAttention(nn.Module):
    def __init__(self, dim, num_heads, rel_dim=4, qkv_bias=True, qk_scale=None, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)

        self.rel_mlp = nn.Sequential(
            nn.Linear(rel_dim, 512, bias=True), nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
            # nn.Linear(512, head_dim * 3, bias=False)
        )

        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * 3), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dim * 3, dim), 
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, relation=None, return_weights=False):
        """
        :param x: (B, A, D)
        :param mask: (B, A)
        :param relation: (B, A, A, 4)
        """
        B, A, D = x.shape

        # attention block
        res_attn = x
        qkv = self.qkv(x).reshape(B, A, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4) # (3, B, nH, A, dH)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, nH, A, dH)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B, nH, A, A)
        if relation is not None:
            rel_emb = self.rel_mlp(relation).permute(0, 3, 1, 2) # (B, nH, A, A)
            rel_bias = 16 * torch.sigmoid(rel_emb)
            attn = attn + rel_bias
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1)) # (B, nH, A, A)
        # if relation is not None:
        #     rel_emb = self.rel_mlp(relation).reshape(B, 1, A, A, D//self.num_heads, 3).repeat(1, self.num_heads, 1, 1, 1, 1) # (B, nH, A, A, dH, 3)
        #     rel_q = (q.unsqueeze(3) * rel_emb[..., 0]).sum(-1)
        #     rel_k = ((k*self.scale).unsqueeze(2) * rel_emb[..., 1]).sum(-1)
        #     rel_v = rel_emb[..., 2]
        #     attn = attn + rel_q + rel_k
        if mask is not None:
            mask_add = torch.zeros_like(mask, dtype=torch.float)
            mask_add = torch.masked_fill(mask_add, mask, float('-inf'))
            attn = attn + mask_add.view(B, 1, 1, A)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, A, D)
        # x = attn @ v
        # if relation is not None:
        #     x = x + (attn.unsqueeze(-1) * rel_v).sum(-2)
        # x = x.transpose(1, 2).reshape(B, A, D)
        x = self.dropout1(x)
        x = self.norm1(x + res_attn)

        # feedforward block
        res_ff = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.norm2(x + res_ff)

        if return_weights:
            attn = attn.mean(1)
            return x, attn
        else:
            return x


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, return_weights: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x_attn, w = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, return_weights)
            x = x + x_attn
            x = x + self._ff_block(self.norm2(x))
        else:
            x_attn, w = self._sa_block(x, src_mask, src_key_padding_mask, return_weights)
            x = self.norm1(x + x_attn)
            x = self.norm2(x + self._ff_block(x))

        return x, w

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  need_weights: bool) -> Tensor:
        x, w = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=need_weights)
        return self.dropout1(x), w

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, return_weights: bool = False) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        weights = list()
        for mod in self.layers:
            output, layer_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, return_weights=return_weights)
            weights.append(layer_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, weights


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


if __name__ == "__main__":
    social_attn_layer = SocialAttention(dim=128, num_heads=16).cuda()
    x = torch.randn(4, 9, 128).cuda()
    mask = ((torch.rand(4, 9)-0.5) > 0).cuda()
    relation = torch.randn(4, 9, 9, 4).cuda()
    social_attn_layer(x, mask, relation)