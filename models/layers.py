import math
import torch
import torch.nn as nn


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
        )

        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * 3), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dim * 3, dim), 
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, relation=None):
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
        if mask is not None:
            mask_add = torch.zeros_like(mask, dtype=torch.float)
            mask_add = torch.masked_fill(mask_add, mask, float('-inf'))
            attn = attn + mask_add.view(B, 1, 1, A)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, A, D)
        x = self.dropout1(x)
        x = self.norm1(x + res_attn)

        # feedforward block
        res_ff = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.norm2(x + res_ff)

        return x


if __name__ == "__main__":
    social_attn_layer = SocialAttention(dim=128, num_heads=16).cuda()
    x = torch.randn(4, 9, 128).cuda()
    mask = ((torch.rand(4, 9)-0.5) > 0).cuda()
    relation = torch.randn(4, 9, 9, 4).cuda()
    social_attn_layer(x, mask, relation)