# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from math import sqrt
# import numpy as np
#
# class TriangularCausalMask():
#     def __init__(self, B, L, device="cpu"):
#         mask_shape = [B, 1, L, L]
#         with torch.no_grad():
#             self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
#
#     @property
#     def mask(self):
#         return self._mask
#
#
# class ProbMask():
#     def __init__(self, B, H, L, index, scores, device="cpu"):
#         _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
#         _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
#         indicator = _mask_ex[torch.arange(B)[:, None, None],
#                     torch.arange(H)[None, :, None],
#                     index, :].to(device)
#         self._mask = indicator.view(scores.shape).to(device)
#
#     @property
#     def mask(self):
#         return self._mask
# class FixedEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(FixedEmbedding, self).__init__()
#
#         w = torch.zeros(c_in, d_model).float()
#         w.require_grad = False
#
#         position = torch.arange(0, c_in).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
#
#         w[:, 0::2] = torch.sin(position * div_term)
#         w[:, 1::2] = torch.cos(position * div_term)
#
#         self.emb = nn.Embedding(c_in, d_model)
#         self.emb.weight = nn.Parameter(w, requires_grad=False)
#
#     def forward(self, x):
#         return self.emb(x).detach()
# class TemporalEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='fixed', freq='h'):
#         super(TemporalEmbedding, self).__init__()
#
#         minute_size = 4
#         hour_size = 24
#         weekday_size = 7
#         day_size = 32
#         month_size = 13
#
#         Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
#         if freq == 't':
#             self.minute_embed = Embed(minute_size, d_model)
#         self.hour_embed = Embed(hour_size, d_model)
#         self.weekday_embed = Embed(weekday_size, d_model)
#         self.day_embed = Embed(day_size, d_model)
#         self.month_embed = Embed(month_size, d_model)
#
#     def forward(self, x):
#         x = x.long()
#
#         minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
#         hour_x = self.hour_embed(x[:, :, 3])
#         weekday_x = self.weekday_embed(x[:, :, 2])
#         day_x = self.day_embed(x[:, :, 1])
#         month_x = self.month_embed(x[:, :, 0])
#
#         return hour_x + weekday_x + day_x + month_x + minute_x
# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEmbedding, self).__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False
#
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return self.pe[:, :x.size(1)]
# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if torch.__version__ >= '1.5.0' else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
#                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#
#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return x
#
# class TimeFeatureEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='timeF', freq='h'):
#         super(TimeFeatureEmbedding, self).__init__()
#
#         freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
#         d_inp = freq_map[freq]
#         self.embed = nn.Linear(d_inp, d_model, bias=False)
#
#     def forward(self, x):
#         return self.embed(x)
# class DataEmbedding(nn.Module):
#     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
#         super(DataEmbedding, self).__init__()
#
#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
#         self.position_embedding = PositionalEmbedding(d_model=d_model)
#         self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
#                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
#             d_model=d_model, embed_type=embed_type, freq=freq)
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x, x_mark):
#         x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
#         return self.dropout(x)
#
#
# class DSAttention(nn.Module):
#     '''De-stationary Attention'''
#
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(DSAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)
#
#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)
#
#         tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
#         delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S
#
#         # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
#         scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
#
#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)
#
#             scores.masked_fill_(attn_mask.mask, -np.inf)
#
#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)
#
#         if self.output_attention:
#             return (V.contiguous(), A)
#         else:
#             return (V.contiguous(), None)
#
#
# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()
#
#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)
#
#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads
#
#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads
#
#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)
#
#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau, delta
#         )
#         out = out.view(B, L, -1)
#
#         return self.out_projection(out), attn
#
#
#
#
# class ConvLayer(nn.Module):
#     def __init__(self, c_in):
#         super(ConvLayer, self).__init__()
#         self.downConv = nn.Conv1d(in_channels=c_in,
#                                   out_channels=c_in,
#                                   kernel_size=3,
#                                   padding=2,
#                                   padding_mode='circular')
#         self.norm = nn.BatchNorm1d(c_in)
#         self.activation = nn.ELU()
#         self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x):
#         x = self.downConv(x.permute(0, 2, 1))  # BxExS
#         x = self.norm(x)
#         x = self.activation(x)
#         x = self.maxPool(x)
#         x = x.transpose(1, 2)
#         return x
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x = x + self.dropout(new_x)
#
#         y = x = self.norm1(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#
#         return self.norm2(x + y), attn
#
#
# class Encoder(nn.Module):
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer
#
#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         # x [B, L, D]
#         attns = []
#         if self.conv_layers is not None:
#             # The reason why we only import delta for the first attn_block of Encoder
#             # is to integrate Informer into our framework, where row size of attention of Informer is changing each layer
#             # and inconsistent to the sequence length of the initial input,
#             # then no way to add delta to every row, so we make delta=0.0 (See our Appendix E.2)
#             #
#             for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
#                 delta = delta if i == 0 else None
#                 x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
#                 x = conv_layer(x)
#                 attns.append(attn)
#             x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
#             attns.append(attn)
#         else:
#             for attn_layer in self.attn_layers:
#                 x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
#                 attns.append(attn)
#
#         if self.norm is not None:
#             x = self.norm(x)
#
#         return x, attns
#
#
# class DecoderLayer(nn.Module):
#     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
#                  dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
#         # Note that delta only used for Self-Attention(x_enc with x_enc)
#         # and Cross-Attention(x_enc with x_dec),
#         # but not suitable for Self-Attention(x_dec with x_dec)
#
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask,
#             tau=tau, delta=None
#         )[0])
#         x = self.norm1(x)
#
#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross,
#             attn_mask=cross_mask,
#             tau=tau, delta=delta
#         )[0])
#
#         y = x = self.norm2(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#
#         return self.norm3(x + y)
#
#
# class Decoder(nn.Module):
#     def __init__(self, layers, norm_layer=None, projection=None):
#         super(Decoder, self).__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer
#         self.projection = projection
#
#     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
#         for layer in self.layers:
#             x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
#
#         if self.norm is not None:
#             x = self.norm(x)
#
#         if self.projection is not None:
#             x = self.projection(x)
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np


# =================================================================================
# 1. 核心子模块 (从您提供的ns_Transformer.py中提取并适配)
# =================================================================================

class PositionalEmbedding(nn.Module):
    """为序列中的每个位置添加位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """使用1D卷积将输入通道映射到d_model维度"""

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class SimplifiedDataEmbedding(nn.Module):
    """
    简化的数据嵌入层，仅包含值嵌入和位置嵌入。
    专门用于处理原始sEMG信号，不需要时间标记x_mark。
    """

    def __init__(self, c_in, d_model, dropout=0.1):
        super(SimplifiedDataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [B, L, C]
        # 仅将值嵌入和位置嵌入相加
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DSAttention(nn.Module):
    """De-stationary Attention (非平稳注意力机制)"""

    def __init__(self, mask_flag=True, factor=5, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / sqrt(E)

        # 注入非平稳因子tau和delta
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        # 注意力计算
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A if self.output_attention else None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        d_keys = d_values = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for i, attn_layer in enumerate(self.attn_layers):
            # 仅在第一层注入delta，以保持与原始论文设计的一致性
            current_delta = delta if i == 0 else None
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=current_delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Projector(nn.Module):
    '''学习非平稳因子的MLP'''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()
        padding = 1
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = torch.cat([x, stats], dim=1)
        x = x.view(batch_size, -1)
        return self.backbone(x)


# =================================================================================
# 2. 最终封装的模型：NSTransformerEncoder
# =================================================================================

class NSTransformerEncoder(nn.Module):
    """
    使用非平稳Transformer的编码器来提取固定维度的特征。
    输入: [B, 400, 8] 的sEMG信号
    输出: [B, 128] 的特征向量
    """

    def __init__(self, in_channels, seq_len, output_features=128,
                 d_model=64, n_heads=8, e_layers=2, d_ff=128,
                 p_hidden_dims=[64, 64], p_hidden_layers=2, dropout=0.1):
        super(NSTransformerEncoder, self).__init__()

        # 1. 嵌入层
        self.embedding = SimplifiedDataEmbedding(in_channels, d_model, dropout)

        # 2. 编码器主干
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads
                    ),
                    d_model, d_ff, dropout=dropout
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 3. 非平稳因子学习器
        self.tau_learner = Projector(enc_in=in_channels, seq_len=seq_len, hidden_dims=p_hidden_dims,
                                     hidden_layers=p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=in_channels, seq_len=seq_len, hidden_dims=p_hidden_dims,
                                       hidden_layers=p_hidden_layers, output_dim=seq_len)

        # 4. 输出投影头
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(d_model, output_features)

    def forward(self, x):
        # x: [B, L, C], 形状: [batch_size, 400, 8]

        # 0. 准备原始数据和统计量
        x_raw = x.clone().detach()
        mean_enc = x_raw.mean(1, keepdim=True).detach()
        std_enc = torch.sqrt(torch.var(x_raw, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        # 1. 学习非平稳因子
        tau = self.tau_learner(x_raw, std_enc).exp()  # -> [B, 1]
        delta = self.delta_learner(x_raw, mean_enc)  # -> [B, L]

        # 2. 标准化输入
        x_enc = (x - mean_enc) / std_enc

        # 3. 数据嵌入
        enc_out = self.embedding(x_enc)  # -> [B, L, d_model]

        # 4. 通过编码器
        enc_out, _ = self.encoder(enc_out, tau=tau, delta=delta)  # -> [B, L, d_model]

        # 5. 全局池化和投影
        # [B, L, d_model] -> [B, d_model, L]
        pooled_out = self.pooling(enc_out.permute(0, 2, 1))
        # [B, d_model, 1] -> [B, d_model]
        pooled_out = pooled_out.squeeze(-1)
        # [B, d_model] -> [B, output_features]
        final_features = self.projection(pooled_out)

        return final_features


# =================================================================================
# 3. 示例用法与测试代码
# =================================================================================
if __name__ == "__main__":
    # --- 1. 定义超参数 ---
    BATCH_SIZE = 16
    SEQ_LEN = 400
    CHANNELS = 8
    OUTPUT_FEATURES = 128

    # --- 2. 实例化模型 ---
    feature_extractor = NSTransformerEncoder(
        in_channels=CHANNELS,
        seq_len=SEQ_LEN,
        output_features=OUTPUT_FEATURES,
        d_model=64,  # Transformer内部维度
        n_heads=4,  # 注意力头数
        e_layers=2,  # Encoder层数
        d_ff=256  # 前馈网络维度
    )

    print("--- 特征提取器已成功实例化 ---")

    # --- 3. 打印模型详细结构 (新增) ---
    print("\n--- 模型详细结构 ---")
    print(feature_extractor)

    # --- 4. 计算并打印模型参数量 (新增) ---
    print("\n--- 模型参数量 ---")
    total_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M")


    # --- 5. 创建模拟输入数据 ---
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, CHANNELS)
    print(f"\n--- 模拟输入数据 ---")
    print(f"输入形状: {dummy_input.shape}")

    # --- 6. 执行前向传播 ---
    try:
        extracted_features = feature_extractor(dummy_input)
        print("\n--- 前向传播成功 ---")

        # --- 7. 打印并验证输出形状 ---
        print("\n--- 验证输出形状 ---")
        print(f"最终输出特征形状: {extracted_features.shape}")

        # 使用断言进行严格检查
        expected_shape = torch.Size([BATCH_SIZE, OUTPUT_FEATURES])
        assert extracted_features.shape == expected_shape, "最终输出的形状不正确！"

        print("\n输出形状验证通过！模型工作正常。")

    except Exception as e:
        print(f"\n--- 前向传播失败 ---")
        print(f"错误信息: {e}")
        import traceback

        traceback.print_exc()