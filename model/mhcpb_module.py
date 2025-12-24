import torch
import torch.nn as nn


class MHCPB(nn.Module):
    """
    Multi-Head Cross and Self Attention with MLP Block (MHCPB)
    结合交叉注意力、自注意力和MLP的多头注意力模块
    """

    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1):
        super(MHCPB, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        # MLP分支
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query, key_value, self_attn_mask=None, cross_attn_mask=None):
        # 交叉注意力
        cross_attn_output, cross_attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=cross_attn_mask
        )

        # 残差连接和归一化
        query = self.norm1(query + self.dropout1(cross_attn_output))

        # 自注意力
        self_attn_output, self_attn_weights = self.self_attention(
            query=query,
            key=query,
            value=query,
            attn_mask=self_attn_mask
        )

        # 残差连接和归一化
        query = self.norm2(query + self.dropout2(self_attn_output))

        # MLP处理
        mlp_output = self.mlp(query)

        # 残差连接和归一化
        output = self.norm3(query + self.dropout3(mlp_output))

        return output, cross_attn_weights, self_attn_weights