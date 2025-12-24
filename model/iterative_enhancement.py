import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """交叉注意力层"""

    def __init__(self, d_model=256, nhead=4, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None):
        # 交叉注意力
        cross_attn_output, cross_attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attn_mask
        )

        # 残差连接和归一化
        output = self.norm(query + self.dropout(cross_attn_output))

        return output, cross_attn_weights


class IterativeEnhancementPipeline(nn.Module):
    """
    Iterative Enhancement Pipeline (IEP)
    迭代增强管道，由交叉注意力和EGPB组成
    """

    def __init__(self, d_model=256, nhead=4, egpb_channels=64, num_iterations=3, dropout=0.1):
        super(IterativeEnhancementPipeline, self).__init__()

        self.d_model = d_model
        self.num_iterations = num_iterations

        # 交叉注意力层
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dropout)
            for _ in range(num_iterations)
        ])

        # EGPB层（简化版，实际使用时需要导入完整的EGPB）
        self.egpb_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, egpb_channels, 1),
                nn.GELU(),
                nn.Conv1d(egpb_channels, d_model, 1)
            ) for _ in range(num_iterations)
        ])

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model, 1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1)
        )

    def forward(self, mhcpb_output, f_11):
        """
        前向传播

        Args:
            mhcpb_output: MHCPB模块的输出 [B, N_mhcpb, C]
            f_11: input_cropped1经过IDS生成的变量 [B, N_f11, C]

        Returns:
            enhanced_output: 增强后的输出 [B, N_mhcpb, C]
        """
        batch_size, num_points_mhcpb, feature_dim = mhcpb_output.shape
        _, num_points_f11, _ = f_11.shape

        # 确保特征维度一致
        if mhcpb_output.shape[-1] != self.d_model:
            mhcpb_output = nn.Linear(mhcpb_output.shape[-1], self.d_model).to(mhcpb_output.device)(mhcpb_output)
        if f_11.shape[-1] != self.d_model:
            f_11 = nn.Linear(f_11.shape[-1], self.d_model).to(f_11.device)(f_11)

        # 重塑为序列格式
        current_query = mhcpb_output.permute(1, 0, 2)  # [N_mhcpb, B, C]
        key_value = f_11.permute(1, 0, 2)  # [N_f11, B, C]

        # 迭代增强
        for i in range(self.num_iterations):
            # 交叉注意力
            current_query, attn_weights = self.cross_attention_layers[i](
                current_query, key_value
            )

            # EGPB处理（简化的实现）
            egpb_input = current_query.permute(1, 2, 0)  # [B, C, N_mhcpb]
            egpb_output = self.egpb_layers[i](egpb_input)  # [B, C, N_mhcpb]
            egpb_output = egpb_output.permute(2, 0, 1)  # [N_mhcpb, B, C]

            # 残差连接
            current_query = current_query + egpb_output

        # 恢复原始形状
        enhanced_output = current_query.permute(1, 0, 2)  # [B, N_mhcpb, C]

        return enhanced_output