import torch
import torch.nn as nn
import torch.nn.functional as F


class EGPB(nn.Module):
    """
    Enhanced Geometric Processing Block (EGPB)
    结合MLP与自注意力机制的几何特征增强模块
    """

    def __init__(self, in_channels=64, hidden_channels=128, out_channels=64,
                 num_heads=4, dropout=0.1, use_skip_connection=True):
        super(EGPB, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_skip_connection = use_skip_connection

        # MLP分支 - 局部特征提取
        self.mlp_branch = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, out_channels, 1)
        )

        # 自注意力分支 - 全局关系建模
        self.self_attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # 使用 (seq_len, batch, features) 格式
        )

        # 注意力后的前馈网络
        self.attention_ffn = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(dropout)
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # 输入输出投影（如果通道数不匹配）
        if in_channels != out_channels:
            self.input_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.input_proj = nn.Identity()

        # 输出层（可选的特征增强）
        self.output_enhance = nn.Conv1d(out_channels, out_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_encoding=None):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, channels, num_points]
            pos_encoding: 位置编码 [batch_size, channels, num_points] 或 None

        Returns:
            enhanced_features: 增强后的特征 [batch_size, out_channels, num_points]
        """
        batch_size, _, num_points = x.shape

        # 1. MLP分支 - 局部特征增强
        mlp_features = self.mlp_branch(x)  # [B, out_channels, N]

        # 2. 自注意力分支 - 全局关系建模
        # 重塑为序列格式 [seq_len, batch_size, features]
        seq_features = mlp_features.permute(2, 0, 1)  # [N, B, C]

        # 添加位置编码（如果提供）
        if pos_encoding is not None:
            pos_seq = pos_encoding.permute(2, 0, 1)
            seq_features = seq_features + pos_seq

        # 自注意力计算
        attended_features, attention_weights = self.self_attention(
            query=seq_features,
            key=seq_features,
            value=seq_features
        )

        # 残差连接和归一化
        seq_features = self.norm1(seq_features + self.dropout(attended_features))

        # 前馈网络
        ffn_output = self.attention_ffn(seq_features)
        seq_features = self.norm2(seq_features + ffn_output)

        # 恢复原始形状 [batch_size, channels, num_points]
        attention_features = seq_features.permute(1, 2, 0)

        # 3. 特征融合
        if self.use_skip_connection:
            # 残差连接
            input_projected = self.input_proj(x)
            enhanced_features = input_projected + attention_features
        else:
            enhanced_features = attention_features

        # 最终输出增强
        output = self.output_enhance(enhanced_features)

        return output

    def get_attention_weights(self, x, pos_encoding=None):
        """
        获取注意力权重用于可视化或分析
        """
        batch_size, _, num_points = x.shape

        # MLP特征提取
        mlp_features = self.mlp_branch(x)
        seq_features = mlp_features.permute(2, 0, 1)

        if pos_encoding is not None:
            pos_seq = pos_encoding.permute(2, 0, 1)
            seq_features = seq_features + pos_seq

        # 只返回注意力权重
        _, attention_weights = self.self_attention(
            query=seq_features,
            key=seq_features,
            value=seq_features,
            need_weights=True
        )

        return attention_weights


class EGPBWithCoordinate(nn.Module):
    """
    带有坐标信息的EGPB变体，适用于点云处理
    """

    def __init__(self, feature_channels=64, coord_channels=3, hidden_channels=128,
                 num_heads=4, dropout=0.1):
        super(EGPBWithCoordinate, self).__init__()

        # 坐标编码网络
        self.coord_encoder = nn.Sequential(
            nn.Conv1d(coord_channels, feature_channels // 2, 1),
            nn.GELU(),
            nn.Conv1d(feature_channels // 2, feature_channels, 1)
        )

        # EGPB模块
        self.egpb = EGPB(
            in_channels=feature_channels * 2,  # 特征 + 坐标编码
            hidden_channels=hidden_channels,
            out_channels=feature_channels,
            num_heads=num_heads,
            dropout=dropout
        )

        # 输出层
        self.output_proj = nn.Conv1d(feature_channels, feature_channels, 1)

    def forward(self, features, coordinates):
        """
        Args:
            features: 输入特征 [B, C, N]
            coordinates: 点云坐标 [B, 3, N]
        """
        # 编码坐标信息
        coord_features = self.coord_encoder(coordinates)

        # 拼接特征和坐标编码
        combined_features = torch.cat([features, coord_features], dim=1)

        # 通过EGPB模块
        enhanced_features = self.egpb(combined_features, pos_encoding=coord_features)

        # 输出投影
        output = self.output_proj(enhanced_features)

        return output



if __name__ == "__main__":
    # 创建EGPB模块
    egpb = EGPB(in_channels=64, hidden_channels=128, out_channels=64)

    # 测试输入
    batch_size, channels, num_points = 4, 64, 1024
    test_input = torch.randn(batch_size, channels, num_points)

    # 前向传播
    output = egpb(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")

    # 获取注意力权重
    attention_weights = egpb.get_attention_weights(test_input)
    print(f"注意力权重形状: {attention_weights.shape}")