import torch
import torch.nn as nn


class IterativeDynamicSelector(nn.Module):
    """
    Iterative Dynamic Selector (IDS)
    迭代动态选择器，用于筛选特征最明显的点
    """

    def __init__(self, feature_dim=64, num_selected=512, num_iterations=3):
        super(IterativeDynamicSelector, self).__init__()

        self.feature_dim = feature_dim
        self.num_selected = num_selected
        self.num_iterations = num_iterations

        # 特征重要性评分网络
        self.importance_net = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(feature_dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 特征增强网络
        self.enhance_net = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim * 2, 1),
            nn.GELU(),
            nn.Conv1d(feature_dim * 2, feature_dim, 1)
        )

    def forward(self, features, coordinates):
        batch_size, _, num_points = features.shape
        current_features = features

        for i in range(self.num_iterations):
            # 计算特征重要性
            importance_scores = self.importance_net(current_features)

            # 选择重要性最高的点
            _, indices = torch.topk(importance_scores.squeeze(1),
                                    self.num_selected, dim=1)

            # 收集选中的特征
            batch_indices = torch.arange(batch_size).view(-1, 1).to(features.device)
            selected_features = current_features[batch_indices, :, indices]

            # 如果不是最后一次迭代，则增强特征
            if i < self.num_iterations - 1:
                enhanced_features = self.enhance_net(selected_features)
                current_features = self.propagate_features(
                    current_features, indices, enhanced_features, coordinates
                )

        return selected_features, indices

    def propagate_features(self, original_features, indices, enhanced_features, coordinates):
        batch_size, feature_dim, num_points = original_features.shape
        num_selected = indices.shape[1]

        # 计算选中的点与所有点之间的距离
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).to(original_features.device)
        selected_coords = coordinates[batch_indices, :, indices]

        # 重塑坐标以便计算距离
        original_coords_reshaped = coordinates.permute(0, 2, 1)
        selected_coords_reshaped = selected_coords.permute(0, 2, 1)

        # 计算所有点与选中点之间的距离
        distances = torch.cdist(original_coords_reshaped, selected_coords_reshaped)

        # 计算权重（距离越近权重越大）
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(dim=2, keepdim=True)

        # 将增强特征传播回所有点
        enhanced_features_reshaped = enhanced_features.permute(0, 2, 1)
        propagated_features = torch.bmm(weights, enhanced_features_reshaped)
        propagated_features = propagated_features.permute(0, 2, 1)

        # 结合原始特征和传播的特征
        alpha = 0.5
        updated_features = alpha * original_features + (1 - alpha) * propagated_features

        return updated_features