'''
Date: 2024-11-27 13:16:25
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-17 23:12:25
Description: 损失函数
'''
import torch
from torch import nn

class KeypointsMSELoss(nn.Module):
    """
    计算关键点预测与真实值之间的均方误差损失。
    """
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        """
        前向传播。

        Args:
            keypoints_pred (torch.Tensor): 预测的关键点，形状为 `(batch_size, ..., dimension)`。
            keypoints_gt (torch.Tensor): 真实的关键点，形状为 `(batch_size, ..., dimension)`。
            keypoints_binary_validity (torch.Tensor, optional): 一个二进制张量，用于指示哪些关键点是有效的。形状与`keypoints_pred`相同。默认为None。

        Returns:
            torch.Tensor: 计算出的损失。
        """
        if keypoints_binary_validity is None:
            keypoints_binary_validity = torch.ones_like(keypoints_pred)
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss
    
class KeypointsMSESmoothLoss(nn.Module):
    """
    计算关键点预测与真实值之间的平滑均方误差损失。
    对于大于阈值的误差，使用幂函数进行平滑。

    Args:
        threshold (int, optional): 误差阈值。默认为400。
    """
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        """
        前向传播。

        Args:
            keypoints_pred (torch.Tensor): 预测的关键点，形状为 `(batch_size, ..., dimension)`。
            keypoints_gt (torch.Tensor): 真实的关键点，形状为 `(batch_size, ..., dimension)`。
            keypoints_binary_validity (torch.Tensor, optional): 一个二进制张量，用于指示哪些关键点是有效的。形状与`keypoints_pred`相同。默认为None。

        Returns:
            torch.Tensor: 计算出的损失。
        """
        if keypoints_binary_validity is None:
            keypoints_binary_validity = torch.ones_like(keypoints_pred)
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss
    
class KeypointsMAELoss(nn.Module):
    """
    计算关键点预测与真实值之间的平均绝对误差损失。
    """
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        """
        前向传播。

        Args:
            keypoints_pred (torch.Tensor): 预测的关键点，形状为 `(batch_size, ..., dimension)`。
            keypoints_gt (torch.Tensor): 真实的关键点，形状为 `(batch_size, ..., dimension)`。
            keypoints_binary_validity (torch.Tensor, optional): 一个二进制张量，用于指示哪些关键点是有效的。形状与`keypoints_pred`相同。默认为None。

        Returns:
            torch.Tensor: 计算出的损失。
        """
        if keypoints_binary_validity is None:
            keypoints_binary_validity = torch.ones_like(keypoints_pred)
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred) * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss
    
class QuaternionAngleLoss(nn.Module):
    """
    计算预测四元数与真实四元数之间的角度差损失。
    """
    def __init__(self):
        super().__init__()

    def forward(self, quat_pred, quat_gt):
        """
        前向传播。

        Args:
            quat_pred (torch.Tensor): 预测的四元数，形状为 `(batch_size, ..., 4)`。
            quat_gt (torch.Tensor): 真实的四元数，形状为 `(batch_size, ..., 4)`。

        Returns:
            torch.Tensor: 计算出的损失。
        """
        dot_product = torch.sum(quat_pred * quat_gt, dim=-1)
        angle_diff = torch.acos(torch.clamp(dot_product, -1 + 1e-7, 1 - 1e-7)) * 2

        loss = torch.mean(angle_diff)
        return loss
    
class QuaternionChordalLoss(nn.Module):
    """
    计算预测四元数与真实四元数之间的弦长损失。
    """
    def __init__(self):
        super().__init__()

    def forward(self, quat_pred, quat_gt):
        """
        前向传播。

        Args:
            quat_pred (torch.Tensor): 预测的四元数，形状为 `(batch_size, ..., 4)`。
            quat_gt (torch.Tensor): 真实的四元数，形状为 `(batch_size, ..., 4)`。

        Returns:
            torch.Tensor: 计算出的损失。
        """
        loss = torch.min(torch.norm(quat_pred - quat_gt, dim=-1),
                         torch.norm(quat_pred + quat_gt, dim=-1)).mean()

        return loss
