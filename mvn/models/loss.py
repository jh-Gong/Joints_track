'''
Date: 2024-11-27 13:16:25
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-17 23:12:25
Description: example
'''
import torch
from torch import nn

class KeypointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        if keypoints_binary_validity is None:
            keypoints_binary_validity = torch.ones_like(keypoints_pred)
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss
    
class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        if keypoints_binary_validity is None:
            keypoints_binary_validity = torch.ones_like(keypoints_pred)
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss
    
class KeypointsMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        if keypoints_binary_validity is None:
            keypoints_binary_validity = torch.ones_like(keypoints_pred)
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred) * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss
    
class QuaternionAngleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, quat_pred, quat_gt):
        dot_product = torch.sum(quat_pred * quat_gt, dim=-1)
        angle_diff = torch.acos(torch.clamp(dot_product, -1 + 1e-7, 1 - 1e-7)) * 2

        loss = torch.mean(angle_diff)
        return loss
    
class QuaternionChordalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, quat_pred, quat_gt):
        loss = torch.min(torch.norm(quat_pred - quat_gt, dim=-1),
                         torch.norm(quat_pred + quat_gt, dim=-1)).mean()

        return loss