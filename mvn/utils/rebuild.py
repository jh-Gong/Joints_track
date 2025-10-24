'''
Date: 2024-12-15 21:26:29
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-17 17:32:14
Description: 三维重建模块
'''

import torch

def rebuild_pose_from_root(root: torch.Tensor, rotations: torch.Tensor, bone_lengths: torch.Tensor) -> torch.Tensor:
    """
    根据根节点位置、旋转和骨骼长度重建姿态。

    Args:
        root (torch.Tensor): 根节点位置，形状为 `(batch_size, seq_len, 3)`。
        rotations (torch.Tensor): 每个关节的旋转四元数，形状为 `(batch_size, seq_len, 16*4)`。
        bone_lengths (torch.Tensor): 每个骨骼的长度，形状为 `(batch_size, 16)`。

    Returns:
        torch.Tensor: 重建后的关节位置，形状为 `(batch_size, seq_len, 17, 3)`。
    """
    parent_map = {
        1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5,
        7: 0, 8: 7, 9: 8, 10: 9, 14: 8, 15: 14, 16: 15, 11: 8, 12: 11, 13: 12
    }
    batch_size = root.shape[0]
    seq_len = root.shape[1]

    # 将旋转四元数 reshape 为 (batch_size, seq_len, 16, 4)
    rotations = rotations.reshape(batch_size, seq_len, 16, 4)
    # 将骨骼长度扩展维度为 (batch_size, 1, 16)
    bone_lengths = bone_lengths.unsqueeze(1)

    # 计算关节位置，形状为 (batch_size, seq_len, 17, 3)
    joint_positions = calculate_joint_positions_vectorized(root, rotations, bone_lengths, parent_map)

    return joint_positions

def rotate_vector_by_quaternion_vectorized(vector: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """
    批量旋转向量。

    Args:
        vector (torch.Tensor): 待旋转的向量，形状为 `(..., 3)`。
        quaternion (torch.Tensor): 旋转四元数，形状为 `(..., 4)`。

    Returns:
        torch.Tensor: 旋转后的向量，形状为 `(..., 3)`。
    """
    # 将向量转换为标量部分为 0 的四元数。
    vector_quaternion = torch.cat([torch.zeros_like(vector[..., :1]), vector], dim=-1)

    # 计算四元数的共轭。
    quaternion_conjugate = torch.cat([quaternion[..., :1], -quaternion[..., 1:]], dim=-1)

    # 执行四元数乘法：q * v * q'
    rotated_quaternion = quaternion_multiply_vectorized(quaternion_multiply_vectorized(quaternion, vector_quaternion), quaternion_conjugate)

    # 返回旋转后四元数的向量部分。
    return rotated_quaternion[..., 1:]

def quaternion_multiply_vectorized(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    批量执行四元数乘法。

    Args:
        q1 (torch.Tensor): 第一个四元数，形状为 `(..., 4)`。
        q2 (torch.Tensor): 第二个四元数，形状为 `(..., 4)`。

    Returns:
        torch.Tensor: 四元数乘法的结果，形状为 `(..., 4)`。
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def calculate_joint_positions_vectorized(root_position: torch.Tensor, rotations: torch.Tensor, bone_lengths: torch.Tensor, parent_map: dict) -> torch.Tensor:
    """
    向量化计算基于根节点位置、旋转和骨骼长度的关节三维坐标。

    Args:
        root_position (torch.Tensor): 根节点位置，形状为 `(batch_size, seq_len, 3)`。
        rotations (torch.Tensor): 每个关节的旋转四元数，形状为 `(batch_size, seq_len, 16, 4)`。
        bone_lengths (torch.Tensor): 每个骨骼的长度，形状为 `(batch_size, 1, 16)`。
        parent_map (dict): 将关节索引映射到其父关节索引的字典。

    Returns:
        torch.Tensor: 关节的三维坐标，形状为 `(batch_size, seq_len, 17, 3)`。
    """
    batch_size, seq_len, _ = root_position.shape
    device = root_position.device

    # 初始化关节位置张量，形状为 (batch_size, seq_len, 17, 3)
    joint_positions = torch.zeros((batch_size, seq_len, 17, 3), device=device)
    joint_positions[:, :, 0, :] = root_position

    # 计算骨骼方向，形状为 (batch_size, seq_len, 16, 3)
    bone_directions = rotate_vector_by_quaternion_vectorized(torch.tensor([1.0, 0.0, 0.0], device=device).expand(batch_size, seq_len, 16, 3), rotations)
    # 计算骨骼向量，形状为 (batch_size, seq_len, 16, 3)
    bone_vectors = bone_directions * bone_lengths.unsqueeze(-1)

    for i in range(1, 17):
        parent_index = parent_map[i]
        joint_positions[:, :, i, :] = joint_positions[:, :, parent_index, :] + bone_vectors[:, :, i - 1, :]

    return joint_positions
