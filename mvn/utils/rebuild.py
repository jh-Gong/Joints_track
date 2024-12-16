'''
Date: 2024-12-15 21:26:29
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-16 01:50:01
Description: 三维重建模块
'''

import torch

def rebuild_pose_from_root(root, rotations, bone_lengths):
    parent_map = {
        1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5,
        7: 0, 8: 7, 9: 8, 10: 9, 14: 8, 15: 14, 16: 15, 11: 8, 12: 11, 13: 12
    }
    batch_size = root.shape[0]
    seq_len = root.shape[1]

    joint_positions_all_batch = []

    for b in range(batch_size):  
        joint_positions_all_seq = []
        for i in range(seq_len):
            root_position = root[b, i, :].squeeze()  
            rotation = rotations[b, i, :].reshape(16, 4)  
            bone_length = bone_lengths[b].squeeze()
            joint_positions = calculate_joint_positions(root_position, rotation, bone_length, parent_map)
            joint_positions_all_seq.append(joint_positions)
        joint_positions_all_batch.append(torch.stack(joint_positions_all_seq))

    return torch.stack(joint_positions_all_batch)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with shape (..., 4)

    Returns:
        Rotation matrices with shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def rotate_vector_by_quaternion(vector, quaternion):
    """
    Rotates a vector by a quaternion.

    Args:
        vector: A tensor of shape (3,) representing the vector.
        quaternion: A tensor of shape (4,) representing the quaternion.

    Returns:
        A tensor of shape (3,) representing the rotated vector.
    """
    # Convert the vector to a quaternion with a scalar part of 0.
    vector_quaternion = torch.cat([torch.tensor([0.0]).to(vector.device), vector])

    # Calculate the conjugate of the quaternion.
    quaternion_conjugate = torch.tensor([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]).to(vector.device)

    # Perform quaternion multiplication: q * v * q'
    rotated_quaternion = quaternion_multiply(quaternion_multiply(quaternion, vector_quaternion), quaternion_conjugate)

    # Return the vector part of the rotated quaternion.
    return rotated_quaternion[1:]

def quaternion_multiply(q1, q2):
    """
    Performs quaternion multiplication.

    Args:
        q1: A tensor of shape (4,) representing the first quaternion.
        q2: A tensor of shape (4,) representing the second quaternion.

    Returns:
        A tensor of shape (4,) representing the resulting quaternion.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.tensor([w, x, y, z]).to(q1.device)

def calculate_joint_positions(root_position, rotations, bone_lengths, parent_map):
    """
    Calculates the 3D coordinates of the joints based on root position, rotations, and bone lengths.

    Args:
        root_position: A tensor of shape (3,) representing the root position.
        rotations: A tensor of shape (16, 4) representing the quaternions for each joint.
        bone_lengths: A tensor of shape (16,) representing the lengths of each bone.
        parent_map: A dictionary mapping joint indices to their parent indices.

    Returns:
        A tensor of shape (17, 3) representing the 3D coordinates of the joints.
    """
    joint_positions = torch.zeros((17, 3)).to(root_position.device)
    joint_positions[0] = root_position

    for i in range(1, 17):
        parent_index = parent_map[i]
        parent_joint_position = joint_positions[parent_index]
        
        # Calculate the relative rotation from parent to current joint
        if parent_index == 0:
            relative_rotation = rotations[i-1]
        else:
            relative_rotation = rotations[i -1 ]

        bone_direction = rotate_vector_by_quaternion(torch.tensor([1.0, 0.0, 0.0]).to(root_position.device), relative_rotation)
        current_bone_length = bone_lengths[i-1]
        joint_positions[i] = parent_joint_position + bone_direction * current_bone_length

    return joint_positions
