r'''
Author: gjhhh 1377019164@qq.com
Date: 2024-12-12 20:44:20
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-12 20:44:48
FilePath: \Joint_track\mvn\datasets\human36m\utils\process.py
Description: 对hdf5格式的数据集的处理函数
'''
import numpy as np
import h5py
from tqdm import tqdm
import quaternion

def process_data(input_file, output_file):
    hdf5_data = {}

    with h5py.File(input_file, 'r') as h5file:
        subjects = list(h5file.keys())
        
        with h5py.File(output_file, 'w') as outfile: 
            for subject in tqdm(subjects, desc="Subjects"):
                if subject not in hdf5_data:
                    hdf5_data[subject] = {}
                subject_group = h5file[subject]

                for action in tqdm(subject_group.keys(), desc=f"Actions in {subject}", leave=False):
                    if action not in hdf5_data[subject]:
                        hdf5_data[subject][action] = {}
                    action_group = subject_group[action]

                    for subaction in tqdm(action_group.keys(), desc=f"Subactions in {subject}/{action}", leave=False):
                        if subaction not in hdf5_data[subject][action]:
                            hdf5_data[subject][action][subaction] = {}
                        subaction_group = action_group[subaction]

                        for video_id in tqdm(subaction_group.keys(), desc=f"Videos in {subject}/{action}/{subaction}", leave=False):
                            if video_id not in hdf5_data[subject][action][subaction]:
                                hdf5_data[subject][action][subaction][video_id] = {}
                            video_group = subaction_group[video_id]

                            joints_3d = video_group['joints_3d'][:]

                            hdf5_data[subject][action][subaction][video_id]['joints_3d'] = joints_3d
                            
                            # 创建 video 组
                            outfile_video = outfile.create_group(f"{subject}/{action}/{subaction}/{video_id}")

                            bone_data, _ = calculate_bone_data(joints_3d)

                            outfile_video.create_dataset('root', data=bone_data["root"], dtype='float32')
                            outfile_video.create_dataset('rotations', data=bone_data["rotations"], dtype='float32')
                            outfile_video.create_dataset('bone_lengths', data=bone_data["bone_lengths"], dtype='float32')


def calculate_bone_data(joints_3d):
    """
    计算根节点、四元数和骨骼长度数据。

    Args:
        joints_3d: numpy 数组，形状为 (num_frames, num_joints, 3)，表示 3D 关节坐标。

    Returns:
        一个字典，包含以下键值对：
        - root: numpy 数组，形状为 (num_frames, 3)，表示根节点坐标。
        - rotations: numpy 数组，形状为 (num_frames, num_joints - 1, 4)，表示四元数旋转。
        - bone_lengths: numpy 数组，形状为 (num_frames, num_joints - 1)，表示骨骼长度。
        一个重建参数
        root_offset: 第一帧的根节点坐标，用于恢复原始根节点位置
    """

    num_frames = joints_3d.shape[0]
    num_joints = joints_3d.shape[1]

    # 定义父节点字典，方便查找
    parent_map = {
        1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5,
        7: 0, 8: 7, 9: 8, 10: 9, 14: 8, 15: 14, 16: 15, 11: 8, 12: 11, 13: 12
    }

    # 1. 生成并归一化根节点数据集
    root_data = joints_3d[:, 0, :]
    root_offset = root_data[0:1, :]  
    root_data = root_data - root_offset

    # 2. 生成关节旋转信息 (相对于父节点, 使用四元数)
    rotations = np.zeros((num_frames, num_joints - 1, 4), dtype='float32')
    # 定义一个参考向量
    ref_vector = np.array([1, 0, 0])
    for f in range(num_frames):
        for j in range(1, num_joints):  # 跳过根节点
            # 计算关节到父节点的向量
            joint_vector = joints_3d[f, j, :] - joints_3d[f, parent_map[j], :]
            # 归一化
            joint_vector = joint_vector / np.linalg.norm(joint_vector) if np.linalg.norm(joint_vector) != 0 else joint_vector

            # 计算旋转四元数
            if np.all(joint_vector == 0):
                # 如果向量为零向量，则设置为单位四元数
                rot_quat = quaternion.quaternion(1, 0, 0, 0)
            else:
                # 计算旋转四元数，使得参考向量旋转到关节向量
                rot_axis = np.cross(ref_vector, joint_vector)
                if np.all(rot_axis == 0):
                    # 如果参考向量和关节向量平行，则旋转角度为0或180度
                    if np.dot(ref_vector, joint_vector) > 0:
                        rot_quat = quaternion.quaternion(1, 0, 0, 0)
                    else:
                        rot_quat = quaternion.quaternion(0, 0, 1, 0)
                else:
                    rot_angle = np.arccos(np.dot(ref_vector, joint_vector))
                    rot_axis = rot_axis / np.linalg.norm(rot_axis)  # 归一化旋转轴
                    rot_quat = quaternion.quaternion(np.cos(rot_angle / 2),
                                                    rot_axis[0] * np.sin(rot_angle / 2),
                                                    rot_axis[1] * np.sin(rot_angle / 2),
                                                    rot_axis[2] * np.sin(rot_angle / 2))
            rotations[f, j - 1, :] = [rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z]

    # 3. 生成骨骼长度数据集
    bone_lengths = np.zeros((num_frames, len(parent_map)), dtype='float32')
    for f in range(num_frames):
        for child, parent in parent_map.items():
            bone_lengths[f, child - 1] = np.linalg.norm(joints_3d[f, child, :] - joints_3d[f, parent, :])

    return {
        "root": root_data,
        "rotations": rotations,
        "bone_lengths": bone_lengths
    }, root_offset
