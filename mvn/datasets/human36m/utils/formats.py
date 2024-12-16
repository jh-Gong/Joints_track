r'''
Author: gjhhh 1377019164@qq.com
Date: 2024-12-12 16:26:14
LastEditors: gjhhh 1377019164@qq.com
LastEditTime: 2024-12-12 19:54:48
FilePath: \Joint_track\mvn\datasets\human36m\utils\formats.py
Description: human36m 的3D坐标标注数据集的不同格式的转换
'''

import pandas as pd
import h5py
import os
import numpy as np
from typing import List
from tqdm import tqdm
import joblib


def get_data_from_csv_to_hdf5(data_path: str, output_path: str, multi_source: bool) -> None:
    """
    将CSV文件中的数据读取并转换为HDF5格式文件。

    参数:
    - data_path (str): CSV文件的路径。如果multi_source为True，则应提供存放文件源的目录，保证文件格式相同，命名按次序排列。
    - output_path (str): 转换后的HDF5文件的保存路径。
    - multi_source (bool): 指示是否从多个数据源读取CSV数据。
    """
    if multi_source:
        data_sources = get_path_from_directory(data_path, ext='.csv')
    else:
        data_sources = [data_path]

    with h5py.File(output_path, 'w') as f:
        for data_source in tqdm(data_sources, desc=f"Processing csv files"):
            df = pd.read_csv(data_source)

            for (subject, action, subaction, camera), group in df.groupby(['subject', 'action', 'subaction', 'camera']):
                group_path = f"/{subject}/{action}/{subaction}/{camera}"

                joints_data_with_frame = {}
                for _, row in group.iterrows():
                    frame = str(int(row['frame']))
                    joints_data = []
                    for i in range(17):
                        for coord in ['x', 'y', 'z']:
                            col_name = f'joint_{i}_{coord}'
                            joints_data.append(row[col_name])  
                    joints_data_with_frame[frame] = joints_data
                

                # 将字典转换为列表，并根据 frame 排序
                sorted_frames = sorted(joints_data_with_frame.keys(), key=int)
                joints_3d_data = [joints_data_with_frame[frame] for frame in sorted_frames]
                joints_3d_array = np.array(joints_3d_data).reshape((len(sorted_frames), 17, 3))

                if group_path not in f:
                    grp = f.create_group(group_path)
                else:
                    grp = f[group_path]

                if "joints_3d" not in grp:
                    grp.create_dataset("joints_3d", data=joints_3d_array, dtype='float32', chunks=True, maxshape=(None, 17, 3))
                else:
                    old_len = grp["joints_3d"].shape[0]
                    grp["joints_3d"].resize((old_len + len(sorted_frames), 17, 3))
                    if sorted_frames[0] == 1:
                        old_data = grp["joints_3d"][:]  
                        grp["joints_3d"][0:len(joints_3d_array)] = joints_3d_array
                        grp["joints_3d"][len(joints_3d_array):] = old_data
                    else:
                        grp["joints_3d"][old_len:] = joints_3d_array

def get_data_from_pkl_to_hdf5(data_path: str, output_path: str, multi_source: bool) -> None:
    """
    将数据从pickle格式文件加载并转换到HDF5格式文件中。

    参数:
    data_path (str): pickle数据文件的路径。
    output_path (str): 输出的HDF5文件路径。
    multi_source (bool): 是否多源数据。
    """
    try:
        data = joblib.load(data_path)
        print(f"file:{data_path} Load done!")
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return

    """data[i]:
    image='s_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_000001.jpg'
    joints_3d=array([[-176.73076784, -321.04861816, 5203.88206303],
       ...(共计17个)
       [  13.89246482, -279.85293245, 5421.06854165]])
    joints_vis=array([[1., 1., 1., ..., 1., 1., 1.],
       ...(共计17个)
       [1., 1., 1., ..., 1., 1., 1.]])
    video_id=0
    image_id=0
    subject=0
    action=0
    subaction=0

    """

    with h5py.File(output_path, 'w') as h5file:
        for item in tqdm(data, desc=f"Processing {data_path}"):
            subject = item['subject']
            action = item['action']
            subaction = item['subaction']
            video_id =  ['video_id']

            # 创建多层嵌套组结构
            group_path = f"{subject}/{action}/{subaction}/{video_id}"
            group = h5file.require_group(group_path)

            # 添加 joints_3d 数据
            joints_3d = np.array(item['joints_3d'][:17]).flatten()

            length = joints_3d.shape[0]

            if 'joints_3d' not in group:
                group.create_dataset('joints_3d', data=joints_3d[None, :], maxshape=(None, length), chunks=True)
            else:
                group['joints_3d'].resize((group['joints_3d'].shape[0] + 1, length))
                group['joints_3d'][-1] = joints_3d


def get_path_from_directory(directory_path: str, ext: str) -> List[str]:
    """
    获取指定目录下的所有文件的路径。

    参数:
    - directory_path (str): 指定目录的路径。
    - ext (str): 文件的扩展名。

    返回:
    - List[str]: 指定目录下的所有文件的路径列表。
    """
    if not ext.startswith('.'):
        ext = '.' + ext
    
    paths = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(ext):
                full_path = os.path.join(root, file)
                paths.append(full_path)
    
    return paths
