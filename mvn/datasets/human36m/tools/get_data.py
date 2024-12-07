import numpy as np
import joblib
import os
import h5py
from tqdm import tqdm

def get_data(input_file, output_file):
    try:
        data = joblib.load(input_file)
        print(f"file:{input_file} Load done!")
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
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

    with h5py.File(output_file, 'w') as h5file:
        for item in tqdm(data, desc=f"Processing {input_file}"):
            subject = item['subject']
            action = item['action']
            subaction = item['subaction']
            video_id = item['video_id']

            # 创建多层嵌套组结构
            group_path = f"{subject}/{action}/{subaction}/{video_id}"
            group = h5file.require_group(group_path)

            # 添加 joints_3d 和 joints_vis 数据
            joints_3d = np.array(item['joints_3d'][:17]).flatten()
            joints_vis = np.array(item['joints_vis'][:17]).flatten()

            length = joints_3d.shape[0]

            if 'joints_3d' not in group:
                group.create_dataset('joints_3d', data=joints_3d[None, :], maxshape=(None, length), chunks=True)
            else:
                group['joints_3d'].resize((group['joints_3d'].shape[0] + 1, length))
                group['joints_3d'][-1] = joints_3d

            if 'joints_vis' not in group:
                group.create_dataset('joints_vis', data=joints_vis[None, :], maxshape=(None, length), chunks=True)
            else:
                group['joints_vis'].resize((group['joints_vis'].shape[0] + 1, length))
                group['joints_vis'][-1] = joints_vis

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    input_file_train = os.path.join(current_directory, '..', "source", f"h36m_train.pkl")
    output_file_train = os.path.join(current_directory, '..', "results", f"data_train-origin.h5")
    input_file_val = os.path.join(current_directory, '..', "source", f"h36m_validation.pkl")
    output_file_val = os.path.join(current_directory, '..', "results", f"data_validation-origin.h5")
    
    for input_file, output_file in [(input_file_train, output_file_train), (input_file_val, output_file_val)]:
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
            continue
        
        get_data(input_file, output_file)