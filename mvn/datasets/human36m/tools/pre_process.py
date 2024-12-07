import os
import numpy as np
import h5py
from tqdm import tqdm

def process_data(input_file, output_file, mode):
    hdf5_data = {}

    try:
        with h5py.File(input_file, 'r') as h5file:
            subjects = list(h5file.keys())
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
                            joints_vis = video_group['joints_vis'][:]

                            hdf5_data[subject][action][subaction][video_id]['joints_3d'] = joints_3d
                            hdf5_data[subject][action][subaction][video_id]['joints_vis'] = joints_vis
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    # 收集所有数据以计算全局统计量
    all_coordinates = []

    for subject in hdf5_data.values():
        for action in subject.values():
            for subaction in action.values():
                for video in subaction.values():
                    all_coordinates.append(video['joints_3d'])

    all_coordinates = np.concatenate(all_coordinates, axis=0)

    scaling_info = {}

    if mode == "norm":
        scaling_info["mode"] = "norm"
        global_mean = np.mean(all_coordinates, axis=0)
        global_std = np.std(all_coordinates, axis=0)
        global_std[global_std == 0] = 1  # 避免除以零
        scaling_info["mean"] = global_mean
        scaling_info["std"] = global_std
    
    elif mode == "linear":
        scaling_info["mode"] = "linear"
        global_min = np.min(all_coordinates, axis=0)
        global_max = np.max(all_coordinates, axis=0)
        global_range = global_max - global_min
        global_range[global_range == 0] = 1  # 避免除以零
        scaling_info["min"] = global_min
        scaling_info["max"] = global_max
    
    else:
        raise ValueError("Invalid mode. Choose 'norm' or 'linear'.")

    try:
        with h5py.File(output_file, 'w') as h5file:
            for subject, actions in tqdm(hdf5_data.items(), desc="Saving data"):
                for action, subactions in actions.items():
                    for subaction, videos in subactions.items():
                        for video_id, video_data in videos.items():
                            grp = h5file.create_group(f"{subject}/{action}/{subaction}/{video_id}")
                            if mode == "norm":
                                normalized = (video_data['joints_3d'] - global_mean) / global_std
                                grp.create_dataset('joints_3d', data=normalized)
                            elif mode == "linear":
                                scaled = 2 * (video_data['joints_3d'] - global_min) / global_range - 1
                                grp.create_dataset('joints_3d', data=scaled)
                            grp.create_dataset('joints_vis', data=video_data['joints_vis'])

            # 添加缩放信息到 HDF5 文件中
            scaling_group = h5file.create_group("scaling_info")
            for key, value in scaling_info.items():
                scaling_group.create_dataset(key, data=value)
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

if __name__ == "__main__":
    mode = "linear"
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    
    input_file_train = os.path.join(current_directory, '..', "results", "data_train-origin.h5")
    output_file_train = os.path.join(current_directory, '..', "results", f"data_train-{mode}.h5")
    input_file_val = os.path.join(current_directory, '..', "results", "data_validation-origin.h5")
    output_file_val = os.path.join(current_directory, '..', "results", f"data_validation-{mode}.h5")
    
    for input_file, output_file in [(input_file_train, output_file_train), (input_file_val, output_file_val)]:
        if not os.path.exists(input_file):
            print(f"Input file {input_file} does not exist.")
            continue
        
        process_data(input_file, output_file, mode)
